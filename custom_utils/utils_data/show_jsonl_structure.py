from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Literal

from tqdm.auto import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _preprocess_underscore_args(argv: list[str]) -> list[str]:
    """Replace underscores with hyphens in ``--arg_name`` / ``--arg_name=val`` prefixes.

    argparse normalises ``-`` and ``_`` internally, so ``--output-path`` and
    ``--output_path`` would normally map to different destinations.  This
    function rewrites ``_`` to ``-`` in the leading ``--key`` portion so that
    argparse treats both spellings identically.
    """
    out: list[str] = []
    for arg in argv:
        if arg.startswith("--") and "=" in arg:
            key, _, val = arg.partition("=")
            arg = key.replace("_", "-") + "=" + val
        elif arg.startswith("--"):
            arg = arg.replace("_", "-")
        out.append(arg)
    return out


class Structure:
    """Inferred structure of a JSON-compatible Python value.

    `kind` is one of:
        "empty", "none", "bool", "int", "float", "str", "list", "dict"

    For lists and dicts:
        `value_structure` is the merged structure of all contained values.

    For numeric structures:
        `numeric_string_sources` counts values that were numeric-looking strings.
        `numeric_native_sources` counts values that were actual JSON numbers.

    Those counts let `numeric_provenance` distinguish:
        "none" -> no numeric values came from strings
        "x%"   -> x% did
        "all"  -> all did
    """

    NUMERIC_KINDS = {"int", "float"}
    CONTAINER_KINDS = {"list", "dict"}

    def __init__(self, data: Any):
        self.kind: Literal["empty", "none", "bool", "int", "float", "str", "list", "dict"]
        self.nullable: bool = False

        # Used only with CONTAINER_KINDS
        self.value_structure: Structure = Structure.create_empty()
        # Design: self.value_structure is None only if self.kind is "empty" 
        #         otherwise is an "empty" Structure
        self.key_structures: dict[str, Structure] = {}
        self.optional_keys: set[str] = set()
        self.empty_sources: int = 0
        self.container_sources: int = 0
        self.numeric_string_sources: int = 0
        self.numeric_native_sources:int  = 0

        if data is None:
            self.kind = "none"

        elif isinstance(data, bool):
            self.kind = "bool"

        elif isinstance(data, int):
            self.kind = "int"
            self.numeric_native_sources = 1

        elif isinstance(data, float):
            self.kind = "float"
            self.numeric_native_sources = 1

        elif isinstance(data, str):
            self._infer_string(data)

        elif isinstance(data, list):
            self.kind = "list"
            self.value_structure = self._merge_values(data)
            self.empty_sources = int(len(data) == 0)
            self.container_sources = 1

        elif isinstance(data, dict):
            self.kind = "dict"
            self.key_structures = {key: Structure(value) for key, value in data.items()}
            self.empty_sources = int(len(data) == 0)
            self.container_sources = 1

    def _infer_string(self, value: str) -> None:
        """Classify a string as int-like, float-like, or genuinely textual."""
        try:
            int(value)
        except ValueError:
            try:
                float(value)
            except ValueError:
                self.kind = "str"
            else:
                self.kind = "float"
                self.numeric_string_sources = 1
        else:
            self.kind = "int"
            self.numeric_string_sources = 1

    @staticmethod
    def _merge_values(values) -> Structure:
        """Infer and merge every value in a list or dictionary."""
        merged = Structure.create_empty()

        for value in values:
            merged = merged.merge(Structure(value))

        return merged

    @classmethod
    def _create(
        cls,
        kind: Literal["empty", "none", "bool", "int", "float", "str", "list", "dict"],
        *,
        nullable: bool = False,
        value_structure: Structure | None = None,
        key_structures: dict[str, Structure] | None = None,
        optional_keys: set[str] | None = None,
        empty_sources: int = 0,
        container_sources: int = 0,
        numeric_string_sources: int = 0,
        numeric_native_sources: int = 0,
    ) -> Structure:
        """Create an already-inferred Structure without re-running inference."""
        result = object.__new__(cls)
        result.kind = kind
        result.nullable = nullable
        result.value_structure = (
            None if kind == "empty" else value_structure or cls._create("empty")
        )
        result.key_structures = key_structures or {}
        result.optional_keys = optional_keys or set()
        result.empty_sources = empty_sources
        result.container_sources = container_sources
        result.numeric_string_sources = numeric_string_sources
        result.numeric_native_sources = numeric_native_sources
        return result

    @classmethod
    def create_empty(cls) -> Structure:
        return Structure._create("empty")

    def merge(self, other: Structure) -> Structure:
        """Return the narrowest structure compatible with both structures.

        Raises TypeError when the structures are fundamentally incompatible,
        such as list vs dict or bool vs int.
        """
        assert other is not None
        # Internal empty structure contributes no type information.
        if self.kind == "empty":
            return other._copy()

        if other.kind == "empty":
            return self._copy()

        # An observed JSON null makes the other structure nullable.
        if other.kind == "none":
            if self.kind == "none":
                return self._create("none")

            result = self._copy()
            result.nullable = True
            return result

        if self.kind == "none":
            result = other._copy()
            result.nullable = True
            return result

        nullable = self.nullable or other.nullable

        # Numeric promotion: int + float -> float.
        if self.kind in self.NUMERIC_KINDS and other.kind in self.NUMERIC_KINDS:
            return self._create(
                "float" if "float" in (self.kind, other.kind) else "int",
                nullable=nullable,
                numeric_string_sources=self.numeric_string_sources + other.numeric_string_sources,
                numeric_native_sources=self.numeric_native_sources + other.numeric_native_sources,
            )

        # A genuine string forces numeric/string mixtures to become str.
        if (self.kind == "str" and other.kind in self.NUMERIC_KINDS) \
        or (other.kind == "str" and self.kind in self.NUMERIC_KINDS):
            return self._create("str", nullable=nullable)

        if self.kind != other.kind:
            raise TypeError(f"incompatible structures: {self.kind} and {other.kind}")

        # Merge the contents of homogeneous containers recursively.
        if self.kind == "list":
            return self._create(
                self.kind,
                nullable=nullable,
                empty_sources=self.empty_sources + other.empty_sources,
                container_sources=self.container_sources + other.container_sources,
                value_structure=self.value_structure.merge(other.value_structure),
            )

        if self.kind == "dict":
            key_structures: dict[str, Structure] = {}
            optional_keys = self.optional_keys | other.optional_keys
            for key in {**self.key_structures, **other.key_structures}:
                if key in self.key_structures and key in other.key_structures:
                    key_structures[key] = self.key_structures[key].merge(other.key_structures[key])
                else:
                    key_structures[key] = (
                        self.key_structures[key] if key in self.key_structures else other.key_structures[key]
                    )._copy()
                    optional_keys.add(key)

            return self._create(
                self.kind,
                nullable=nullable,
                empty_sources=self.empty_sources + other.empty_sources,
                container_sources=self.container_sources + other.container_sources,
                key_structures=key_structures,
                optional_keys=optional_keys,
            )

        # Same primitive type, e.g. bool + bool or str + str
        return self._create(self.kind, nullable=nullable)  # update nullable state only

    def _copy(self) -> Structure:
        return self._create(
            self.kind,
            nullable=self.nullable,
            value_structure=(
                self.value_structure._copy()
                if self.value_structure is not None
                else None
            ),
            key_structures={
                key: structure._copy() for key, structure in self.key_structures.items()
            },
            optional_keys=set(self.optional_keys),
            empty_sources=self.empty_sources,
            container_sources=self.container_sources,
            numeric_string_sources=self.numeric_string_sources,
            numeric_native_sources=self.numeric_native_sources,
        )

    @property
    def numeric_provenance(self) -> str | None:
        """Return how numeric values were represented in the source JSON.

        None   -> not a numeric structure
        "none" -> all were JSON numbers
        "x%"   -> x% were numeric-looking strings
        "all"  -> all were numeric-looking strings
        """
        if self.kind not in self.NUMERIC_KINDS:
            return None
        if self.numeric_string_sources == 0:
            return "none"
        if self.numeric_native_sources == 0:
            return f"all ({self.numeric_string_sources})"

        total = self.numeric_string_sources + self.numeric_native_sources
        percent = 100 * self.numeric_string_sources // total
        return f"{percent or '<1'}% ({self.numeric_string_sources}-of-{total})"

    @property
    def empty_sources_percentage(self) -> str | None:
        """Return the percentage of observed containers that were empty."""
        if self.kind not in self.CONTAINER_KINDS or self.empty_sources == 0:
            return None

        percent = 100 * self.empty_sources // self.container_sources
        return f"{percent or '<1'}%"

    def _metadata(self) -> list[str]:
        """Return the metadata entries describing this structure's own top level."""
        metadata: list[str] = []

        if self.numeric_provenance not in ("none", None):
            metadata.append(f"{self.numeric_provenance} as str")

        if self.kind == "list":
            if self.value_structure.kind != "empty" and self.empty_sources_percentage:
                metadata.append(f"{self.empty_sources_percentage} empty")
        elif self.kind == "dict":
            if self.key_structures and self.empty_sources_percentage:
                metadata.append(f"{self.empty_sources_percentage} empty")

        if self.nullable and self.kind != "none":
            metadata.append("could be null")

        return metadata

    def _repr_entries(self) -> list[str]:
        """Render the per-key entries of a dict: each key's structure and metadata."""
        entries: list[str] = []
        for key, structure in self.key_structures.items():
            rendered_key = key + ("__???" if key in self.optional_keys else "")
            entries.append(f"{json.dumps(rendered_key)}: {structure._repr_body()}")

            key_metadata = structure._metadata()
            if key_metadata:
                metadata_items = ", ".join(json.dumps(entry) for entry in key_metadata)
                entries.append(f"{json.dumps(key + '__metadata__')}: [{metadata_items}]")

        return entries

    def _repr_body(self) -> str:
        """Render this structure as valid JSON without its own top-level metadata."""
        if self.kind == "empty":
            return '"empty"'

        if self.kind == "none":
            return '"None"'

        if self.kind == "list":
            return f"[{repr(self.value_structure)}]"

        if self.kind == "dict":
            return "{" + ", ".join(self._repr_entries()) + "}"

        return json.dumps(self.kind)

    def __str__(self) -> str:
        if self.kind == "empty":
            return "empty"

        if self.kind == "none":
            return "None"

        if self.kind == "list":
            result = f"list[{self.value_structure}]"

            assert (self.value_structure.kind == "empty") == (self.empty_sources_percentage == "100%"), \
                f"Implementation has errors somewhere, {self.value_structure.kind=} but {self.empty_sources_percentage=}"
            if self.value_structure.kind != "empty" and self.empty_sources_percentage:
                result += f" ({self.empty_sources_percentage} empty)"

        elif self.kind == "dict":
            entries: list[str] = []
            for key, structure in self.key_structures.items():
                rendered_key = key + ("__???" if key in self.optional_keys else "")
                entries.append(f"{rendered_key}: {structure}")

            result = "dict[str, {" + ", ".join(entries) + "}]"
            if self.key_structures and self.empty_sources_percentage:
                result += f" ({self.empty_sources_percentage} empty)"

        else:
            result = self.kind

            if self.numeric_provenance not in ("none", None):
                result += f" ({self.numeric_provenance} as str)"

        if self.nullable and self.kind != "none":
            result += " | None"

        return result

    def __repr__(self) -> str:
        if self.kind == "empty":
            return '"empty"'

        if self.kind == "none":
            return '"None"'

        metadata = self._metadata()
        metadata_items = ", ".join(json.dumps(entry) for entry in metadata)

        if self.kind == "list":
            assert (self.value_structure.kind == "empty") == (self.empty_sources_percentage == "100%"), \
                f"Implementation has errors somewhere, {self.value_structure.kind=} but {self.empty_sources_percentage=}"

            result = f"[{repr(self.value_structure)}"
            if metadata:
                result += f', {{"__metadata__": [{metadata_items}]}}'
            return result + "]"

        if self.kind == "dict":
            entries = self._repr_entries()
            if metadata:
                entries.append(f'"__metadata__": [{metadata_items}]')
            return "{" + ", ".join(entries) + "}"

        result = json.dumps(self.kind)
        if metadata:
            result = json.dumps(f"{self.kind}, " + ", ".join(f"({entry})" for entry in metadata))
        return result

    def as_json(self) -> Any:
        """Return the Structure as a parsed JSON-compatible Python value."""
        return json.loads(repr(self))

    def as_json_str(self, indent: int = 4, **kwargs) -> str:
        """Return the Structure serialized as a JSON string."""
        return json.dumps(self.as_json(), indent=indent, **kwargs)


def analyze_jsonl_structure(file_path):
    try:
        data_structure = Structure.create_empty()

        first_structure = None
        first_structure_raw = None

        differing_structure = None
        differing_structure_raw = None
        diff_line_num = -1

        print(f"Analyzing structure of {file_path}...")
        idx = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            # tqdm will close nicely even if we break early
            for idx, line in enumerate(tqdm(f, desc="Processing"), start=1):
                line = line.strip()
                if not line:
                    continue
                
                line_data = json.loads(line)
                line_structure = Structure(line_data)

                try:
                    data_structure = data_structure.merge(line_structure)
                except TypeError:
                    differing_structure = line_structure
                    differing_structure_raw = line_data
                    diff_line_num = idx
                    break  # Stop processing immediately on the very first mismatch

                if first_structure is None:
                    first_structure = line_structure
                    first_structure_raw = line_data
                    
        total_lines_processed = idx
        # --- Output Results ---
        print("\n" + "="*50)
        print(" ANALYSIS RESULTS")
        print("="*50)
        
        if differing_structure is None:
            print("✅ Success: All lines in the JSONL file have a UNIFORM structure!")
            print(f"\nStructure (found in all {total_lines_processed} lines):")
            print(data_structure.as_json_str())
        else:
            print("❌ Alert: Differing structures detected!")
            print(f"The structural mismatch occurred at **Line {diff_line_num}**.")
            print("Scan halted immediately.\n")
            print("\n" + "-"*30)

            first_line_str = f"1{f' to {idx-1}' if idx-1 > 1 else ''}"
            diff_line_str = f"{diff_line_num}"
            spacing = max(len(first_line_str), len(diff_line_str))

            print(f"--- Structure of line {first_line_str:^{spacing}} ---")
            print(first_structure.as_json_str())
            print(f"--- Structure of line {diff_line_str:^{spacing}} ---")
            print(differing_structure.as_json_str())
            print("\n" + "-"*30)

            print(f"Example Data (Line {1:>{len(diff_line_str)}}):")
            print(json.dumps(first_structure_raw, indent=4, default=str), end="\n\n")
            print(f"Example Data (Line {diff_line_num}):")
            print(json.dumps(differing_structure_raw, indent=4, default=str))

    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except json.JSONDecodeError as e:
        print(f"\nError: Failed to parse JSON. {e}")


def main() -> None:
    sys.argv = _preprocess_underscore_args(sys.argv)

    parser = argparse.ArgumentParser(
        description="Show the data structure of a JSONL file and check that it is consistent throughout the file.",
    )
    parser.add_argument(
        "input",
        type=str,
        help="Path to the input JSONL file (single file, no glob).",
    )

    args = parser.parse_args()

    input_path = Path(args.input.strip().rstrip("/").strip()).resolve()
    if not input_path.is_file():
        parser.error(f"Input file does not exist: {input_path}")

    analyze_jsonl_structure(str(input_path))


if __name__ == "__main__":
    main()

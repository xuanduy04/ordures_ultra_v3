"""Read-only verification of the transfer ZIP and every manifest member."""

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    args = parser.parse_args()
    outer = hashlib.sha256()
    with args.zip.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            outer.update(block)
    if outer.hexdigest() != args.sha256:
        raise ValueError("Transfer ZIP SHA256 mismatch")
    with zipfile.ZipFile(args.zip) as archive:
        names = archive.namelist()
        assert len(names) == len(set(names)), "Duplicate ZIP entries"
        for name in names:
            path = PurePosixPath(name)
            assert not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name
        manifest = json.loads(archive.read("packet-manifest.json"))
        assert set(names) == set(manifest["files"]) | {"packet-manifest.json"}
        for name, expected in manifest["files"].items():
            data = archive.read(name)
            assert len(data) == expected["bytes"] and hashlib.sha256(data).hexdigest() == expected["sha256"], name
    print(json.dumps({"success": True, "source_commit": manifest["source_commit"],
                      "verified_files": len(manifest["files"]), "scope": "ZIP integrity only"}))


if __name__ == "__main__":
    main()

# utils_data

Standalone CLI helpers for preparing and manipulating JSON/JSONL datasets.
Each script is self-contained (helpers are duplicated, not imported) and only
depends on `tqdm`. Run directly, e.g. `python convert_to_general_qa.py ...`.


## Terminology:
- SFT data: SFT-training compatible data, rows containing non-empty `messages` and optionally `tools`.
- RL QA data: Question-Answer data, rows containing non-empty fields `question` and `ground_truth`.
- NeMo-Gym data: NeMo-Gym compatible data, rows may contain complex fields. Read the specific environment data example in (usually `3rdparty/Gym-workspace/Gym/resources_servers/`) for the specific field requirements.


## Common CLI for all scripts:

Scripts share a common CLI, with specific script types also containing their own unique extra requirements:
- `--output`:  path to the output file; refuses to overwrite unless confirmed. Optional (with defaults) or required depends on script category and the script itself.
- `--yes`, `-y`: preemptively always confirm overwrite of output file


## Converters:

With script name `convert_<optional input description>_to_<output description>.py`, converters reformat one type of data into another. 

They share the extra common CLI:
- `input` (positional): source `.jsonl`, mandates to be a single file
And may share extra common CLIs depending on the category.

### NeMo-Gym Converters

Scripts that convert data into a NeMo-Gym environment compatible data, they share extra common CLIs:
- `--output`: Default `<input stem>-<output environment name>.jsonl` (example `<input stem>-general_qa.jsonl`, `<input stem>-genrm_compare.jsonl`)
- `--prompt-field`: prompt as a plain string, a dict, or chat-template turns. Default `messages`
- `--tools-field`: optional tool definitions. Default `tools`
- `--no-reasoning`: use the `_reasoning_off` agent variant
- `--dataset` (required): stamped on every row; blank value omits the field
- `--skip-on-error`: log-and-skip bad rows instead of aborting (fatal setup errors still abort)

List of implemented NeMo-Gym converters:

1. `convert_to_general_qa.py`

Writes rows with `agent_ref`, `responses_create_params`, `question`,
`expected_answer`, `should_use_judge`, and optional `dataset`. Extra flags:

- `--answer-field`: Default `expected_answer`
- `--should-use-judge`
- `--instruction-prefix`: prepended to plain-string prompts only

2. `convert_to_genrm_compare.py`

Writes rows with `agent_ref` (type `responses_api_agents`) `responses_create_params` (`parallel_tool_calls: false`), and optional `dataset` and `principle`. The principle comes from exactly one of:

- `--principle-string TEXT`: a literal principle
- `--principle-file PATH`: a text file whose content is the principle
- `--principle-script PATH`: a `.py` file defining `get_principle(entry: dict) -> str`,
  called once per entry to produce that row's principle

### Other Converters

List of implemented other converters:

1. `convert_sft_data_to_rl_qa.py`:

All turns of `messages` except the last become `question`, the last turn's content becomes `ground_truth`. Default for `--output` is `<input stem>-rl_qa.jsonl`


## Dataset Functions

With script name `dataset_<function description>.py`, dataset functions perform set/ arithmetic operations on (usually multiple) dataset(s).

They share the extra common CLI:
- `--output`: Default is at `./output/<script name>.jsonl` as they may process multiple input files (e.g. default output for `dataset_set_minus.py` is `./output/dataset_set_minus.jsonl`)


List of implemented dataset functions:

1. `dataset_set_minus.py`:

Drops lines of datasets `A` that appear in datasets `B` or is repeated within `A`, compared as raw stripped JSON strings. Extra flags: `--a`/`--b` (both required, glob-supported).


## Sample Scripts

With script name `sample_<input file description>.py`, sample scripts sample from an input dataset and write them to output.

They share the common CLI:
- `input` (positional): source `.jsonl`, mandates to be a single file
- `--output` (required)
- `--size` (required): number of samples to extract. Scripts may optionally define sample size in a different manner. This description is for scripts that don't have special methods of doing so.
- `--seed`: random seed for sampling. Default `67`

List of implemented sample scripts:

1. `sample_jsonl_file.py`:

A general sampling script for all types of `.jsonl` data. Only samples from valid `json` lines in input file.

2. `sample_nemo_gym_datasets.py`:

Samples a NeMo-Gym compatible dataset by per-agent counts defined in `AGENT_COUNT` (users are expected to open the script and edit this dict to change targets).


## Other Scripts
Other scripts not matching the above categories go here.


## For Agents

### Script Layout

All scripts follow the same top-to-bottom layout; helpers are defined in the order they are called:

1. Imports: `from __future__ import annotations`, stdlib, then `tqdm.auto`.
2. Logging: `logging.basicConfig(...)` + module-level `logger`.
3. Constants (`UPPER_CASE`), e.g. `AGENT_NAME`, `BUF_SIZE`, `AGENT_COUNT`.
4. Private helpers (`_`-prefixed), defined in call order:
    - common helpers first (`_preprocess_underscore_args`, `_validate_output_path`)
    - script-specific helpers follow (`_convert_*`, `_resolve_input_files`, `_count_*`, `_*_pass`, ...);
5. Within `if __name__ == "__main__": main()`: parse args, validate paths, run the passes, log summary.

### Helper duplication

Scripts never import each other. When a new script needs an existing
helper, copy it from the current version in an existing script.
Per-file variation is acceptable.

### `--yes`/`-y` contract

`--yes` only bypasses the interactive overwrite prompt when the output file already exists. Setup errors stay fatal: input must exist, input and output files must be mutually exclusive, output must end in `.jsonl`.

### Adding a new script

1. Name it per the taxonomy (`convert_*`, `dataset_*`, `sample_*`, else Other).
2. Implement the common CLI + its category CLI; normalize `_`→`-` in args via `_preprocess_underscore_args`.
3. Copy helpers, never import; keep the layout above. Do not add top-level module docstring.
4. Add an entry to the matching list in this README.

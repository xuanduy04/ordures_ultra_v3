# utils_data

Standalone CLI helpers for preparing and manipulating JSON/JSONL datasets.
Each script is self-contained (helpers are duplicated, not imported) and only
depends on `tqdm`. Run directly, e.g. `python convert_to_general_qa.py ...`.


## Terminology:

### Data Types:
- SFT data: SFT-training compatible data, rows containing non-empty `messages` and optionally `tools`
- RL QA data: Question-Answer data, rows containing non-empty fields `question` and `ground_truth`
- NeMo-Gym data: NeMo-Gym compatible data, rows may contain complex fields. Read the specific environment data example in (usually `3rdparty/Gym-workspace/Gym/resources_servers/`) for the specific field requirements

### Filename Standard:
The filename stem for finalized files should follow `<content>-<metadata>-<data type>`, where:
- `content`: the data's semantic contents. The user is responsible for this field. Use underscores (`_`) instead of hyphens (`-`).
- `metadata`: compact metadata consisting of hyphen-separated fields in the following order (all numbers rounded to nearest thousands, values below `1k` represented as `1k`):
    1. `len<length>k` (optional): The intended maximum token length of the data type's Main field, expressed in thousands of tokens.
    Main fields of each data type:
    - SFT data: `messages`
    - RL QA data: `question`
    - NeMo-Gym data: `responses_create_params.input`
    (It is not the Agent's job to care about what the tokenizer is)
    2. `sam<samples>k`: Number of samples in the current dataset
- `data type`: the dataset type:
    - SFT data: `SFT`
    - RL QA data: `RL_QA`
    - NeMo-Gym data containing exactly 1 environment: that environment's name
    - NeMo-Gym data containing multiple environments: `nemo_gym`

Not all input files processed by scripts are expected to follow this standard. Users should run `rename_data_to_follow_standard.py` when a finalized file is ready to be renamed. Other scripts do not enforce or depend on this filename format, although their default output names help show the chronological data-processing pipeline that `rename_data_to_follow_standard.py` uses when renaming finalized files.

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

Implemented NeMo-Gym converters:

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

Implemented other converters:

1. `convert_sft_data_to_rl_qa.py`:

All turns of `messages` except the last become `question`, the last turn's content becomes `ground_truth`. Default for `--output` is `<input stem>-RL_QA.jsonl`


## Dataset Functions

Dataset functions perform operations on "dataset group(s)", each "dataset group" can contain 1 to more `jsonl` files. They have script name `get_<feature of output data>_data.py` and `dataset_<function description>.py` for functions using one and multiple dataset group, respectively.

They share the extra common CLI:
- `--output`: Default is at `./output/<script name>.jsonl` as they may process multiple input files (e.g. default output for `dataset_set_minus.py` is `./output/dataset_set_minus.jsonl`)


Implemented dataset functions:

1. `dataset_set_minus.py`:

Drops lines of datasets `A` that appear in datasets `B` or is repeated within `A`, compared as raw stripped JSON strings (output = `A` set minus `B`). Extra flags: `--a`/`--b` (both required, glob-supported).

2. `get_untrained_data.py`:

**UNFINISHED**, works inconsistently, not full-proof. Extract the untrained data from given dataset(s) `A` according to training logs `B`. Extra flags: `--a`/`--b` (both required, glob-supported, `b` must expand to folder(s) containing `training_data_step<step number>.jsonl` logs).


## Sample Scripts

With script name `sample_<input file description>.py`, sample scripts uniformly sample from an input dataset and write them to output.

They share the common CLI:
- `input` (positional): source `.jsonl`, mandates to be a single file
- `--output`: Default `<input stem>-sam<size>.jsonl` if the script implements a `--size` flag. Otherwise this field is required
- `--size` (required): number of samples to extract. Specific scripts that specify the sample size in a different manner will not have this flag.
- `--seed`: random seed for sampling. Default `67`

Implemented sample scripts:

1. `sample_jsonl_file.py`:

A general sampling script for all types of `.jsonl` data. Only samples from valid `json` lines in input file.

2. `sample_nemo_gym_datasets.py`:

Samples a NeMo-Gym compatible dataset by per-agent counts defined in `AGENT_COUNT` (users are expected to open the script file and edit this dict to change targets).


## Filename Standard Scripts

Scripts used to maintain filename standards.

Implemented scripts:

1. `rename_data_to_follow_standard.py`: 

Renames **inplace** the json file into one that follows the Filename Standard.

Reads the current filename right-to-left to determine its `len` metdata (if present) and `data type`; additionally counts lines to determine the `sam` metadata.

Valid data types are `SFT`, `RL_QA`, `nemo_gym`, or a NeMo-Gym environment name (discovered from `3rdparty/Gym-workspace/Gym/resources_servers/`). If the filename has no valid data type, the script errors out and asks the user to specify `--data-type`. Extra flags: `--data-type` (always replaces the data type parsed from the filename when given).

2. `filter_overlong_token.py`: 

Filters the input data's main field (according to `data_type`) to be less than `max-token` tokens according to the `tokenizer`. Extra flags: `--max-token`, `--tokenizer`, `--data-type`. Default for `--output` is `<input stem>-len<resolved max token in thousands>k.jsonl`

`--max-token` supports shorthand in thousands: if its value is not divisible by `1000`, it is interpreted as thousands of tokens (e.g. `6` becomes `6000` tokens), while values divisible by `1000` are used directly (e.g. `7000` stay `7000` tokens)

*Note*: Shuffles the dataset due to multiprocessing usage


## Other Scripts
Other scripts not matching the above categories go here.

Implemented scripts:

1. `show_jsonl_structure.py`:

Shows the input file's data structure, also checks if the structure is consistent throughout the file.

2. `statistic_of_train_data.py`:

Generates an ASCII-art table of the input training data. Positional `input` flag supports glob, `--output` has default `./statistic_of_train_data.md`


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


from __future__ import annotations

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


def main(default_config_path="examples/llm_pretrain/nanogpt_pretrain.yaml"):
    """Entry-point for launching NanoGPT-style pre-training.

    The script follows the same invocation pattern as *examples/llm_finetune/finetune.py*:

    ```bash
    torchrun --nproc-per-node <NGPU> examples/llm_pretrain/pretrain.py \
        --config examples/llm/nanogpt_pretrain.yaml
    ```
    """
    cfg = parse_args_and_load_config(default_config_path)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()

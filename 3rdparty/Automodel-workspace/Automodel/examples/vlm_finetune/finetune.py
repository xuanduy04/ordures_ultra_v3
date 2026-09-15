

from __future__ import annotations

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.vlm.finetune import FinetuneRecipeForVLM


def main(config="examples/vlm_finetune/gemma3/gemma3_vl_4b_cord_v2.yaml"):
    """Main entry point for the fine-tuning recipe.

    Loads the configuration, sets up the recipe, and initiates the training loop.
    """
    cfg = parse_args_and_load_config(config)
    recipe = FinetuneRecipeForVLM(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()

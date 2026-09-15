

from __future__ import annotations

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.train_ft import TrainFinetuneRecipeForNextTokenPrediction


def main(default_config_path="examples/llm_finetune/llama3_2/llama3_2_1b_hellaswag.yaml"):
    """Main entry point for the fine-tuning recipe.

    Loads the configuration, sets up the recipe, and initiates the training loop.
    """
    cfg = parse_args_and_load_config(default_config_path)
    recipe = TrainFinetuneRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()



from __future__ import annotations

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.biencoder import TrainBiencoderRecipe


def main(default_config_path="examples/biencoder/llama3_2_1b_biencoder.yaml"):
    """Main entry point for the biencoder fine-tuning recipe.

    Loads the configuration, sets up the recipe, and initiates the training loop.

    Args:
        default_config_path: Path to the default configuration file
    """
    cfg = parse_args_and_load_config(default_config_path)
    recipe = TrainBiencoderRecipe(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":
    main()



"""Example launcher for knowledge distillation fine-tuning.

Usage (single GPU):
    python examples/llm/knowledge_distillation.py -c examples/llm/llama_3_2_1b_kd.yaml

When run without ``-c`` it defaults to the YAML above.
"""

from __future__ import annotations

from nemo_automodel.components.config._arg_parser import parse_args_and_load_config
from nemo_automodel.recipes.llm.kd import (
    KnowledgeDistillationRecipeForNextTokenPrediction,
)


def main(default_config_path="examples/llm_kd/llama3_2/llama3_2_1b_kd.yaml") -> None:
    """Entry-point mirroring ``examples/llm/finetune.py`` but for KD."""
    cfg = parse_args_and_load_config(default_config_path)
    recipe = KnowledgeDistillationRecipeForNextTokenPrediction(cfg)
    recipe.setup()
    recipe.run_train_validation_loop()


if __name__ == "__main__":  # pragma: no cover
    main()

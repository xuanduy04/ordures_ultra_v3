

import torch.nn as nn


def is_tied_word_embeddings(model: nn.Module) -> bool:
    """
    Check if the model's word embeddings are tied.

    Args:
        model (nn.Module): The model to check.

    Returns:
        bool: True if the model's word embeddings are tied, False otherwise.
    """
    non_tied_lm_head_models = {
        "Qwen3OmniMoeThinkerForConditionalGeneration",  # complicated config structure
    }
    model_class_name = type(model).__name__
    for m in non_tied_lm_head_models:
        if m in model_class_name:
            return False
    config = getattr(model, "config", None)
    text_config = getattr(config, "get_text_config", lambda: None)()
    return bool(getattr(text_config, "tie_word_embeddings", getattr(config, "tie_word_embeddings", False)))

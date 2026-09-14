

from types import SimpleNamespace

import torch.nn as nn

import nemo_automodel.components.checkpoint.utils as checkpoint_utils


def test_is_tied_word_embeddings_prefers_text_config_value():
    class DummyTextConfig:
        def __init__(self, tied: bool) -> None:
            self.tie_word_embeddings = tied

    class DummyConfig:
        def __init__(self) -> None:
            self.tie_word_embeddings = True
            self._text = DummyTextConfig(False)

        def get_text_config(self):
            return self._text

    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = DummyConfig()

    model = DummyModel()
    assert checkpoint_utils.is_tied_word_embeddings(model) is False


def test_is_tied_word_embeddings_falls_back_to_top_level_when_no_text_config():
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=True)

    model = DummyModel()
    assert checkpoint_utils.is_tied_word_embeddings(model) is True


def test_is_tied_word_embeddings_handles_missing_config():
    class DummyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()

    model = DummyModel()
    assert checkpoint_utils.is_tied_word_embeddings(model) is False


def test_is_tied_word_embeddings_respects_exclusion_list():
    class Qwen3OmniMoeThinkerForConditionalGeneration(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(tie_word_embeddings=True)

    model = Qwen3OmniMoeThinkerForConditionalGeneration()
    assert checkpoint_utils.is_tied_word_embeddings(model) is False

"""Forward explicit tokenizer compatibility arguments without changing assets."""

from unittest.mock import patch

import pytest

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.algorithms.xtoken_off_policy_distillation import MasterConfig
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from omegaconf import OmegaConf


class Tokenizer:
    pad_token = None
    eos_token = "<eos>"
    chat_template = "template"

    def apply_chat_template(self, messages, **kwargs):
        return kwargs


def test_explicit_gemma_metadata_override_and_chat_kwargs():
    kwargs = {"extra_special_tokens": {"eot_token": "<turn|>"},
              "additional_special_tokens": ["<|video|>"], "local_files_only": True}
    config = {"name": "/staged/gemma", "from_pretrained_kwargs": kwargs,
              "chat_template_kwargs": {"enable_thinking": False}}
    tokenizer = Tokenizer()
    with patch("nemo_rl.algorithms.utils.maybe_patch_fastokens"), patch(
        "nemo_rl.algorithms.utils.AutoTokenizer.from_pretrained", return_value=tokenizer
    ) as loader:
        result = get_tokenizer(config)
    loader.assert_called_once_with("/staged/gemma", trust_remote_code=True, **kwargs)
    assert result.pad_token == "<eos>"
    assert result.apply_chat_template([])["enable_thinking"] is False
    assert config["from_pretrained_kwargs"] == kwargs


@pytest.mark.parametrize("mode", ["off", "on"])
def test_qwen3_gemma4_recipe(mode):
    register_omegaconf_resolvers()
    config = MasterConfig(**OmegaConf.to_container(load_config(
        f"examples/configs/xtoken_qwen3_gemma4_{mode}_policy_distillation.yaml"), resolve=True))
    assert config.cluster["gpus_per_node"] == 4
    assert config.policy["model_name"] == "/models/Qwen3-4B"
    assert config.teachers[0].remote["model_vocab_size"] == 262144
    assert config.teachers[0].remote["top_k"] == 20
    metadata = config.teachers[0].policy_config()["tokenizer"]["from_pretrained_kwargs"]
    assert isinstance(metadata["extra_special_tokens"], dict)
    assert metadata["extra_special_tokens"]["escape_token"] == '<|"|>'

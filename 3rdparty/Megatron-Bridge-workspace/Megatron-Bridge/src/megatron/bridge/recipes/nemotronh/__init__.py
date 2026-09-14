

# Nemotron Nano v2 models
# Nemotron 3 Nano models
from megatron.bridge.recipes.nemotronh.nemotron_3_nano import (
    nemotron_3_nano_finetune_config,
    nemotron_3_nano_pretrain_config,
)
from megatron.bridge.recipes.nemotronh.nemotron_nano_v2 import (
    nemotron_nano_9b_v2_finetune_config,
    nemotron_nano_9b_v2_pretrain_config,
    nemotron_nano_12b_v2_finetune_config,
    nemotron_nano_12b_v2_pretrain_config,
)

# NemotronH models
from megatron.bridge.recipes.nemotronh.nemotronh import (
    nemotronh_4b_finetune_config,
    nemotronh_4b_pretrain_config,
    nemotronh_8b_finetune_config,
    nemotronh_8b_pretrain_config,
    nemotronh_47b_finetune_config,
    nemotronh_47b_pretrain_config,
    nemotronh_56b_finetune_config,
    nemotronh_56b_pretrain_config,
)


__all__ = [
    # NemotronH models
    "nemotronh_4b_pretrain_config",
    "nemotronh_8b_pretrain_config",
    "nemotronh_47b_pretrain_config",
    "nemotronh_56b_pretrain_config",
    "nemotronh_4b_finetune_config",
    "nemotronh_8b_finetune_config",
    "nemotronh_47b_finetune_config",
    "nemotronh_56b_finetune_config",
    # Nemotron Nano v2 models
    "nemotron_nano_9b_v2_pretrain_config",
    "nemotron_nano_12b_v2_pretrain_config",
    "nemotron_nano_9b_v2_finetune_config",
    "nemotron_nano_12b_v2_finetune_config",
    # Nemotron 3 Nano models
    "nemotron_3_nano_pretrain_config",
    "nemotron_3_nano_finetune_config",
]

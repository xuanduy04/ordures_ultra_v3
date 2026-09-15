

"""State dict adapter for Qwen2 model with combined projections.

Uses the generic CombinedProjectionStateDictAdapter from common/.
"""

from transformers import Qwen2Config

from nemo_automodel.components.models.common.combined_projection.state_dict_adapter import (
    CombinedProjectionStateDictAdapter,
)


class Qwen2StateDictAdapter(CombinedProjectionStateDictAdapter):
    """State dict adapter for Qwen2 models.

    Inherits from the generic CombinedProjectionStateDictAdapter,
    providing a clean interface specific to Qwen2.

    Example:
        from transformers import Qwen2Config

        config = Qwen2Config.from_pretrained("Qwen/Qwen2.5-7B")
        adapter = Qwen2StateDictAdapter(config)

        # Convert HF checkpoint to custom format
        custom_state_dict = adapter.from_hf(hf_state_dict)

        # Convert custom checkpoint back to HF format
        hf_state_dict = adapter.to_hf(custom_state_dict)
    """

    def __init__(self, config: Qwen2Config):
        """Initialize adapter with Qwen2 config."""
        super().__init__(config)

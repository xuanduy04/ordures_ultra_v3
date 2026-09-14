

import logging


logger = logging.getLogger(__name__)


SAFE_REPOS: list[str] = [
    "deepseek-ai",
    "gpt2",
    "google",
    "llava-hf",
    "meta-llama",
    "mistralai",
    "moonshotai",
    "nvidia",
    "openai",
    "Qwen",
]


def is_safe_repo(hf_path: str, trust_remote_code: bool | None) -> bool:
    """
    Decide whether remote code execution should be enabled for a Hugging Face
    model or dataset repository.

    This function follows three rules:
        1. If `trust_remote_code` is explicitly provided (True/False), its value
            takes precedence.
        2. If `trust_remote_code` is None, the function checks whether the repo
            belongs to a predefined list of trusted repositories (`SAFE_REPOS`).
        3. Otherwise, remote code execution is disabled.

    Args:
        hf_path (str):
            The Hugging Face repository identifier (e.g., "org/model_name").
        trust_remote_code (bool | None):
            If True, always allow remote code execution.
            If False, always disable it.
            If None, fall back to internal safety rules and trusted repo list.

    Returns:
        bool: Whether remote code execution should be enabled.
    """
    if trust_remote_code is not None:
        if trust_remote_code is False:
            logger.warning(
                "`trust_remote_code=False`. Remote code may not be executed. "
                "Set `trust_remote_code=True` only if you fully trust the Hugging Face repository."
            )
        return trust_remote_code

    hf_repo = hf_path.split("/")[0]
    if hf_repo in SAFE_REPOS:
        return True

    return False

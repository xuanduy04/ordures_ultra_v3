


MAJOR = 0
MINOR = 5
PATCH = 0
PRE_RELEASE = "rc0"

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = ".".join(map(str, VERSION[:3]))
__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = "nemo_rl"
__contact_names__ = "NVIDIA"
__contact_emails__ = "nemo-tookit@nvidia.com"
__homepage__ = "https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/"
__repository_url__ = "https://github.com/NVIDIA-NeMo/RL"
__download_url__ = "https://github.com/NVIDIA-NeMo/RL/releases"
__description__ = "NeMo-RL - a toolkit for model alignment"
__license__ = "Apache2"
__keywords__ = "deep learning, machine learning, gpu, NLP, NeMo, nvidia, pytorch, torch, language, reinforcement learning, RLHF, preference modeling, SteerLM, DPO"

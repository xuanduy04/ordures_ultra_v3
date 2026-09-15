


MAJOR = 0
MINOR = 2
PATCH = 0
PRE_RELEASE = ""

# Use the following formatting: (major, minor, patch, pre-release)
VERSION = (MAJOR, MINOR, PATCH, PRE_RELEASE)

__shortversion__ = ".".join(map(str, VERSION[:3]))
__version__ = ".".join(map(str, VERSION[:3])) + "".join(VERSION[3:])

__package_name__ = "nemo_automodel"
__contact_names__ = "NVIDIA"
__contact_emails__ = "nemo-toolkit@nvidia.com"
__homepage__ = "https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/stable/"
__repository_url__ = "https://github.com/NVIDIA-NeMo/Automodel"
__download_url__ = "https://github.com/NVIDIA-NeMo/Automodel/releases"
__description__ = "NeMo Automodel - Delivers zero-day integration with Hugging Face models, \
                   automating fine-tuning and pretraining with built-in parallelism, \
                   custom-kernels and optimized recipes"
__license__ = "Apache2"
__keywords__ = "deep learning, machine learning, gpu, NLP, NeMo, nvidia, pytorch, torch"

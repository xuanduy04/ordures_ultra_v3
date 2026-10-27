"""Adapt legacy NeMo worker setup to both installed Automodel FSDP2 APIs."""

import inspect
from copy import deepcopy


def validate_fsdp_mode(config):
    """HF sharding is opt-in; never silently replace the legacy model backend."""
    if "fsdp_mode" not in config:
        return
    mode = config["fsdp_mode"]
    if mode not in ("sharded", "hf_sharded"):
        raise ValueError("Use explicit fsdp_mode: sharded or hf_sharded")
    if mode == "sharded":
        return
    kwargs = config["automodel_kwargs"]
    if kwargs["force_hf"] is not True or kwargs["use_liger_kernel"] is not False:
        raise ValueError("hf_sharded requires explicit force_hf=true and use_liger_kernel=false")


def use_hf_meta_init(model_config, registry, force_hf=False):
    """HF 5 from_pretrained cannot run inside Accelerate's empty-weight patch."""
    import transformers

    return int(transformers.__version__.split(".")[0]) >= 5 and (force_hf or model_config.architectures[0] not in registry.model_arch_name_to_cls)


def hf_model_from_config(model_class, model_config, automodel_kwargs, attn_impl):
    """Build parameter-only meta weights; the caller ties weights after the context."""
    unsupported = set(automodel_kwargs) - {"use_liger_kernel", "force_hf"}
    if unsupported or automodel_kwargs.get("use_liger_kernel"):
        raise NotImplementedError("HF meta initialization cannot honor these Automodel options: " + str(sorted(automodel_kwargs)))
    import torch

    factories = [base for base in model_class.__mro__
                 if base.__module__.startswith("transformers.") and base.__name__.startswith("AutoModel")]
    if not factories:
        raise TypeError("No Hugging Face auto-model factory found for meta initialization")
    construction_config = deepcopy(model_config)
    tied = construction_config.tie_word_embeddings
    construction_config.tie_word_embeddings = False
    model = factories[0].from_config(construction_config, torch_dtype=torch.float32,
                                     attn_implementation=attn_impl, trust_remote_code=True)
    model.config.tie_word_embeddings = tied
    return model


def tensor_to_hf(adapter, fqn, tensor, *, quantization):
    """Convert refit weights through the installed adapter's public API."""
    if hasattr(adapter, "convert_single_tensor_to_hf"):
        return adapter.convert_single_tensor_to_hf(
            fqn, tensor, exclude_key_regex=r".*_extra_state.*", quantization=quantization,
        )
    if quantization:
        raise NotImplementedError("Installed adapter cannot honor quantized single-tensor refit")
    return list(adapter.to_hf({fqn: tensor}, exclude_key_regex=r".*_extra_state.*").items())


def load_base_checkpoint(checkpointer, model, **kwargs):
    """Handle the removed PEFT initialization argument without changing LoRA."""
    if "peft_init_method" not in inspect.signature(checkpointer.load_base_model).parameters:
        method = kwargs.pop("peft_init_method")
        if method is not None:
            raise NotImplementedError("Installed Automodel cannot honor peft_init_method")
    return checkpointer.load_base_model(model, **kwargs)


def create_fsdp2_manager(manager_class, **kwargs):
    """Preserve the legacy constructor or build explicit config and meshes."""
    parameters = inspect.signature(manager_class).parameters
    if "dp_size" in parameters:
        return manager_class(**kwargs)
    if not {"config", "device_mesh"}.issubset(parameters):
        raise TypeError("Unsupported installed FSDP2Manager API: " + str(inspect.signature(manager_class)))
    from nemo_automodel.components.distributed.config import FSDP2Config
    from nemo_automodel.components.distributed.mesh_utils import create_device_mesh

    if kwargs["use_hf_tp_plan"]:
        raise NotImplementedError("Explicit HF TP plans require a version-specific tp_plan; automatic plans remain supported")
    config = FSDP2Config(
        sequence_parallel=kwargs["sequence_parallel"], tp_plan=kwargs["custom_tp_plan"],
        mp_policy=kwargs["mp_policy"], offload_policy=kwargs["offload_policy"],
        activation_checkpointing=kwargs["activation_checkpointing"],
        defer_fsdp_grad_sync=kwargs["defer_fsdp_grad_sync"], backend=kwargs["backend"],
    )
    mesh, moe_mesh = create_device_mesh(
        config, dp_size=kwargs["dp_size"], dp_replicate_size=kwargs["dp_replicate_size"],
        tp_size=kwargs["tp_size"], pp_size=kwargs["pp_size"], cp_size=kwargs["cp_size"],
        ep_size=kwargs["ep_size"], world_size=kwargs["world_size"],
    )
    manager = manager_class(config=config, device_mesh=mesh, moe_mesh=moe_mesh)
    for axis in ("dp", "tp", "cp", "pp", "dp_replicate", "dp_shard"):
        setattr(manager, axis + "_size", mesh[axis].size())
        setattr(manager, axis + "_mesh", mesh[axis])
    manager.ep_size = kwargs["ep_size"]
    return manager

"""Adapt legacy NeMo worker setup to both installed Automodel FSDP2 APIs."""

import inspect


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

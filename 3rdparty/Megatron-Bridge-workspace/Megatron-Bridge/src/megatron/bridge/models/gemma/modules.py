

import torch


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(
        base_cls_name, (mixin, base_cls), {}
    )  # mixin needs to go first for our forward() logic to work


class EmbeddingScalingMixin(torch.nn.Module):
    """
    A mixin class for scaling embeddings in Megatron GPT.
    The scaling is applied only if the configuration (accessible via `self.config`)
    includes `apply_embedding_scaling` set to True.
    """

    def forward(self, **kwargs):
        """
        Forward pass that scales the output embeddings from the `forward` method of
        the superclass by the square root of the hidden size specified in the configuration.
        """
        embeddings = super().forward(**kwargs)
        return embeddings * torch.tensor(self.config.hidden_size**0.5, dtype=embeddings.dtype)

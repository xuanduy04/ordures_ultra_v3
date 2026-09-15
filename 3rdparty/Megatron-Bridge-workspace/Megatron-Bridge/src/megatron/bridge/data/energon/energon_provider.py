

from dataclasses import dataclass
from typing import Any, Optional

from torch import int_repr

from megatron.bridge.data.energon.base_energon_datamodule import EnergonMultiModalDataModule
from megatron.bridge.data.utils import DatasetBuildContext, DatasetProvider


@dataclass(kw_only=True)
class EnergonProvider(DatasetProvider):
    """Energon Provider."""

    path: str
    image_processor: Optional[Any] = None
    seq_length: int
    micro_batch_size: int
    global_batch_size: int
    num_workers: int_repr
    dataloader_type: str = "external"
    task_encoder: Optional[Any] = None

    def build_datasets(self, context: DatasetBuildContext):
        dataset = EnergonMultiModalDataModule(
            path=self.path,
            tokenizer=context.tokenizer if context.tokenizer is not None else self.tokenizer,
            image_processor=self.image_processor,
            seq_length=self.seq_length,
            task_encoder=self.task_encoder,
            micro_batch_size=self.micro_batch_size,
            global_batch_size=self.global_batch_size,
            num_workers=self.num_workers,
        )
        return (
            iter(dataset.train_dataloader()),
            iter(dataset.val_dataloader()),
            iter(dataset.val_dataloader()),
        )

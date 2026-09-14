
from typing import Any, Optional

import torch

from nemo_rl.algorithms.interfaces import LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class SimpleLoss:
    loss_type = LossType.SEQUENCE_LEVEL

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor | None,
        global_valid_toks: torch.Tensor | None,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # Just return mean of logprobs as the loss for testing
        loss = next_token_logits.mean()
        metrics = {
            "loss": loss.item(),
            "test_metric": loss.item() * 0.5,
            "num_valid_samples": 1,
        }
        return loss, metrics


# Create a simple masked NLL loss function
class SimpleNLLLoss:
    loss_type = LossType.SEQUENCE_LEVEL

    def __call__(
        self,
        next_token_logits: torch.Tensor,
        data: BatchedDataDict,
        global_valid_seqs: torch.Tensor | None,
        global_valid_toks: torch.Tensor | None,
        vocab_parallel_rank: Optional[int] = None,
        vocab_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
        context_parallel_group: Optional[torch.distributed.ProcessGroup] = None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        # logits shape: [batch_size, seq_len, vocab_size]
        # Get the next token logits for each position
        next_tokens = data["input_ids"][:, 1:].cuda()  # Skip first token
        next_token_logprobs = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
        logprobs = next_token_logprobs[:, :-1]  # Remove last position's logits

        # Gather the logprobs for the actual next tokens
        token_logprobs = logprobs.gather(
            dim=-1, index=next_tokens.unsqueeze(-1)
        ).squeeze(-1)

        # Only compute loss on generated tokens (not input tokens)
        # by applying the token_loss_mask (shifted by 1 since we're predicting next tokens)
        token_loss_mask = data["token_loss_mask"][:, 1:].cuda()
        loss = -torch.sum(token_logprobs * token_loss_mask)

        return loss, {
            "loss": loss.item(),
            "num_valid_samples": 1,
        }

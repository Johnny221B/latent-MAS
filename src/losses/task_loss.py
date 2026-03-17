"""
TaskLoss: standard masked cross-entropy loss.

Labels tensor convention (constructed in data preprocessing):
  - Supervised positions: actual token IDs
  - Non-supervised positions (prompt, role, padding): -100

This is identical to HuggingFace's standard LM training convention.
No alignment logic needed here — all alignment happens in data prep.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskLoss(nn.Module):

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """Compute masked CE loss.

        Args:
            logits: [B, seq_len, V] from terminal agent
            labels: [B, seq_len] with -100 at non-supervised positions

        Returns:
            scalar loss
        """
        # Standard causal LM shift: logits[t] predicts labels[t+1]
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        return F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
        )
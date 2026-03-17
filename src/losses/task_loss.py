"""
TaskLoss: unified loss interface supporting multiple task types.

Two modes:
  1. CE mode (for open-ended generation tasks):
     Standard cross-entropy on answer tokens. Used when ground truth
     is a text sequence (e.g., explanation, code).

  2. Reward mode (for tasks with definitive answers):
     Binary reward (correct/wrong) with REINFORCE-style gradient.
     Used when ground truth is a single value (e.g., number, letter).
     Since REINFORCE has high variance, we use a simpler approach:
     compute CE loss on the correct answer token only.

Task type is specified per-call, so the same module works for all datasets.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TaskLoss(nn.Module):
    """Unified task loss supporting CE and reward-based modes."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.LongTensor,
        mode: str = "ce",
    ) -> torch.Tensor:
        """Compute task loss.

        Args:
            logits: [B, seq_len, V] from terminal agent
            labels: [B, label_len] ground truth tokens
            mode: "ce" for cross-entropy, "reward" for outcome-based

        Returns:
            scalar loss tensor
        """
        if mode == "ce":
            return self._ce_loss(logits, labels)
        elif mode == "reward":
            return self._reward_loss(logits, labels)
        else:
            raise ValueError(f"Unknown loss mode: {mode}. Use 'ce' or 'reward'.")

    def _ce_loss(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """Cross-entropy loss, right-aligned with answer tokens.

        For open-ended tasks where the answer is a text sequence.
        Only computes loss on answer token positions; other positions
        are masked with ignore_index.

        logits: [B, seq_len, V]
        labels: [B, label_len]
        """
        B, seq_len, V = logits.shape
        label_len = labels.shape[1]

        # Right-align labels with logits, fill rest with ignore_index
        aligned_labels = torch.full(
            (B, seq_len), self.ignore_index,
            dtype=labels.dtype, device=labels.device,
        )
        if label_len <= seq_len:
            aligned_labels[:, -label_len:] = labels
        else:
            aligned_labels[:, :] = labels[:, -seq_len:]

        # Standard causal LM shift
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = aligned_labels[:, 1:].contiguous()

        return self.ce_loss(
            shift_logits.view(-1, V),
            shift_labels.view(-1),
        )

    def _reward_loss(self, logits: torch.Tensor, labels: torch.LongTensor) -> torch.Tensor:
        """Reward-based loss for tasks with definitive short answers.

        For math/multiple-choice tasks where the answer is a single token
        or very short sequence. We compute CE loss only on the answer tokens,
        which acts like a targeted reward signal: the model is rewarded for
        assigning high probability to the correct answer.

        This is more stable than REINFORCE while achieving the same goal:
        push the model to predict the correct answer token.

        logits: [B, seq_len, V]
        labels: [B, label_len] (typically 1-3 tokens, e.g., "14" or "A")
        """
        B, seq_len, V = logits.shape
        label_len = labels.shape[1]

        # Take only the last `label_len` positions from logits
        # These are the positions that should predict the answer
        if label_len >= seq_len:
            answer_logits = logits
            answer_labels = labels[:, -seq_len:]
        else:
            answer_logits = logits[:, -(label_len + 1):-1, :]  # shifted: predict next token
            answer_labels = labels

        # Flatten and compute CE
        return F.cross_entropy(
            answer_logits.reshape(-1, V),
            answer_labels.reshape(-1),
        )
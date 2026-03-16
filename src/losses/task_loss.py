# src/losses/task_loss.py
"""
TaskLoss: computes the task-specific loss from the final agent's output.

Default: cross-entropy loss between the final agent's logits and the target tokens.

The key design decision here is WHERE to compute the loss:
  - The final agent produces hidden states over [prefix; role_prompt; task_input]
  - We need to get logits from these hidden states
  - Loss is computed only on the answer portion of the sequence

Extensibility:
  - For classification tasks (ARC, MedQA): override to extract answer token logits
  - For code generation (MBPP+, HumanEval+): override with execution-based reward
  - For RL-based training: replace with reward signal
"""

import torch
import torch.nn as nn


class TaskLoss(nn.Module):
    """Cross-entropy loss on the final agent's output."""

    def __init__(self, ignore_index: int = -100):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.ignore_index = ignore_index

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.LongTensor,
        prefix_len: int = 0,
    ) -> torch.Tensor:
        """Compute CE loss on the generated portion.

        Args:
            logits: [batch_size, seq_len, vocab_size]
                Full logits from the final agent's forward pass.
            labels: [batch_size, label_len]
                Ground truth token IDs for the answer.
            prefix_len: number of prefix tokens to skip in the logits.
                The loss is only computed on positions that correspond to answer tokens.

        Returns:
            loss: scalar tensor
        """
        # Standard causal LM loss: predict next token
        # Shift logits and labels for next-token prediction
        # logits[:, t, :] predicts token at position t+1
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        # Flatten for CE loss
        loss = self.ce_loss(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )
        return loss


class AnswerExtractionLoss(TaskLoss):
    """Loss computed only on the answer portion of the output.

    For tasks like GSM8K where the answer is a specific number,
    this masks out all non-answer positions.
    """

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.LongTensor,
        answer_start_positions: torch.LongTensor | None = None,
        prefix_len: int = 0,
    ) -> torch.Tensor:
        """Compute CE loss only on answer token positions.

        Args:
            logits: [batch_size, seq_len, vocab_size]
            labels: [batch_size, seq_len] with ignore_index at non-answer positions
            answer_start_positions: [batch_size] start index of answer in each sequence
            prefix_len: prefix tokens to account for in position mapping

        Returns:
            loss: scalar tensor
        """
        # Labels should already be padded with ignore_index at non-answer positions
        # Just call parent's forward
        return super().forward(logits, labels, prefix_len)

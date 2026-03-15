"""
LatentMessage: structured container for latent communication between agents.

Currently simple — just wraps a tensor with metadata.
Future use: track sender/receiver roles, compression ratio, etc.
"""

from dataclasses import dataclass
import torch


@dataclass
class LatentMessage:
    """A latent message from one agent to another."""
    sender_id: int
    receiver_id: int
    prefix: torch.Tensor  # [batch_size, Lp, hidden_dim]
    weight: float = 1.0   # edge weight A[sender, receiver]

    @property
    def shape(self):
        return self.prefix.shape

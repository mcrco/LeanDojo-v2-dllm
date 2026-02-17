"""Loss helpers for diffusion denoising training."""

from __future__ import annotations

import torch
import torch.nn.functional as F


def masked_denoising_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over masked positions (labels == -100 are ignored)."""
    if logits.ndim != 3:
        raise ValueError("logits must have shape (batch, seq_len, vocab_size)")
    if labels.ndim != 2:
        raise ValueError("labels must have shape (batch, seq_len)")

    vocab_size = logits.size(-1)
    return F.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )

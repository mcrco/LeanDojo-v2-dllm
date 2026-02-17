"""Diffusion training utilities and objectives for non-retrieval proving."""

from .config import DiffusionTrainingConfig
from .corruption import corrupt_tactic_script, mask_target_tokens
from .formatting import (
    format_infill_prompt,
    format_next_tactic_prompt,
    normalize_tactic_target,
)
from .objectives import masked_denoising_loss
from .whole_proof_data import build_whole_proof_examples, reconstruct_proof_from_tactics

__all__ = [
    "DiffusionTrainingConfig",
    "corrupt_tactic_script",
    "mask_target_tokens",
    "format_next_tactic_prompt",
    "format_infill_prompt",
    "normalize_tactic_target",
    "masked_denoising_loss",
    "build_whole_proof_examples",
    "reconstruct_proof_from_tactics",
]

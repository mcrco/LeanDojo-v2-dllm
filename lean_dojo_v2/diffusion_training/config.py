"""Configuration objects for diffusion training."""

from dataclasses import dataclass
from typing import Literal


ObjectiveType = Literal["next_tactic", "infill", "whole_proof"]


@dataclass
class DiffusionTrainingConfig:
    """Core training configuration for discrete denoising objectives."""

    model_name: str
    output_dir: str = "outputs-diffusion"
    objective: ObjectiveType = "next_tactic"
    epochs_per_repo: int = 1
    batch_size: int = 1
    gradient_accumulation_steps: int = 1
    lr: float = 2e-5
    max_length: int = 768
    mask_prob: float = 0.3
    seed: int = 42
    num_holes: int = 1
    max_hole_len: int = 3
    hole_token_template: str = "<HOLE_{i}>"

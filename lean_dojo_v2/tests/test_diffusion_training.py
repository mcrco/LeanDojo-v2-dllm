"""Unit tests for diffusion training corruption/objectives helpers."""

import random

import torch

from lean_dojo_v2.diffusion_training.corruption import (
    corrupt_tactic_script,
    mask_target_tokens,
)
from lean_dojo_v2.diffusion_training.objectives import masked_denoising_loss
from lean_dojo_v2.prompting import postprocess_tactic_candidates


def test_deterministic_span_corruption():
    tactics = ["intro h", "cases h", "exact h_left", "exact h_right"]
    rng = random.Random(7)
    corrupted_1, targets_1 = corrupt_tactic_script(
        tactics=tactics,
        num_holes=1,
        max_hole_len=2,
        rng=rng,
    )

    rng = random.Random(7)
    corrupted_2, targets_2 = corrupt_tactic_script(
        tactics=tactics,
        num_holes=1,
        max_hole_len=2,
        rng=rng,
    )

    assert corrupted_1 == corrupted_2
    assert targets_1 == targets_2


def test_postprocess_single_line_constraint():
    raw = ["exact h\nextra", "sorry", "simp<;>done", "admit", ""]
    assert postprocess_tactic_candidates(raw) == ["exact h", "simp"]


def test_corrupt_then_teacher_forced_reconstruction_loss_near_zero():
    input_ids = torch.tensor([[10, 20, 30, 40, 50]], dtype=torch.long)
    target_mask = torch.tensor([[False, False, True, True, True]])
    corrupted, labels = mask_target_tokens(
        input_ids=input_ids,
        target_mask=target_mask,
        mask_token_id=99,
        mask_prob=1.0,
        generator=torch.Generator().manual_seed(0),
    )

    assert corrupted.tolist() == [[10, 20, 99, 99, 99]]
    vocab_size = 120
    logits = torch.full((1, 5, vocab_size), fill_value=-30.0)
    for i in range(5):
        if labels[0, i] != -100:
            logits[0, i, labels[0, i].item()] = 30.0

    loss = masked_denoising_loss(logits, labels)
    assert loss.item() < 1e-5

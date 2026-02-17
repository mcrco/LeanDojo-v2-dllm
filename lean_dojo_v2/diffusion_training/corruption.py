"""Corruption operators for discrete denoising objectives."""

from __future__ import annotations

import random
from typing import Dict, List, Sequence, Tuple

import torch


def mask_target_tokens(
    input_ids: torch.Tensor,
    target_mask: torch.Tensor,
    mask_token_id: int,
    mask_prob: float,
    generator: torch.Generator | None = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Mask a random subset of target positions and return denoising labels.

    Returns:
        corrupted_input_ids: input ids with a subset of target tokens replaced by mask id
        labels: original token ids on masked target positions, -100 elsewhere
    """
    if input_ids.shape != target_mask.shape:
        raise ValueError("input_ids and target_mask must have the same shape")

    probs = torch.full_like(input_ids, fill_value=mask_prob, dtype=torch.float32)
    sampled = torch.bernoulli(probs, generator=generator).bool()
    masked_positions = sampled & target_mask.bool()

    labels = torch.full_like(input_ids, fill_value=-100)
    labels[masked_positions] = input_ids[masked_positions]

    corrupted = input_ids.clone()
    corrupted[masked_positions] = mask_token_id
    return corrupted, labels


def corrupt_tactic_script(
    tactics: Sequence[str],
    num_holes: int = 1,
    max_hole_len: int = 3,
    hole_token_template: str = "<HOLE_{i}>",
    rng: random.Random | None = None,
) -> Tuple[List[str], List[Dict[str, List[str] | str]]]:
    """Create span-corrupted tactic scripts for infilling.

    Returns:
        corrupted_tactics: script containing hole markers
        targets: list of {"hole_id": marker, "original_span": [..]}
    """
    if not tactics:
        return [], []

    if rng is None:
        rng = random.Random()

    n = len(tactics)
    starts = list(range(n))
    rng.shuffle(starts)

    occupied = [False] * n
    spans: List[Tuple[int, int, str]] = []
    for idx in range(min(num_holes, n)):
        start = starts[idx]
        if occupied[start]:
            continue
        span_len = rng.randint(1, max(1, max_hole_len))
        end = min(n, start + span_len)
        if any(occupied[start:end]):
            continue
        for i in range(start, end):
            occupied[i] = True
        spans.append((start, end, hole_token_template.format(i=len(spans) + 1)))

    if not spans:
        marker = hole_token_template.format(i=1)
        return [marker], [{"hole_id": marker, "original_span": list(tactics)}]

    spans.sort(key=lambda x: x[0])
    corrupted: List[str] = []
    targets: List[Dict[str, List[str] | str]] = []
    cursor = 0
    for start, end, marker in spans:
        while cursor < start:
            corrupted.append(tactics[cursor])
            cursor += 1
        corrupted.append(marker)
        targets.append({"hole_id": marker, "original_span": list(tactics[start:end])})
        cursor = end

    while cursor < n:
        corrupted.append(tactics[cursor])
        cursor += 1

    return corrupted, targets

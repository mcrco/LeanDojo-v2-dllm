"""Tests for whole-proof dataset reconstruction helpers."""

from lean_dojo_v2.diffusion_training.whole_proof_data import (
    build_whole_proof_examples,
    reconstruct_proof_from_tactics,
)


def test_reconstruct_proof_from_tactics_filters_sorry_and_dot():
    traced = [
        {"tactic": "intro h"},
        {"tactic": "· exact h"},
        {"tactic": "sorry"},
        {"tactic": "exact h"},
    ]
    proof = reconstruct_proof_from_tactics(traced, include_focused_tactics=False)
    assert proof == "by\n  intro h\n  exact h"


def test_build_whole_proof_examples_from_merged_items():
    items = [
        {
            "theorem_statement": "theorem t : True := by",
            "traced_tactics": [{"tactic": "exact trivial"}],
        }
    ]
    examples = build_whole_proof_examples(items, include_focused_tactics=True)
    assert len(examples) == 1
    assert examples[0]["prompt"] == "theorem t : True := by"
    assert examples[0]["target"] == "by\n  exact trivial"

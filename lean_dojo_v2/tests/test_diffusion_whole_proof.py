"""Tests for diffusion whole-proof generation helpers."""

from unittest.mock import MagicMock, patch

from lean_dojo_v2.prover.diffusion_prover import (
    DiffusionProver,
    _postprocess_whole_proof,
)


def test_postprocess_whole_proof_strips_fences_and_prefix():
    raw = "```lean\nHere's the proof:\nexact trivial\n```"
    assert _postprocess_whole_proof(raw) == "exact trivial"


def test_generate_whole_proof_returns_non_empty_with_mock_sampler():
    with patch.object(DiffusionProver, "__init__", lambda self, **kw: None):
        prover = DiffusionProver()
        prover.sampler = MagicMock()
        prover.sampler.sample_proof.return_value = ["```lean\nexact trivial\n```"]
        theorem = MagicMock()
        theorem.__str__ = lambda self: "theorem t : True := by"

        proof = prover.generate_whole_proof(theorem)
        assert proof == "exact trivial"


def test_infill_sorry_returns_non_empty_with_mock_sampler():
    with patch.object(DiffusionProver, "__init__", lambda self, **kw: None):
        prover = DiffusionProver()
        prover.sampler = MagicMock()
        prover.sampler.sample_proof.return_value = ["by\n  exact trivial"]

        proof = prover.infill_sorry(prefix="theorem t : True := by\n", suffix="\n")
        assert proof.startswith("by")

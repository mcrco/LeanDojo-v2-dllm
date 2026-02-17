"""Shared prompt formatting and output post-processing helpers."""

from __future__ import annotations

from typing import Iterable, List


TACTIC_SYSTEM_PROMPT = (
    "You are a Lean 4 tactic generator. Given a goal state, "
    "output exactly ONE Lean tactic that advances or solves the goal.\n"
    "Rules:\n"
    "- Output only the tactic text; no prose, quotes, or code fences.\n"
    "- Single line only; no `by` blocks.\n"
    "- Never use `sorry` or `admit`.\n"
)

WHOLE_PROOF_SYSTEM_PROMPT = (
    "Given a theorem statement, "
    "output the complete proof of the theorem in Lean 4 code.\n"
    "Only output the proof, no explanation, no comments, no theorem, nothing else."
)


def format_tactic_prompt(goal_str: str) -> str:
    """Format a next-tactic prompt using the repo's canonical template."""
    return (
        "### System:\n"
        f"{TACTIC_SYSTEM_PROMPT}"
        "### User:\n"
        f"{goal_str}\n\n"
        "### Assistant:\n"
    )


def format_whole_proof_prompt(theorem_str: str) -> str:
    """Format a whole-proof prompt using the repo's canonical template."""
    return (
        "### System:\n"
        f"{WHOLE_PROOF_SYSTEM_PROMPT}"
        "### User:\n"
        f"{theorem_str}\n\n"
        "### Assistant:\n"
    )


def postprocess_tactic_candidates(raw_samples: Iterable[str]) -> List[str]:
    """Convert raw model outputs into valid one-line Lean tactics."""
    tactics: List[str] = []
    for text in raw_samples:
        tactic = text.strip()
        tactic = tactic.split("\n")[0].split("<;>")[0].strip()
        if tactic and tactic not in {"sorry", "admit"}:
            tactics.append(tactic)
    return tactics

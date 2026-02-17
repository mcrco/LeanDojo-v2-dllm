"""Utilities for constructing theorem->proof pairs from merged exports."""

from __future__ import annotations

from typing import Any, Dict, List

from .formatting import normalize_tactic_target


def reconstruct_proof_from_tactics(
    traced_tactics: List[Dict[str, Any]],
    include_focused_tactics: bool = False,
) -> str:
    """Rebuild a simple `by` script from traced tactic steps."""
    script_lines: List[str] = []
    for entry in traced_tactics:
        tactic = normalize_tactic_target(entry.get("tactic", ""))
        if not tactic or tactic in {"sorry", "admit"}:
            continue
        if (not include_focused_tactics) and "·" in tactic:
            continue
        script_lines.append(f"  {tactic}")

    if not script_lines:
        return ""
    return "by\n" + "\n".join(script_lines)


def build_whole_proof_examples(
    merged_items: List[Dict[str, Any]],
    include_focused_tactics: bool = False,
) -> List[Dict[str, str]]:
    """Build theorem/prompt/target examples for whole-proof training."""
    examples: List[Dict[str, str]] = []
    for item in merged_items:
        theorem_statement = item.get("theorem_statement", "").strip()
        if not theorem_statement:
            continue
        proof = reconstruct_proof_from_tactics(
            traced_tactics=item.get("traced_tactics", []),
            include_focused_tactics=include_focused_tactics,
        )
        if not proof:
            continue
        examples.append(
            {
                "prompt": theorem_statement,
                "target": proof,
            }
        )
    return examples

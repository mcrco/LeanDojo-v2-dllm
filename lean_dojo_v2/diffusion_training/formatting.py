"""Prompt/data formatting shared by diffusion training objectives."""

from __future__ import annotations

from typing import Iterable, List

from lean_dojo_v2.prompting import format_tactic_prompt


def _remove_marks(text: str) -> str:
    return text.replace("<a>", "").replace("</a>", "")


def format_next_tactic_prompt(state_before: str) -> str:
    """Canonical next-tactic prompt used by both training and inference."""
    return format_tactic_prompt(_remove_marks(state_before).strip())


def normalize_tactic_target(tactic: str) -> str:
    """Normalize tactic target to a single line with no empty output."""
    return _remove_marks(tactic).splitlines()[0].strip()


def format_infill_prompt(
    corrupted_tactics: Iterable[str],
    theorem_statement: str | None = None,
) -> str:
    """Format an infilling prompt over a corrupted tactic script."""
    lines: List[str] = []
    if theorem_statement:
        lines.append("### Theorem:")
        lines.append(_remove_marks(theorem_statement).strip())
        lines.append("")

    lines.append("### Corrupted tactic script:")
    lines.extend(f"- {t}" for t in corrupted_tactics)
    lines.append("")
    lines.append("### Task:")
    lines.append("Recover each <HOLE_i> span as Lean tactics.")
    lines.append("Output only hole completions, one line per hole.")
    return "\n".join(lines)

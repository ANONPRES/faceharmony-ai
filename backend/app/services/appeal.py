"""Appeal score: attractiveness from the four FaceIQ-style pillars."""

from __future__ import annotations

from typing import Any

from .gender_ideals import Gender
from .pillars import appeal_from_pillars, compute_pillars


def appeal_score(
    scores: dict[str, float],
    gender: Gender,
    pillars: dict[str, float] | None = None,
) -> dict[str, Any]:
    """
    Build Appeal (0–100 + 0–10) from Harmony / Angularity / Dimorphism / Features.

    Soft ceilings removed — elite faces can clear 9/10 when all pillars are strong.
    """
    p = pillars if pillars is not None else compute_pillars(scores, gender)
    return appeal_from_pillars(p, gender)

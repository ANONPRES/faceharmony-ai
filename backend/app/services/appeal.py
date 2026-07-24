"""Appeal score: educational attractiveness proxy from geometry only.

Combines eye halo, jaw/angularity, cheekbones, midface, dimorphism alignment
with the user-selected gender, and a symmetry penalty. Not a beauty verdict.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .gender_ideals import Gender
from .scoring import power_mean


def appeal_score(
    scores: dict[str, float],
    gender: Gender,
) -> dict[str, Any]:
    """
    Build Appeal metric (0–100 + 0–10) with a harsh curve.

    Soft ceiling ~85 so 8.5/10 is rare; weak features pull hard (p=0.5).
    """
    eye = scores.get("eye_cut", 50.0)
    jaw = scores.get("jaw", 50.0)
    cheek = scores.get("cheekbones", 50.0)
    mid = scores.get("midface", 50.0)
    shape = scores.get("face_shape", 50.0)
    chin = scores.get("chin", 50.0)
    sym = scores.get("symmetry", 70.0)
    nose = scores.get("nose", 50.0)
    lips = scores.get("lips", 50.0)

    # Dimorphism proxy: how well angular vs soft traits match selected gender.
    angular = 0.4 * jaw + 0.35 * cheek + 0.25 * chin
    soft = 0.45 * lips + 0.30 * eye + 0.25 * mid
    if gender == "male":
        dimorphism = 0.65 * angular + 0.35 * soft
        dimo_note = "мужской акцент: челюсть/скулы"
    else:
        dimorphism = 0.55 * soft + 0.45 * angular
        dimo_note = "женский акцент: губы/глаза при мягкой структуре"

    parts = [
        (eye, 0.22),
        (jaw, 0.18),
        (cheek, 0.14),
        (dimorphism, 0.16),
        (mid, 0.10),
        (shape, 0.08),
        (nose, 0.07),
        (sym, 0.05),
    ]
    raw = power_mean(parts, p=0.50)

    # Symmetry / weak-eye penalties.
    if sym < 60:
        raw -= (60.0 - sym) * 0.15
    if eye < 55:
        raw -= (55.0 - eye) * 0.12

    # Soft ceiling — elite appeal stays rare.
    if raw > 82:
        raw = 82.0 + (raw - 82.0) * 0.28
    elif raw > 72:
        raw = 72.0 + (raw - 72.0) * 0.55

    score100 = float(np.clip(raw, 0.0, 100.0))
    score10 = round(score100 / 10.0, 1)

    return {
        "score": round(score100, 1),
        "score_10": score10,
        "label": "Appeal",
        "explanation": (
            f"Геометрический proxy привлекательности ({dimo_note}). "
            f"Глаза {eye:.0f}, челюсть {jaw:.0f}, скулы {cheek:.0f}, "
            f"диморфизм {dimorphism:.0f}. Не вердикт о человеке."
        ),
        "ratio": round(score10, 2),
    }

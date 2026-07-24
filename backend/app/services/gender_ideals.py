"""Gender-specific ideal bands for facial measurements and feature scorers.

Manual gender from the client (male | female). Shared FaceIQ-style bands stay
as defaults; only dimorphism-sensitive metrics are overridden.
"""

from __future__ import annotations

from typing import Literal

Gender = Literal["male", "female"]

# id -> (ideal_min, ideal_max) overrides
IDEAL_OVERRIDES: dict[Gender, dict[str, tuple[float, float]]] = {
    "male": {
        # Keep FaceIQ universal bands for JFA / one-eye-apart / canthal —
        # gender only shifts truly dimorphic soft-tissue ideals.
        "gonial_angle": (110.0, 125.0),
        "jaw_width_bigonial": (88.0, 93.0),
        "cheek_jaw_ratio": (1.05, 1.12),
        "chin_taper": (0.30, 0.42),
        "lip_ratio": (1.30, 1.50),
        "brow_eye_gap": (2.5, 4.8),
        "face_ratio": (0.72, 0.78),
        "mouth_nose_width": (1.45, 1.55),
    },
    "female": {
        "gonial_angle": (118.0, 135.0),
        "jaw_width_bigonial": (84.0, 90.0),
        "cheek_jaw_ratio": (1.10, 1.20),
        "chin_taper": (0.24, 0.36),
        "lip_ratio": (1.40, 1.70),
        "brow_eye_gap": (4.0, 6.5),
        "face_ratio": (0.68, 0.75),
        "mouth_nose_width": (1.40, 1.60),
    },
}


def resolve_ideal(
    gender: Gender | None,
    mid: str,
    default_min: float,
    default_max: float,
) -> tuple[float, float]:
    """Return gender-aware ideal band, falling back to defaults."""
    if gender is None:
        return default_min, default_max
    override = IDEAL_OVERRIDES.get(gender, {}).get(mid)
    if override is None:
        return default_min, default_max
    return override


def gender_label(gender: Gender) -> str:
    return "мужчина" if gender == "male" else "женщина"

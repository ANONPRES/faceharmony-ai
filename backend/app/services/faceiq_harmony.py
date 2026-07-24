"""FaceIQ-style Harmony from frontal measurement scores.

FaceIQ currently publishes only Harmony (Angularity / Dimorphism / Features /
Overall are Coming Soon). Front harmony is a weak-link-sensitive blend of the
ratio scores in their frontal categories (thirds, face shape, eyes, nose,
mouth, jaw, other).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .scoring import power_mean

# Map our measurement ids → FaceIQ frontal category weights.
# Weights mirror FaceIQ ratio-breakdown group sizes (approx).
_HARMONY_GROUPS: list[tuple[str, list[str], float]] = [
    (
        "thirds",
        ["upper_third", "mid_third", "lower_third", "mid_third_ratio"],
        0.14,
    ),
    (
        "face_shape",
        ["face_wh_cheek", "total_face_wh", "face_ratio", "forehead_width", "cheekbone_height"],
        0.16,
    ),
    (
        "eyes",
        ["eye_spacing", "eye_aspect", "canthal_tilt", "brow_eye_gap", "brow_width"],
        0.18,
    ),
    (
        "nose",
        ["iaa", "iaa_jfa_diff", "nose_width_bridge", "nose_eye_width", "nose_length", "nose_deviation"],
        0.14,
    ),
    (
        "mouth",
        ["mouth_nose_width", "lip_ratio", "philtrum", "mouth_face_width", "mouth_corners"],
        0.14,
    ),
    (
        "jaw",
        ["jfa", "jaw_width_bigonial", "chin_share", "chin_taper", "cheek_jaw_ratio", "lower_face_total"],
        0.16,
    ),
    (
        "other",
        ["sym_eyes", "sym_brows", "sym_mouth", "sym_jaw", "sym_cheeks", "golden_face", "neck_width"],
        0.08,
    ),
]


def _group_score(by_id: dict[str, dict[str, Any]], ids: list[str]) -> float | None:
    scores = []
    for mid in ids:
        m = by_id.get(mid)
        if not m:
            continue
        # Prefer score_10 * 10; fall back to score.
        if "score_10" in m:
            scores.append(float(m["score_10"]) * 10.0)
        elif "score" in m:
            scores.append(float(m["score"]))
    if not scores:
        return None
    # Within-group: mild weak-link (FaceIQ category sub-scores punish outliers).
    parts = [(s, 1.0) for s in scores]
    return float(power_mean(parts, p=0.55))


def harmony_from_measurements(measurements: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute FaceIQ-style frontal Harmony 0–100 / 0–10 from detailed rows.

    Only frontal (or unspecified) measurements contribute; profile ceph rows
    are ignored here (side harmony is separate when a profile photo exists).
    """
    by_id: dict[str, dict[str, Any]] = {}
    for m in measurements:
        if m.get("view") == "profile":
            continue
        mid = m.get("id")
        if isinstance(mid, str):
            by_id[mid] = m

    group_parts: list[tuple[float, float]] = []
    breakdown: dict[str, float] = {}
    for name, ids, weight in _HARMONY_GROUPS:
        g = _group_score(by_id, ids)
        if g is None:
            continue
        group_parts.append((g, weight))
        breakdown[name] = round(g / 10.0, 2)

    if not group_parts:
        return {
            "score": 55.0,
            "score_10": 5.5,
            "breakdown": {},
            "explanation": "Недостаточно фронтальных метрик для Harmony.",
        }

    raw = float(power_mean(group_parts, p=0.50))
    ordered = sorted(s for s, _ in group_parts)
    weak = float(np.mean(ordered[: max(1, len(ordered) // 3)]))
    # Mild weak-link — FaceIQ front still drops for bad categories, but one
    # MediaPipe-noisy group shouldn't dominate (OEA often runs high on mesh).
    blended = 0.72 * raw + 0.28 * weak

    # Soft curve toward FaceIQ front scale (~6.5–8.7 for models).
    x = float(np.clip(blended, 0.0, 100.0)) / 100.0
    scored = 100.0 * (x**1.08)
    scored = float(np.clip(scored, 0.0, 99.0))

    score10 = round(scored / 10.0, 1)
    return {
        "score": round(scored, 1),
        "score_10": score10,
        "breakdown": breakdown,
        "explanation": (
            "FaceIQ-стиль Harmony (только анфас): трети / форма / глаза / нос / "
            f"рот / челюсть. Слабые группы тянут вниз. "
            + ", ".join(f"{k} {v:.1f}" for k, v in breakdown.items())
        ),
    }

"""FaceIQ-style Harmony from frontal measurement scores.

FaceIQ currently publishes only Harmony (Angularity / Dimorphism / Features /
Overall are Coming Soon). Front harmony is a weak-link-sensitive blend of the
ratio scores in their frontal categories (thirds, face shape, eyes, nose,
mouth, jaw, other).
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Map our measurement ids → FaceIQ frontal category weights.
# Weights mirror FaceIQ ratio-breakdown group sizes (approx).
# Only FaceIQ-published frontal ratios — avoid diluting with filler metrics.
_HARMONY_GROUPS: list[tuple[str, list[str], float]] = [
    (
        "thirds",
        ["upper_third", "mid_third", "lower_third"],
        0.12,
    ),
    (
        "face_shape",
        ["face_wh_cheek", "total_face_wh", "cheekbone_height"],
        0.14,
    ),
    (
        "eyes",
        [
            "eye_spacing",
            "eye_aspect",
            "canthal_tilt",
            "outer_eye_span",
            "brow_tilt",
            "brow_low_set",
            "brow_width",
        ],
        0.18,
    ),
    (
        "nose",
        ["iaa", "iaa_jfa_diff", "nose_width_bridge", "intercanthal_nasal"],
        0.14,
    ),
    (
        "mouth",
        ["mouth_nose_width", "lip_ratio", "cupid_bow", "chin_philtrum", "mouth_corners"],
        0.14,
    ),
    (
        "jaw",
        ["jfa", "jaw_width_bigonial", "lower_face_total"],
        0.16,
    ),
    (
        "other",
        ["neck_width"],
        0.12,
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
    # FaceIQ category tiles use near-arithmetic means (Sean Eyes ≈ 8.9 with OEA 2.8).
    return float(np.mean(scores))


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

    # Weighted arithmetic mean of category tiles, then mild weak-link toward
    # the lowest third (FaceIQ front Sean 7.6 with strong eyes but weak OEA).
    raw = float(np.average([s for s, _ in group_parts], weights=[w for _, w in group_parts]))
    ordered = sorted(s for s, _ in group_parts)
    weak = float(np.mean(ordered[: max(1, len(ordered) // 3)]))
    blended = 0.70 * raw + 0.30 * weak

    x = float(np.clip(blended, 0.0, 100.0)) / 100.0
    scored = 100.0 * (x**1.10)
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

"""Four-pillar attractiveness scoring (FaceIQ / PSL-style).

Pillars
-------
1. Harmony   — proportions, symmetry, balance
2. Angularity — bone / jaw / cheek structure
3. Dimorphism — gender-typical traits
4. Features  — eyes, nose, lips as individual assets

Overall / Appeal are derived from pillars with a rarity curve so elite faces
can reach the high 80s–90s while ordinary captures stay mid-50s to low-70s.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .gender_ideals import Gender
from .scoring import power_mean


def _pm(scores: dict[str, float], keys: list[tuple[str, float]], p: float = 0.45) -> float:
    parts = [(float(scores.get(k, 55.0)), w) for k, w in keys]
    return float(power_mean(parts, p=p))


def compute_pillars(scores: dict[str, float], gender: Gender) -> dict[str, float]:
    """Compute 0–100 pillar scores from feature metrics."""
    harmony = _pm(
        scores,
        [
            ("symmetry", 0.28),
            ("thirds", 0.18),
            ("fifths", 0.12),
            ("golden_ratio", 0.14),
            ("face_ratio", 0.12),
            ("midface", 0.10),
            ("eyes", 0.06),
        ],
        p=0.50,
    )

    angularity = _pm(
        scores,
        [
            ("jaw", 0.30),
            ("cheekbones", 0.28),
            ("chin", 0.20),
            ("face_shape", 0.14),
            ("brow", 0.08),
        ],
        p=0.42,
    )

    eye = float(scores.get("eye_cut", 55.0))
    jaw = float(scores.get("jaw", 55.0))
    cheek = float(scores.get("cheekbones", 55.0))
    chin = float(scores.get("chin", 55.0))
    lips = float(scores.get("lips", 55.0))
    mid = float(scores.get("midface", 55.0))
    brow = float(scores.get("brow", 55.0))

    angular_block = 0.40 * jaw + 0.30 * cheek + 0.20 * chin + 0.10 * brow
    soft_block = 0.40 * lips + 0.35 * eye + 0.25 * mid
    if gender == "male":
        # Male dimorphism rewards angular structure; soft traits still matter a bit.
        dimorphism = 0.72 * angular_block + 0.28 * soft_block
    else:
        dimorphism = 0.62 * soft_block + 0.38 * angular_block

    features = _pm(
        scores,
        [
            ("eye_cut", 0.40),
            ("nose", 0.35),
            ("lips", 0.25),
        ],
        p=0.45,
    )

    return {
        "harmony": float(np.clip(harmony, 0.0, 100.0)),
        "angularity": float(np.clip(angularity, 0.0, 100.0)),
        "dimorphism": float(np.clip(dimorphism, 0.0, 100.0)),
        "features": float(np.clip(features, 0.0, 100.0)),
    }


def rarity_curve(raw: float) -> float:
    """
    Expand dynamic range so mid faces don't pile up near elites.

    Intended landmarks (after pillar power-mean):
      ~55 raw → ~52 overall   (below average)
      ~65 raw → ~60           (average)
      ~75 raw → ~71           (solid)
      ~82 raw → ~80           (clearly attractive)
      ~88 raw → ~87           (model-tier)
      ~94 raw → ~93           (near-ideal, rare)
    """
    x = float(np.clip(raw, 0.0, 100.0)) / 100.0
    # Super-linear: mid scores drop, top scores stay high and separable.
    y = 100.0 * (x**1.38)
    # Gentle lift at the very top so true elites aren't capped too early.
    if y > 86:
        y = 86.0 + (y - 86.0) * 1.15
    return float(np.clip(y, 0.0, 99.5))


def overall_from_pillars(pillars: dict[str, float]) -> float:
    """
    Combine pillars → overall.

    Weak-link sensitive. Angularity + dimorphism carry more weight so
    model bone structure outruns 'good eyes/nose on an average frame'.
    """
    h = pillars["harmony"]
    a = pillars["angularity"]
    d = pillars["dimorphism"]
    f = pillars["features"]

    # Features can't float far above bone structure (common inflation pattern).
    f_eff = min(f, a + 5.0, d + 5.0)

    raw = power_mean(
        [
            (h, 0.30),
            (a, 0.30),
            (d, 0.26),
            (f_eff, 0.14),
        ],
        p=0.40,
    )

    weakest = min(h, a, d, f_eff)
    if weakest < 78:
        raw = 0.50 * raw + 0.50 * weakest
    elif weakest < 88:
        raw = 0.65 * raw + 0.35 * weakest

    mean_p = 0.25 * (h + a + d + f_eff)
    spread = float(np.std([h, a, d, f_eff]))

    # Bone-structure bonus (Sean / Barrett class).
    if a >= 90 and d >= 90:
        raw += 4.0
    elif a >= 87 and d >= 87:
        raw += 2.5
    elif a >= 84 and d >= 84:
        raw += 1.0

    if weakest >= 91 and mean_p >= 92 and spread <= 4.0:
        raw += 3.0
    elif weakest >= 88 and mean_p >= 89 and spread <= 5.5:
        raw += 1.5

    return rarity_curve(raw)


def appeal_from_pillars(pillars: dict[str, float], gender: Gender) -> dict[str, Any]:
    """
    Appeal = attractiveness from the four pillars (0–100 / 0–10).

    Slightly more weight on angularity + features than pure harmony.
    """
    h = pillars["harmony"]
    a = pillars["angularity"]
    d = pillars["dimorphism"]
    f = pillars["features"]
    f_eff = min(f, a + 6.0, d + 6.0)

    raw = power_mean(
        [
            (h, 0.26),
            (a, 0.30),
            (d, 0.26),
            (f_eff, 0.18),
        ],
        p=0.38,
    )
    weakest = min(h, a, d, f_eff)
    if weakest < 80:
        raw = 0.48 * raw + 0.52 * weakest

    if a >= 90 and d >= 90:
        raw += 3.5
    elif a >= 87 and d >= 87:
        raw += 2.0

    score100 = rarity_curve(raw)
    score10 = round(score100 / 10.0, 1)

    return {
        "score": round(score100, 1),
        "score_10": score10,
        "label": "Appeal",
        "explanation": (
            f"4 столпа (FaceIQ-стиль): Harmony {h:.0f}, Angularity {a:.0f}, "
            f"Dimorphism {d:.0f}, Features {f:.0f}. "
            f"{'Мужской' if gender == 'male' else 'Женский'} акцент диморфизма."
        ),
        "ratio": score10,
        "pillars": {k: round(v, 1) for k, v in pillars.items()},
    }


def pack_pillar_metrics(pillars: dict[str, float]) -> dict[str, dict[str, Any]]:
    """MetricDetail-shaped entries for the four pillars."""
    labels = {
        "harmony": ("Harmony", "Пропорции, симметрия, баланс частей лица."),
        "angularity": ("Angularity", "Костная структура: челюсть, скулы, подбородок."),
        "dimorphism": ("Dimorphism", "Насколько черты типичны для выбранного пола."),
        "features": ("Features", "Глаза, нос, губы как отдельные фичи."),
    }
    out: dict[str, dict[str, Any]] = {}
    for key, (label, expl) in labels.items():
        score = float(pillars[key])
        out[key] = {
            "score": round(score, 1),
            "label": label,
            "explanation": expl,
            "ratio": round(score / 10.0, 2),
        }
    return out

"""Four-pillar attractiveness scoring (FaceIQ / PSL-style).

Pillars
-------
1. Harmony   — proportions, symmetry, balance
2. Angularity — bone / jaw / cheek structure
3. Dimorphism — gender-typical traits
4. Features  — eyes, nose, lips as individual assets

Overall / Appeal are derived from pillars with a rarity curve so elite faces
can reach the high 80s–90s while soft / average captures stay ~50–68 — not 80+.
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
        p=0.42,
    )

    eye = float(scores.get("eye_cut", 55.0))
    jaw = float(scores.get("jaw", 55.0))
    cheek = float(scores.get("cheekbones", 55.0))
    chin = float(scores.get("chin", 55.0))
    lips = float(scores.get("lips", 55.0))
    mid = float(scores.get("midface", 55.0))
    brow = float(scores.get("brow", 55.0))
    face_shape = float(scores.get("face_shape", 55.0))

    # Jaw width alone is easy to max on soft faces; elite angularity needs
    # cheekbones + chin in the same band (FaceIQ bone-consistency).
    if cheek < 92:
        jaw = min(jaw, cheek + 5.0)
        chin = min(chin, cheek + 8.0)
        face_shape = min(face_shape, cheek + 10.0)
    if cheek < 82:
        # Soft / low zygoma: further crush "square jaw" inflation.
        jaw = min(jaw, cheek + 2.0)
        chin = min(chin, cheek + 4.0)

    angularity = power_mean(
        [
            (jaw, 0.28),
            (cheek, 0.34),
            (chin, 0.20),
            (face_shape, 0.12),
            (brow, 0.06),
        ],
        p=0.32,
    )

    angular_block = 0.38 * jaw + 0.36 * cheek + 0.18 * chin + 0.08 * brow
    soft_block = 0.40 * lips + 0.35 * eye + 0.25 * mid
    if gender == "male":
        # Male dimorphism rewards angular structure; soft traits still matter a bit.
        dimorphism = 0.78 * angular_block + 0.22 * soft_block
    else:
        dimorphism = 0.62 * soft_block + 0.38 * angular_block

    features = _pm(
        scores,
        [
            ("eye_cut", 0.40),
            ("nose", 0.35),
            ("lips", 0.25),
        ],
        p=0.40,
    )
    # Pretty eyes/nose can't carry Features far above bone.
    features = min(features, angularity + 4.0, cheek + 12.0)

    return {
        "harmony": float(np.clip(harmony, 0.0, 100.0)),
        "angularity": float(np.clip(angularity, 0.0, 100.0)),
        "dimorphism": float(np.clip(dimorphism, 0.0, 100.0)),
        "features": float(np.clip(features, 0.0, 100.0)),
    }


def rarity_curve(raw: float) -> float:
    """
    Crush mid-band inflation; keep true elites separable at the top.

    Intended landmarks (after pillar blend):
      ~55 raw → ~42 overall   (below average)
      ~65 raw → ~52           (average)
      ~75 raw → ~60           (solid / above avg)
      ~82 raw → ~68           (attractive — not elite)
      ~88 raw → ~78           (strong)
      ~92 raw → ~90           (model-tier)
      ~96 raw → ~97           (near-ideal)
    """
    x = float(np.clip(raw, 0.0, 100.0)) / 100.0
    # Mid crush; steeper than ^1.38 so soft faces don't sit at 80+.
    y = 100.0 * (x**1.92)
    # Elite lift once raw clears model threshold.
    if raw >= 88.5:
        over = raw - 88.5
        y = max(y, 76.0 + over * 1.75)
        if raw >= 92.0:
            y += (raw - 92.0) * 0.55
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
    f_eff = min(f, a + 3.0, d + 3.0)

    raw = power_mean(
        [
            (h, 0.26),
            (a, 0.34),
            (d, 0.28),
            (f_eff, 0.12),
        ],
        p=0.32,
    )

    weakest = min(h, a, d, f_eff)
    bone = 0.5 * (a + d)
    # Mid faces often have flat high pillars — pull hard toward the weak/bone floor.
    if weakest < 82:
        raw = 0.32 * raw + 0.48 * weakest + 0.20 * bone
    elif weakest < 90:
        raw = 0.48 * raw + 0.38 * weakest + 0.14 * bone
    elif weakest < 94:
        raw = 0.62 * raw + 0.38 * weakest

    mean_p = 0.25 * (h + a + d + f_eff)
    spread = float(np.std([h, a, d, f_eff]))

    # Bone-structure bonus for model-tier A+D.
    if a >= 92 and d >= 91:
        raw += 5.0
    elif a >= 89 and d >= 89:
        raw += 3.0

    if weakest >= 93 and mean_p >= 94 and spread <= 3.5:
        raw += 3.0
    elif weakest >= 90 and mean_p >= 91 and spread <= 4.5:
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
    f_eff = min(f, a + 3.0, d + 3.0)

    raw = power_mean(
        [
            (h, 0.24),
            (a, 0.34),
            (d, 0.26),
            (f_eff, 0.16),
        ],
        p=0.30,
    )
    weakest = min(h, a, d, f_eff)
    if weakest < 82:
        raw = 0.35 * raw + 0.65 * weakest
    elif weakest < 90:
        raw = 0.50 * raw + 0.50 * weakest

    if a >= 92 and d >= 91:
        raw += 3.5
    elif a >= 89 and d >= 89:
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

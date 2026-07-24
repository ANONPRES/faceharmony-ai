"""Four-pillar attractiveness scoring (FaceIQ / PSL-style).

Pillars (canonical definitions)
------------------------------
1. Harmony    — how features fit together: symmetry, thirds/fifths, ratios.
2. Dimorphism — how male/female the face reads for the *selected* gender
                (direction of traits, not just “sharpness”).
3. Angularity — bone definition / sharpness: jaw, zygoma, chin, contours
                (can be high on either gender — e.g. model angularity).
4. Features   — individual asset quality (eyes, nose, lips). FaceIQ “Features”
                / PSL “Miscellaneous” without skin (we lack a skin scorer).

Overall weights (common PSL blend): Harmony 30%, Dimorphism 30%,
Angularity 25%, Features 15%, plus a looks-penalty for pillar spread.
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
    # --- Harmony: balance / proportions (front-profile heavy) ---
    # FaceIQ / CA-style: thirds, FWHR-ish face_ratio, midface, symmetry, fifths.
    harmony = _pm(
        scores,
        [
            ("symmetry", 0.24),
            ("thirds", 0.20),
            ("face_ratio", 0.14),
            ("midface", 0.12),
            ("fifths", 0.12),
            ("golden_ratio", 0.10),
            ("eyes", 0.08),  # interocular / eye spacing harmony
        ],
        p=0.38,
    )

    jaw = float(scores.get("jaw", 55.0))
    cheek = float(scores.get("cheekbones", 55.0))
    chin = float(scores.get("chin", 55.0))
    brow = float(scores.get("brow", 55.0))
    face_shape = float(scores.get("face_shape", 55.0))
    eye = float(scores.get("eye_cut", 55.0))
    nose = float(scores.get("nose", 55.0))
    lips = float(scores.get("lips", 55.0))
    mid = float(scores.get("midface", 55.0))
    face_ratio = float(scores.get("face_ratio", 55.0))

    # --- Angularity: sharpness / bone protrusion / contours ---
    # Independent of gender. Jaw width alone ≠ angularity — zygoma + chin matter.
    # Soft faces often max jaw score; keep cheekbones as the consistency anchor.
    jaw_a = jaw
    chin_a = chin
    shape_a = face_shape
    if cheek < 92:
        jaw_a = min(jaw, cheek + 6.0)
        chin_a = min(chin, cheek + 8.0)
        shape_a = min(face_shape, cheek + 10.0)
    if cheek < 82:
        jaw_a = min(jaw_a, cheek + 2.0)
        chin_a = min(chin_a, cheek + 4.0)

    angularity = power_mean(
        [
            (cheek, 0.32),      # zygomatic prominence / height
            (jaw_a, 0.26),      # mandibular definition
            (chin_a, 0.18),     # menton / taper
            (shape_a, 0.12),    # overall contour class
            (eye, 0.08),        # eye angularity (hunter vs round)
            (brow, 0.04),       # brow ridge mass (subtle)
        ],
        p=0.30,
    )

    # --- Dimorphism: gender-typical *direction* ---
    # Must diverge from angularity: a feminine face can be very angular (FaceIQ
    # examples), and a soft masculine face can still read male.
    # Feature scorers already use gender ideals for jaw/chin/lips/canthal —
    # weights emphasize markers that differentiate sex, not just sharpness.
    if gender == "male":
        # Male: square jaw, chin projection, brow, zygoma width, hunter eyes,
        # compact midface; lips scored to male (thinner) ideal already.
        dimorphism = power_mean(
            [
                (jaw, 0.24),
                (chin, 0.18),
                (brow, 0.16),
                (cheek, 0.16),
                (eye, 0.12),
                (face_ratio, 0.08),
                (mid, 0.06),
            ],
            p=0.34,
        )
        # Soft / full lips (high female-coded fullness) mildly pull male dimo down
        # when lips scorer is still high from mouth width alone.
        if lips >= 92 and jaw < 90:
            dimorphism -= 2.0
    else:
        # Female: fuller lips, eye cut/openness, softer lower third still
        # harmonious, zygoma, midface; jaw uses female ideal (more taper).
        dimorphism = power_mean(
            [
                (lips, 0.24),
                (eye, 0.18),
                (cheek, 0.16),
                (mid, 0.12),
                (jaw, 0.12),
                (chin, 0.10),
                (brow, 0.08),
            ],
            p=0.34,
        )
        if jaw >= 94 and lips < 80:
            dimorphism -= 2.0

    # --- Features (Misc): individual assets, not bone ---
    # FaceIQ Features / PSL Misc: eyes, nose, lips (skin/undereye omitted).
    # Do NOT hard-cap to angularity — pillars must be able to diverge.
    features = power_mean(
        [
            (eye, 0.42),
            (nose, 0.33),
            (lips, 0.25),
        ],
        p=0.40,
    )

    return {
        "harmony": float(np.clip(harmony, 0.0, 100.0)),
        "angularity": float(np.clip(angularity, 0.0, 100.0)),
        "dimorphism": float(np.clip(dimorphism, 0.0, 100.0)),
        "features": float(np.clip(features, 0.0, 100.0)),
    }


def rarity_curve(raw: float) -> float:
    """
    Mid-band crush so soft faces don't sit at 80+; no fake ceiling to 100.

    FaceIQ-style: model-tier overall often lands ~8.5–9.3/10 (85–93), not 10.0.
    Hard cap 99.0 — UI shows one decimal.
    """
    x = float(np.clip(raw, 0.0, 100.0)) / 100.0
    y = 100.0 * (x**1.55)
    return float(np.clip(y, 0.0, 99.0))


def _looks_penalty(pillars: dict[str, float]) -> float:
    """
    PSL-style looks penalty: (max_pillar − min_pillar) / 4 on 0–10 scale
    → (max − min) / 4 on 0–100 (same numeric drop in “points”).
    """
    vals = [
        pillars["harmony"],
        pillars["angularity"],
        pillars["dimorphism"],
        pillars["features"],
    ]
    return float(max(vals) - min(vals)) / 4.0


def overall_from_pillars(pillars: dict[str, float]) -> float:
    """
    Combine pillars → overall.

    PSL blend ≈ Harmony 30% / Dimorphism 30% / Angularity 25% / Features 15%,
    weak-link sensitive, looks-penalty for pillar spread.
    """
    h = pillars["harmony"]
    a = pillars["angularity"]
    d = pillars["dimorphism"]
    f = pillars["features"]

    raw = power_mean(
        [
            (h, 0.30),
            (d, 0.30),
            (a, 0.25),
            (f, 0.15),
        ],
        p=0.40,
    )

    weakest = min(h, a, d, f)
    # Soft faces with one strong pillar shouldn't average up to elite.
    if weakest < 78:
        raw = 0.40 * raw + 0.60 * weakest
    elif weakest < 88:
        raw = 0.55 * raw + 0.45 * weakest
    elif weakest < 93:
        raw = 0.82 * raw + 0.18 * weakest

    raw -= _looks_penalty(pillars)

    # Small consistency bonus only when all pillars are truly elite & tight.
    mean_p = 0.25 * (h + a + d + f)
    spread = float(np.std([h, a, d, f]))
    if weakest >= 93 and mean_p >= 94 and spread <= 3.0:
        raw += 1.2
    elif weakest >= 90 and mean_p >= 91 and spread <= 4.0:
        raw += 0.6

    return rarity_curve(raw)


def appeal_from_pillars(pillars: dict[str, float], gender: Gender) -> dict[str, Any]:
    """
    Appeal = attractiveness from the four pillars (0–100 / 0–10).

    Same PSL weights as overall; reported on a 0–10 scale for FaceIQ parity.
    """
    h = pillars["harmony"]
    a = pillars["angularity"]
    d = pillars["dimorphism"]
    f = pillars["features"]

    raw = power_mean(
        [
            (h, 0.30),
            (d, 0.30),
            (a, 0.25),
            (f, 0.15),
        ],
        p=0.38,
    )
    weakest = min(h, a, d, f)
    if weakest < 80:
        raw = 0.40 * raw + 0.60 * weakest
    elif weakest < 90:
        raw = 0.55 * raw + 0.45 * weakest

    raw -= _looks_penalty(pillars)

    score100 = rarity_curve(raw)
    score10 = round(score100 / 10.0, 1)

    return {
        "score": round(score100, 1),
        "score_10": score10,
        "label": "Appeal",
        "explanation": (
            f"PSL/FaceIQ 4 столпа: H {h / 10:.1f}, A {a / 10:.1f}, "
            f"D {d / 10:.1f}, F {f / 10:.1f} "
            f"({'♂' if gender == 'male' else '♀'} dimorphism). "
            f"Overall ≈ 30/30/25/15 + looks-penalty."
        ),
        "ratio": score10,
        "pillars": {k: round(v, 1) for k, v in pillars.items()},
    }


def pack_pillar_metrics(pillars: dict[str, float]) -> dict[str, dict[str, Any]]:
    """MetricDetail-shaped entries for the four pillars."""
    labels = {
        "harmony": (
            "Harmony",
            "Баланс и пропорции: симметрия, трети/пятые, midface, FWHR.",
        ),
        "angularity": (
            "Angularity",
            "Острота кости: скулы, челюсть, подбородок, контур (не пол).",
        ),
        "dimorphism": (
            "Dimorphism",
            "Насколько лицо читается как выбранный пол (♂/♀ маркеры).",
        ),
        "features": (
            "Features",
            "Отдельные фичи: глаза, нос, губы (Misc без кожи).",
        ),
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

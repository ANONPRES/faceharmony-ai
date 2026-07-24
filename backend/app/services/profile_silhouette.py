"""Profile scoring from automatic soft-tissue cephalometric landmarks.

Detection lives in `profile_ceph.py` (silhouette → Tr/G/N'/Prn/Sn/Ls/Li/Pog/Go).
This module turns those FaceIQ-style side metrics into 0–100 feature scores and
detailed measurement rows for the UI.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .profile_ceph import compute_side_ceph_metrics, detect_soft_tissue_profile
from .scoring import combine_scores, range_score


# Keep old name so callers/imports do not break.
def extract_profile_contour(bgr, tip_hint=None):  # type: ignore[no-untyped-def]
    """Back-compat wrapper around soft-tissue autodetection."""
    return detect_soft_tissue_profile(bgr, tip_hint=tip_hint)


def _score_band(value: float, lo: float, hi: float, soft: float | None = None) -> float:
    return float(range_score(value, lo, hi, soft))


def build_profile_silhouette_scores(calc: Any) -> dict[str, dict[str, Any]]:
    """Score a dedicated profile photo via automatic soft-tissue ceph metrics."""
    tip_hint = None
    try:
        tip_hint = calc.px("nose_tip")
    except Exception:
        tip_hint = None

    lm = None
    if getattr(calc, "bgr_image", None) is not None:
        lm = detect_soft_tissue_profile(calc.bgr_image, tip_hint=tip_hint)

    if lm is None:
        return _neutral_profile_pack("Силуэт профиля не выделен — нейтральная оценка.")

    gender = getattr(calc, "gender", "male")
    ceph = compute_side_ceph_metrics(lm, gender=gender)
    v = ceph["values"]
    ideals = ceph["ideals"]

    # Store on calculator for measurement export / overlay.
    calc.profile_landmarks = lm
    calc.profile_ceph = ceph

    nose = combine_scores(
        [
            (_score_band(v["nose_projection"], *ideals["nose_projection"]), 0.40),
            (_score_band(v["nose_tip_rotation"], *ideals["nose_tip_rotation"]), 0.30),
            (_score_band(v["nasal_tip_angle"], *ideals["nasal_tip_angle"], 12.0), 0.30),
        ]
    )

    chin = combine_scores(
        [
            (_score_band(v["lower_lip_e_line"], *ideals["lower_lip_e_line"], 2.5), 0.35),
            (_score_band(v["facial_convexity_nasion"], *ideals["facial_convexity_nasion"], 8.0), 0.35),
            (_score_band(v["lower_third"], *ideals["lower_third"], 0.06), 0.30),
        ]
    )

    jaw = _score_band(v["gonial_angle"], *ideals["gonial_angle"], 12.0)

    midface = combine_scores(
        [
            (_score_band(v["midface_projection_angle"], *ideals["midface_projection_angle"], 10.0), 0.55),
            (_score_band(v["mid_third"], 0.28, 0.36, 0.06), 0.45),
        ]
    )

    lips = combine_scores(
        [
            (_score_band(v["upper_lip_e_line"], *ideals["upper_lip_e_line"], 2.5), 0.35),
            (_score_band(v["lower_lip_e_line"], *ideals["lower_lip_e_line"], 2.5), 0.35),
            (_score_band(v["nasolabial_angle"], *ideals["nasolabial_angle"], 12.0), 0.30),
        ]
    )

    face_shape = combine_scores(
        [
            (_score_band(v["total_facial_convexity"], *ideals["total_facial_convexity"], 8.0), 0.45),
            (_score_band(v["facial_convexity_glabella"], *ideals["facial_convexity_glabella"], 8.0), 0.35),
            (_score_band(v["facial_convexity_nasion"], *ideals["facial_convexity_nasion"], 8.0), 0.20),
        ]
    )

    def pack(score: float, label: str, explanation: str, ratio: float | None = None) -> dict[str, Any]:
        return {
            "score": round(float(np.clip(score, 0.0, 100.0)), 1),
            "label": label,
            "explanation": explanation,
            "ratio": None if ratio is None else round(float(ratio), 4),
        }

    return {
        "nose": pack(
            nose,
            "Нос (профиль)",
            f"Проекция {v['nose_projection']:.3f}, tip rotation {v['nose_tip_rotation']:.0f}°.",
            v["nose_projection"],
        ),
        "chin": pack(
            chin,
            "Подбородок (профиль)",
            f"E-line lower {v['lower_lip_e_line']:.1f}%, convexity N–Sn–Pog {v['facial_convexity_nasion']:.0f}°.",
            v["lower_lip_e_line"],
        ),
        "jaw": pack(
            jaw,
            "Угол челюсти (профиль)",
            f"Gonial (Tragus–Go–Me) ≈ {v['gonial_angle']:.0f}° (идеал {ideals['gonial_angle'][0]:.0f}–{ideals['gonial_angle'][1]:.0f}°).",
            v["gonial_angle"],
        ),
        "midface": pack(
            midface,
            "Средняя зона (профиль)",
            f"Midface angle {v['midface_projection_angle']:.0f}°, mid third {v['mid_third']:.2f}.",
            v["midface_projection_angle"],
        ),
        "lips": pack(
            lips,
            "Губы (профиль)",
            f"E-line Ls {v['upper_lip_e_line']:.1f}% / Li {v['lower_lip_e_line']:.1f}%, nasolabial {v['nasolabial_angle']:.0f}°.",
            v["upper_lip_e_line"],
        ),
        "face_shape": pack(
            face_shape,
            "Контур профиля",
            f"Total convexity G–Prn–Pog {v['total_facial_convexity']:.0f}°.",
            v["total_facial_convexity"],
        ),
    }


def build_profile_ceph_measurements(calc: Any) -> list[dict[str, Any]]:
    """
    Emit FaceIQ-style side measurement rows for the results UI.

    Requires `calc.profile_ceph` from a prior `build_profile_silhouette_scores` call.
    """
    from .detailed_measurements import _pack, _pt, _seg

    ceph = getattr(calc, "profile_ceph", None)
    if not ceph:
        return []

    lm = ceph["landmarks"]
    v = ceph["values"]
    ideals = ceph["ideals"]
    out: list[dict[str, Any]] = []

    def add(
        mid: str,
        label: str,
        category: str,
        key: str,
        unit: str,
        explanation: str,
        point_ids: list[str],
        segments: list[tuple[str, str]],
        soft: float | None = None,
    ) -> None:
        lo, hi = ideals[key]
        pts = [_pt(calc, pid.lower(), lm[pid], "anchor") for pid in point_ids if pid in lm]
        segs = [
            _seg(calc, lm[a], lm[b], style="primary")
            for a, b in segments
            if a in lm and b in lm
        ]
        out.append(
            _pack(
                mid=mid,
                label=label,
                category=category,
                value=float(v[key]),
                unit=unit,
                ideal_min=lo,
                ideal_max=hi,
                explanation=explanation,
                points=pts,
                segments=segs,
                formula={"type": "profile_ceph", "key": key},
                soft_margin=soft,
                view="profile",
            )
        )

    add(
        "upper_lip_e_line",
        "Верхняя губа к E-line",
        "губы",
        "upper_lip_e_line",
        "%",
        "Ricketts E-line (Prn–Pog'). Отрицательное = позади линии. Авто с силуэта.",
        ["Prn", "Pog", "Ls"],
        [("Prn", "Pog")],
        2.5,
    )
    add(
        "lower_lip_e_line",
        "Нижняя губа к E-line",
        "губы",
        "lower_lip_e_line",
        "%",
        "Ricketts E-line (Prn–Pog'). Автоопределение soft-tissue точек.",
        ["Prn", "Pog", "Li"],
        [("Prn", "Pog")],
        2.5,
    )
    add(
        "nasolabial_angle",
        "Назолабиальный угол",
        "нос",
        "nasolabial_angle",
        "°",
        "Угол Prn–Sn–Ls. Классика soft-tissue профиля.",
        ["Prn", "Sn", "Ls"],
        [("Prn", "Sn"), ("Sn", "Ls")],
        12.0,
    )
    add(
        "gonial_angle_profile",
        "Gonial angle (профиль)",
        "челюсть",
        "gonial_angle",
        "°",
        "Угол ветви/тела: Tragion–Go–Me по авто-силуэту (как FaceIQ side).",
        ["Tragion", "Go", "Me"],
        [("Tragion", "Go"), ("Go", "Me")],
        12.0,
    )
    add(
        "total_facial_convexity",
        "Total facial convexity",
        "профиль",
        "total_facial_convexity",
        "°",
        "Угол G–Prn–Pog' (общий контур профиля).",
        ["G", "Prn", "Pog"],
        [("G", "Prn"), ("Prn", "Pog")],
        8.0,
    )
    add(
        "facial_convexity_nasion",
        "Facial convexity (Nasion)",
        "профиль",
        "facial_convexity_nasion",
        "°",
        "Угол N'–Sn–Pog'.",
        ["N", "Sn", "Pog"],
        [("N", "Sn"), ("Sn", "Pog")],
        8.0,
    )
    add(
        "nose_tip_rotation",
        "Nose tip rotation",
        "нос",
        "nose_tip_rotation",
        "°",
        "Ротация кончика носа относительно спинки.",
        ["N", "Prn", "Sn"],
        [("N", "Prn"), ("Prn", "Sn")],
        8.0,
    )
    add(
        "midface_projection_angle",
        "Interior midface projection",
        "средняя зона",
        "midface_projection_angle",
        "°",
        "Угол G–Sn–Pog' — проекция midface на профиле.",
        ["G", "Sn", "Pog"],
        [("G", "Sn"), ("Sn", "Pog")],
        10.0,
    )
    add(
        "upper_lip_burstone",
        "Upper lip Burstone",
        "губы",
        "upper_lip_burstone",
        "%",
        "Положение верхней губы к линии Sn–Pog' (Burstone).",
        ["Sn", "Pog", "Ls"],
        [("Sn", "Pog")],
        2.5,
    )
    add(
        "lower_lip_burstone",
        "Lower lip Burstone",
        "губы",
        "lower_lip_burstone",
        "%",
        "Положение нижней губы к линии Sn–Pog' (Burstone).",
        ["Sn", "Pog", "Li"],
        [("Sn", "Pog")],
        2.5,
    )

    # Landmark overlay row so UI can draw all auto points.
    out.append(
        _pack(
            mid="profile_soft_tissue_points",
            label="Soft-tissue точки (авто)",
            category="профиль",
            value=float(len(lm)),
            unit="x",
            ideal_min=10.0,
            ideal_max=12.0,
            explanation="Авто: Tr, G, N', Prn, Sn, Ls, Li, Pog', Me', Go'.",
            points=[_pt(calc, k.lower(), p, "anchor") for k, p in lm.items()],
            segments=[],
            formula={"type": "static"},
            soft_margin=5.0,
            view="profile",
        )
    )
    return out


def _neutral_profile_pack(reason: str) -> dict[str, dict[str, Any]]:
    def pack(score: float, label: str) -> dict[str, Any]:
        return {"score": score, "label": label, "explanation": reason, "ratio": None}

    return {
        "nose": pack(62.0, "Нос (профиль)"),
        "chin": pack(62.0, "Подбородок (профиль)"),
        "jaw": pack(62.0, "Угол челюсти (профиль)"),
        "midface": pack(62.0, "Средняя зона (профиль)"),
        "lips": pack(62.0, "Губы (профиль)"),
        "face_shape": pack(62.0, "Контур профиля"),
    }

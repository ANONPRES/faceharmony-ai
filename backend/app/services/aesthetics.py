"""Aesthetic facial-feature metrics from MediaPipe landmarks.

Geometric traits often discussed in facial aesthetics: cheekbones, eye cut,
nose, face shape. Still photo ratios — not a moral verdict on a person.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .landmark_detector import LANDMARK
from .scoring import soft_score, combine_scores, distance as _distance


def eye_cut_score(calc: Any) -> dict[str, Any]:
    """
    Вырез глаз: canthal tilt + aperture + согласованность сторон.

    Важно: landmark 33/263 сидят чуть выше настоящего латерального кантуса
    и дают ложный «positive tilt». Берём стык верхнего/нижнего века и меряем
    относительно линии внутренних углов глаз (не относительно кадра).
    """
    def mix(i_a: int, i_b: int, w_a: float = 0.5) -> tuple[float, float]:
        ax, ay = calc.px_index(i_a)
        bx, by = calc.px_index(i_b)
        return ax * w_a + bx * (1.0 - w_a), ay * w_a + by * (1.0 - w_a)

    # Inner canthi: average of classic inner + lower-inner.
    # Outer canthi: bias toward lower-lid corner (7/249) — closer to true canthus.
    left_inner = mix(133, 173, 0.55)
    left_outer = mix(33, 7, 0.35)
    right_inner = mix(362, 398, 0.55)
    right_outer = mix(263, 249, 0.35)

    # Reference axis: both medial canthi (stable "eye horizontal").
    mlx, mly = left_inner
    mrx, mry = right_inner
    ref = np.array([mrx - mlx, mry - mly], dtype=float)
    ref_n = float(np.hypot(ref[0], ref[1])) or 1.0
    u = ref / ref_n
    # Image "down" side of the perpendicular.
    perp = np.array([-u[1], u[0]], dtype=float)
    if np.dot(perp, [0.0, 1.0]) < 0:
        perp = -perp
    up = -perp  # face-up in image space

    def tilt_of(inner: tuple[float, float], outer: tuple[float, float]) -> float:
        d = np.array([outer[0] - inner[0], outer[1] - inner[1]], dtype=float)
        height = float(np.dot(d, up))  # >0 if outer higher
        width = abs(float(np.dot(d, u)))
        return float(np.degrees(np.arctan2(height, max(width, 1e-3))))

    left_tilt = tilt_of(left_inner, left_outer)
    right_tilt = tilt_of(right_inner, right_outer)
    mean_tilt = float(np.nan_to_num((left_tilt + right_tilt) / 2.0, nan=0.0))
    tilt_sym = abs(left_tilt - right_tilt)

    # Neutral ≈ 0°; soft positive often ~2–5°. Don't treat 0 as a failure.
    tilt_score = soft_score(mean_tilt, 2.5, 5.5)
    tilt_sym_score = float(np.clip(100.0 * np.exp(-tilt_sym / 4.0), 0, 100))

    le_t, le_b = calc.px("left_eye_top"), calc.px("left_eye_bottom")
    re_t, re_b = calc.px("right_eye_top"), calc.px("right_eye_bottom")
    left_w = max(_distance(left_inner, left_outer), 1e-3)
    right_w = max(_distance(right_inner, right_outer), 1e-3)
    aperture = float(
        np.nan_to_num(
            (
                _distance(le_t, le_b) / left_w
                + _distance(re_t, re_b) / right_w
            )
            / 2.0,
            nan=0.36,
        )
    )
    ap_score = max(soft_score(aperture, 0.36, 0.12), 25.0)

    score = combine_scores(
        [
            (tilt_score, 0.45),
            (ap_score, 0.35),
            (tilt_sym_score, 0.20),
        ]
    )
    score = float(np.clip(max(score, 15.0), 0.0, 100.0))

    if abs(mean_tilt) <= 3.5:
        vibe = "нейтральный (почти без подъёма/опускания угла)"
    elif mean_tilt > 3.5:
        vibe = "слабо/умеренно приподнятый (positive canthal tilt)"
    else:
        vibe = "опущенный (negative canthal tilt)"

    return {
        "score": round(score, 1),
        "label": "Вырез глаз",
        "explanation": (
            f"Canthal tilt ≈ {mean_tilt:.1f}° "
            f"(L {left_tilt:.1f}° / R {right_tilt:.1f}°), "
            f"раскрытие ≈ {aperture:.2f}. Сейчас {vibe}."
        ),
        "ratio": round(mean_tilt, 4),
    }


def cheekbone_score(calc: Any) -> dict[str, Any]:
    """Скулы: ширина vs челюсть, высота, относительный выступ.

    Высокие скулы = landmark скул ближе к линии глаз → меньший cheek_pos.
    """
    cheek_w = abs(calc.face("right_cheek")[0] - calc.face("left_cheek")[0])
    jaw_w = abs(calc.face("jaw_right")[0] - calc.face("jaw_left")[0])
    temple_w = abs(calc.face("right_temple")[0] - calc.face("left_temple")[0])

    cheek_jaw = cheek_w / max(jaw_w, 1e-3)
    cheek_temple = cheek_w / max(temple_w, 1e-3)

    eye_v = (calc.face("left_eye_outer")[1] + calc.face("right_eye_outer")[1]) / 2.0
    nose_v = calc.face("nose_bottom")[1]
    cheek_v = (calc.face("left_cheek")[1] + calc.face("right_cheek")[1]) / 2.0
    span = max(nose_v - eye_v, 1e-3)
    cheek_pos = (cheek_v - eye_v) / span
    # High / sculpted cheekbones sit early in the eye→nose span (~0.18–0.28).
    # Softer / lower cheek mass (higher cheek_pos) falls off steeply.
    if cheek_pos <= 0.28:
        height_score = soft_score(cheek_pos, 0.22, 0.09)
    else:
        height_score = float(
            np.clip(100.0 * np.exp(-(((cheek_pos - 0.28) / 0.055) ** 2)), 0.0, 100.0)
        )

    width_score = combine_scores(
        [
            # Clear zygomatic width vs jaw (V / heart taper), not ultra-wide only.
            (soft_score(cheek_jaw, 1.20, 0.16), 0.70),
            (soft_score(cheek_temple, 1.0, 0.10), 0.30),
        ]
    )

    z_cheek = abs(
        (
            calc.landmarks[LANDMARK["left_cheek"]]["z"]
            + calc.landmarks[LANDMARK["right_cheek"]]["z"]
        )
        / 2.0
    )
    z_nose = abs(calc.landmarks[LANDMARK["nose_tip"]]["z"])
    # MediaPipe cheek |z| is typically larger than tip |z| on frontal shots (~1.6–2.2).
    prom = z_cheek / max(z_nose, 1e-4)
    prom_score = soft_score(prom, 1.85, 0.55)

    score = combine_scores(
        [
            (width_score, 0.40),
            (height_score, 0.45),
            (prom_score, 0.15),
        ]
    )

    if cheek_pos <= 0.28 and cheek_jaw >= 1.10:
        note = "высокие скулы с заметным расширением относительно челюсти"
    elif cheek_pos <= 0.28:
        note = "скулы посажены высоко (ближе к линии глаз)"
    elif cheek_jaw >= 1.08:
        note = "скулы шире челюсти — читается более «скульптурная» ширина"
    elif cheek_jaw <= 0.98:
        note = "челюсть близка/шире скул — скуловая линия мягче"
    else:
        note = "ширина скул и челюсти близки"

    return {
        "score": round(score, 1),
        "label": "Скулы",
        "explanation": (
            f"Отношение скулы/челюсть ≈ {cheek_jaw:.2f}; высота скул в средней зоне ≈ {cheek_pos:.2f}. "
            f"Сейчас {note}."
        ),
        "ratio": round(cheek_jaw, 4),
    }


def nose_aesthetics_score(calc: Any) -> dict[str, Any]:
    """Нос: крылья, длина, кончик, ровность спинки."""
    alar = _distance(calc.px("left_nostril"), calc.px("right_nostril"))
    mouth = _distance(calc.px("mouth_left"), calc.px("mouth_right"))
    intercanthal = _distance(calc.px("left_eye_inner"), calc.px("right_eye_inner"))
    face_w = abs(calc.face("right_cheek")[0] - calc.face("left_cheek")[0])
    length = abs(calc.face("nose_bottom")[1] - calc.face("nose_bridge")[1])

    alar_eye = alar / max(intercanthal, 1e-3)
    alar_mouth = alar / max(mouth, 1e-3)
    length_face = length / max(face_w, 1e-3)

    tip_u, _ = calc.face("nose_tip")
    tip_center = float(np.clip(100.0 * np.exp(-abs(tip_u) / max(face_w, 1e-3) * 12.0), 0, 100))

    bridge_u, _ = calc.face("nose_bridge")
    bottom_u, _ = calc.face("nose_bottom")
    bend = abs(tip_u - (bridge_u + bottom_u) / 2.0) / max(face_w, 1e-3)
    straight = float(np.clip(100.0 * np.exp(-bend * 18.0), 0, 100))

    score = combine_scores(
        [
            (soft_score(alar_eye, 1.0, 0.18), 0.28),
            (soft_score(alar_mouth, 0.70, 0.14), 0.22),
            (soft_score(length_face, 0.34, 0.09), 0.20),
            (tip_center, 0.15),
            (straight, 0.15),
        ]
    )

    return {
        "score": round(score, 1),
        "label": "Нос",
        "explanation": (
            f"Ширина крыльев к межглазью ≈ {alar_eye:.2f}, к рту ≈ {alar_mouth:.2f}. "
            "Учитываются длина, центрирование кончика и ровность спинки на фото."
        ),
        "ratio": round(alar_eye, 4),
    }


def face_shape_score(calc: Any) -> dict[str, Any]:
    """Форма лица + гармоничность пропорций этой формы."""
    _, hair_v = calc.axes.to_face(*calc.hairline_xy)
    _, chin_v = calc.face("chin")
    face_h = abs(chin_v - hair_v)
    cheek_w = abs(calc.face("right_cheek")[0] - calc.face("left_cheek")[0])
    jaw_w = abs(calc.face("jaw_right")[0] - calc.face("jaw_left")[0])
    temple_w = abs(calc.face("right_temple")[0] - calc.face("left_temple")[0])
    brow_u_span = abs(calc.face("right_eye_outer")[0] - calc.face("left_eye_outer")[0])

    wh = cheek_w / max(face_h, 1e-3)
    cheek_jaw = cheek_w / max(jaw_w, 1e-3)
    cheek_temple = cheek_w / max(temple_w, 1e-3)

    if wh < 0.68 and cheek_jaw > 1.02:
        # Elongated model faces are common; don't force them toward oval WH≈0.74.
        shape = "вытянутая (oblong)"
        ideal = {"wh": 0.64, "cheek_jaw": 1.22, "wh_s": 0.10, "cj_s": 0.12}
    elif wh > 0.86 and cheek_jaw < 1.05:
        shape = "круглая"
        ideal = {"wh": 0.78, "cheek_jaw": 1.10, "wh_s": 0.08, "cj_s": 0.10}
    elif cheek_jaw < 1.02 and wh >= 0.72:
        shape = "квадратная"
        ideal = {"wh": 0.76, "cheek_jaw": 1.08, "wh_s": 0.08, "cj_s": 0.10}
    elif cheek_temple > 1.06 and cheek_jaw > 1.10:
        shape = "сердцевидная"
        ideal = {"wh": 0.74, "cheek_jaw": 1.18, "wh_s": 0.08, "cj_s": 0.12}
    else:
        shape = "овальная"
        ideal = {"wh": 0.72, "cheek_jaw": 1.18, "wh_s": 0.09, "cj_s": 0.12}

    score = combine_scores(
        [
            (soft_score(wh, ideal["wh"], ideal["wh_s"]), 0.35),
            (soft_score(cheek_jaw, ideal["cheek_jaw"], ideal["cj_s"]), 0.40),
            (soft_score(cheek_temple, 1.02, 0.10), 0.15),
            (soft_score(brow_u_span / max(cheek_w, 1e-3), 0.72, 0.12), 0.10),
        ]
    )

    return {
        "score": round(score, 1),
        "label": "Форма лица",
        "explanation": (
            f"По ширинам скул/челюсти/лба и высоте лицо ближе к типу «{shape}». "
            f"Ширина/высота ≈ {wh:.2f}, скулы/челюсть ≈ {cheek_jaw:.2f}."
        ),
        "ratio": round(wh, 4),
        "face_shape": shape,
    }


def brow_score(calc: Any) -> dict[str, Any]:
    """Положение бровей относительно глаз."""
    left_brow = calc.px_index(70) if len(calc.landmarks) > 70 else calc.px("glabella")
    right_brow = calc.px_index(300) if len(calc.landmarks) > 300 else calc.px("glabella")
    le_t = calc.px("left_eye_top")
    re_t = calc.px("right_eye_top")
    face_h = abs(calc.face("chin")[1] - calc.axes.to_face(*calc.hairline_xy)[1])

    left_gap = abs(calc.axes.to_face(*left_brow)[1] - calc.axes.to_face(*le_t)[1]) / max(face_h, 1e-3)
    right_gap = abs(calc.axes.to_face(*right_brow)[1] - calc.axes.to_face(*re_t)[1]) / max(face_h, 1e-3)
    gap = (left_gap + right_gap) / 2.0
    score = soft_score(gap, 0.045, 0.025)
    return {
        "score": round(score, 1),
        "label": "Брови",
        "explanation": (
            f"Зазор бровь–глаз ≈ {gap:.3f} высоты лица. "
            "Слишком низкие/высокие брови меняют выражение и баланс верхней трети."
        ),
        "ratio": round(gap, 4),
    }


def midface_score(calc: Any) -> dict[str, Any]:
    """Средняя зона лица относительно нижней трети."""
    _, brow_v = calc.face("glabella")
    _, nose_v = calc.face("nose_bottom")
    _, chin_v = calc.face("chin")
    mid = abs(nose_v - brow_v)
    lower = abs(chin_v - nose_v)
    ratio = mid / max(lower, 1e-3)
    score = soft_score(ratio, 0.95, 0.22)
    return {
        "score": round(score, 1),
        "label": "Средняя зона",
        "explanation": (
            f"Midface / нижняя треть ≈ {ratio:.2f}. "
            "Слишком длинная средняя зона визуально «вытягивает» лицо."
        ),
        "ratio": round(ratio, 4),
    }

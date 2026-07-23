"""Atomic facial measurements with ideal ranges (competitor-style catalog).

Each entry exposes value, ideal band, 0–10 score, overlay points/segments,
and a formula hint so the frontend can recalculate after manual point edits.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .scoring import distance as _distance, midpoint as _midpoint, range_score


Point = tuple[float, float]


def _n(calc: Any, p: Point) -> dict[str, float]:
    """Pixel → normalized image coords."""
    return {
        "x": float(p[0] / max(calc.width, 1)),
        "y": float(p[1] / max(calc.height, 1)),
    }


def _seg(
    calc: Any,
    p1: Point,
    p2: Point,
    *,
    style: str = "primary",
    label: str = "",
) -> dict[str, Any]:
    a, b = _n(calc, p1), _n(calc, p2)
    return {
        "x1": a["x"],
        "y1": a["y"],
        "x2": b["x"],
        "y2": b["y"],
        "style": style,
        "label": label,
    }


def _pt(calc: Any, pid: str, p: Point, role: str = "anchor", landmark: int | None = None) -> dict[str, Any]:
    n = _n(calc, p)
    out: dict[str, Any] = {"id": pid, "x": n["x"], "y": n["y"], "role": role}
    if landmark is not None:
        out["landmark"] = landmark
    return out


def _fmt(value: float, unit: str) -> str:
    if unit == "%":
        return f"{value:.1f}%"
    if unit == "°":
        return f"{value:.1f}°"
    if unit == "x":
        return f"{value:.2f}x"
    if unit == "mm":
        return f"{value:.1f} mm"
    return f"{value:.2f}"


def _pack(
    *,
    mid: str,
    label: str,
    category: str,
    value: float,
    unit: str,
    ideal_min: float,
    ideal_max: float,
    explanation: str,
    points: list[dict[str, Any]],
    segments: list[dict[str, Any]],
    formula: dict[str, Any],
    soft_margin: float | None = None,
    scale_pad: float | None = None,
    view: str = "frontal",
) -> dict[str, Any]:
    score100 = range_score(value, ideal_min, ideal_max, soft_margin)
    span = max(ideal_max - ideal_min, 1e-3)
    pad = scale_pad if scale_pad is not None else max(span * 1.8, span + 0.05)
    scale_min = ideal_min - pad
    scale_max = ideal_max + pad
    score10 = round(score100 / 10.0, 1)
    # Effective sigma used for peaked scoring (for client recompute).
    half = max(0.5 * span, 1e-6)
    sigma = soft_margin if soft_margin is not None else half * 1.15
    return {
        "id": mid,
        "label": label,
        "category": category,
        "value": round(float(value), 4),
        "unit": unit,
        "display": _fmt(value, unit),
        "ideal_min": round(float(ideal_min), 4),
        "ideal_max": round(float(ideal_max), 4),
        "score": round(score100, 1),
        "score_10": score10,
        "explanation": explanation,
        "scale_min": round(float(scale_min), 4),
        "scale_max": round(float(scale_max), 4),
        "soft_margin": round(float(sigma), 4),
        "points": points,
        "segments": segments,
        "formula": formula,
        "view": view,
    }


def _face_height_v(calc: Any) -> tuple[float, float, float, Point, Point]:
    """Hairline/chin v-coords and points."""
    hair = calc.hairline_xy
    chin = calc.px("chin")
    _, hv = calc.axes.to_face(*hair)
    _, cv = calc.axes.to_face(*chin)
    h = max(abs(cv - hv), 1.0)
    return hv, cv, h, hair, chin


def _canthi(calc: Any) -> tuple[Point, Point, Point, Point]:
    def mix(i_a: int, i_b: int, w_a: float = 0.5) -> Point:
        ax, ay = calc.px_index(i_a)
        bx, by = calc.px_index(i_b)
        return ax * w_a + bx * (1.0 - w_a), ay * w_a + by * (1.0 - w_a)

    left_inner = mix(133, 173, 0.55)
    left_outer = mix(33, 7, 0.35)
    right_inner = mix(362, 398, 0.55)
    right_outer = mix(263, 249, 0.35)
    return left_inner, left_outer, right_inner, right_outer


def _tilt_deg(inner: Point, outer: Point, ref_u: np.ndarray, up: np.ndarray) -> float:
    """Positive = outer canthus higher than inner (hunter / positive canthal tilt).

    `up` must already point toward the top of the face in image space (as built
    in build_detailed_measurements from the eye-line perpendicular).
    """
    d = np.array([outer[0] - inner[0], outer[1] - inner[1]], dtype=float)
    height = float(np.dot(d, up))
    width = abs(float(np.dot(d, ref_u)))
    return float(np.degrees(np.arctan2(height, max(width, 1e-3))))


def _angle_at(a: Point, vertex: Point, c: Point) -> float:
    """Interior angle at vertex in degrees."""
    v1 = np.array([a[0] - vertex[0], a[1] - vertex[1]], dtype=float)
    v2 = np.array([c[0] - vertex[0], c[1] - vertex[1]], dtype=float)
    n1 = float(np.hypot(v1[0], v1[1])) or 1.0
    n2 = float(np.hypot(v2[0], v2[1])) or 1.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _line_intersect(p1: Point, p2: Point, q1: Point, q2: Point) -> Point | None:
    """Intersection of infinite lines p1→p2 and q1→q2, or None if parallel."""
    a = np.array(p1[:2], dtype=float)
    b = np.array(p2[:2], dtype=float)
    c = np.array(q1[:2], dtype=float)
    d = np.array(q2[:2], dtype=float)
    d1 = b - a
    d2 = d - c
    den = float(d1[0] * d2[1] - d1[1] * d2[0])
    if abs(den) < 1e-9:
        return None
    t = float(((c - a)[0] * d2[1] - (c - a)[1] * d2[0]) / den)
    x, y = a + t * d1
    return float(x), float(y)


def _fit_line_2d(pts: list[Point]) -> tuple[np.ndarray, np.ndarray]:
    """Centroid + unit direction of a 2D least-squares line (direction → +y)."""
    p = np.asarray(pts, dtype=float)
    c = p.mean(axis=0)
    _, _, vh = np.linalg.svd(p - c, full_matrices=False)
    d = vh[0].astype(float)
    if d[1] < 0:
        d = -d
    n = float(np.hypot(d[0], d[1])) or 1.0
    return c, d / n


def _line_ends_on_fit(pts: list[Point], c: np.ndarray, d: np.ndarray) -> tuple[Point, Point]:
    """Project contour points onto the fit; return outer (gonion) and inner (near-chin) ends."""
    ts = [float(np.dot(np.asarray(pt, dtype=float) - c, d)) for pt in pts]
    t0, t1 = min(ts), max(ts)
    p0 = c + t0 * d
    p1 = c + t1 * d
    # Outer gonion is higher (smaller y) than near-chin end.
    a = (float(p0[0]), float(p0[1]))
    b = (float(p1[0]), float(p1[1]))
    if a[1] > b[1]:
        a, b = b, a
    return a, b


def _dir_intersect(c1: np.ndarray, d1: np.ndarray, c2: np.ndarray, d2: np.ndarray) -> Point | None:
    """Intersection of lines (c1,d1) and (c2,d2)."""
    a = np.column_stack((d1, -d2))
    if abs(float(np.linalg.det(a))) < 1e-9:
        return None
    t, _ = np.linalg.solve(a, c2 - c1)
    p = c1 + float(t) * d1
    return float(p[0]), float(p[1])


def _jaw_frontal_angle(calc: Any) -> tuple[float, Point, Point, Point, Point, Point]:
    """
    Jaw Frontal Angle = angle between the two mandibular *border* lines.

    Fit a least-squares line to each side of the jaw contour (gonion → lateral
    chin, NOT through soft-tissue menton 152). Extend both lines; they meet at
    an apex usually *below* the chin — FaceIQ / competitor geometry
    (Sean/Elias ≈ 79°, not menton vertex ~105°+).
    """
    chin = calc.px("chin")

    def _from_ids(left_ids: tuple[int, ...], right_ids: tuple[int, ...]):
        left_pts = [calc.px_index(i) for i in left_ids]
        right_pts = [calc.px_index(i) for i in right_ids]
        c1, d1 = _fit_line_2d(left_pts)
        c2, d2 = _fit_line_2d(right_pts)
        left_g, left_m = _line_ends_on_fit(left_pts, c1, d1)
        right_g, right_m = _line_ends_on_fit(right_pts, c2, d2)
        apex = _dir_intersect(c1, d1, c2, d2)
        if apex is None or apex[1] <= chin[1]:
            apex = _line_intersect(left_g, left_m, right_g, right_m)
        if apex is None:
            return None
        return _angle_at(left_g, apex, right_g), left_g, left_m, right_g, right_m, apex

    # Primary: gonion → antegonial (FaceIQ ~79° on Sean/Elias-class jaws).
    primary = _from_ids((172, 136, 150, 149), (397, 365, 379, 378))
    # Fallback: drop near-chin 149/378 when primary collapses too acute (<76°) —
    # those points often sit too medial and steepen the V (Elias-style under-read).
    opened = _from_ids((172, 136, 150, 176), (397, 365, 379, 400))
    if primary is None and opened is None:
        left_g, right_g = calc.px_index(172), calc.px_index(397)
        left_m, right_m = calc.px_index(176), calc.px_index(400)
        return _angle_at(left_g, chin, right_g), left_g, left_m, right_g, right_m, chin
    if primary is None:
        return opened  # type: ignore[return-value]
    if opened is None:
        return primary
    jfa_p = primary[0]
    jfa_o = opened[0]
    if jfa_p < 78.0 and jfa_o > jfa_p:
        # Blend toward the opener so collapsed ~73° photos land nearer FaceIQ ~79°.
        t = min(1.0, (78.0 - jfa_p) / 6.0)
        jfa = (1.0 - t) * jfa_p + t * jfa_o
        return (jfa, *opened[1:]) if t >= 0.45 else (jfa, *primary[1:])
    return primary


def build_detailed_measurements(calc: Any) -> list[dict[str, Any]]:
    """Return 45+ atomic measurements with ideals and overlay geometry."""
    out: list[dict[str, Any]] = []
    hv, cv, face_h, hair, chin = _face_height_v(calc)
    brow = calc.px("glabella")
    nose_bottom = calc.px("nose_bottom")
    nose_tip = calc.px("nose_tip")
    nose_bridge = calc.px("nose_bridge")
    _, brow_v = calc.axes.to_face(*brow)
    _, nose_v = calc.axes.to_face(*nose_bottom)
    mid_u = 0.0

    left_cheek = calc.px("left_cheek")
    right_cheek = calc.px("right_cheek")
    left_jaw = calc.px("jaw_left")
    right_jaw = calc.px("jaw_right")
    # Soft-tissue gonion (FaceIQ JFA / bigonial): jaw contour near mouth-corner height.
    soft_gonion_l = calc.px_index(132)
    soft_gonion_r = calc.px_index(361)
    left_temple = calc.px_index(54)
    right_temple = calc.px_index(284)
    # Lateral face edge at eye level (for facial fifths) — not forehead temples.
    face_side_l = calc.px_index(127)
    face_side_r = calc.px_index(356)
    mouth_l = calc.px("mouth_left")
    mouth_r = calc.px("mouth_right")
    upper_lip = calc.px("upper_lip")
    lower_lip = calc.px("lower_lip")
    philtrum = calc.px("philtrum")
    left_nostril = calc.px("left_nostril")
    right_nostril = calc.px("right_nostril")

    cheek_w = _distance(left_cheek, right_cheek)
    jaw_w = _distance(soft_gonion_l, soft_gonion_r)
    temple_w = _distance(left_temple, right_temple)
    mouth_w = _distance(mouth_l, mouth_r)
    alar_w = _distance(left_nostril, right_nostril)
    bridge_l, bridge_r = calc.px_index(193), calc.px_index(417)
    bridge_w = _distance(bridge_l, bridge_r)

    left_inner, left_outer, right_inner, right_outer = _canthi(calc)
    eye_line = np.array(
        [right_inner[0] - left_inner[0], right_inner[1] - left_inner[1]], dtype=float
    )
    ref_n = float(np.hypot(eye_line[0], eye_line[1])) or 1.0
    ref_u = eye_line / ref_n
    perp = np.array([-ref_u[1], ref_u[0]], dtype=float)
    if np.dot(perp, [0.0, 1.0]) < 0:
        perp = -perp
    up = -perp

    # --- Proportions / thirds ---
    upper_len = abs(brow_v - hv)
    mid_len = abs(nose_v - brow_v)
    lower_len = abs(cv - nose_v)
    upper_pct = 100.0 * upper_len / face_h
    mid_pct = 100.0 * mid_len / face_h
    lower_pct = 100.0 * lower_len / face_h
    mid_to_lower = mid_len / max(lower_len, 1e-3)

    hair_pt = hair
    brow_pt = brow
    nose_pt = nose_bottom
    chin_pt = chin
    midline = [
        _pt(calc, "hairline", hair_pt, "anchor"),
        _pt(calc, "brow", brow_pt, "anchor", 9),
        _pt(calc, "nose_base", nose_pt, "anchor", 2),
        _pt(calc, "chin", chin_pt, "anchor", 152),
    ]

    # ── IAA / JFA (FaceIQ Labs definitions) ───────────────────────────
    # IAA: lateral canthi → nose tip. Ideal ≈ 87–91°.
    # JFA: angle between mandibular border lines (172→176 / 397→400),
    #      apex usually BELOW menton — NOT angle at landmark 152 (~106°).
    # |IAA−JFA| ideal ≈ 0–2°.
    iaa_lo = calc.px("left_eye_outer")
    iaa_ro = calc.px("right_eye_outer")
    iaa = _angle_at(iaa_lo, nose_tip, iaa_ro)
    jfa, jfa_lg, jfa_lm, jfa_rg, jfa_rm, jfa_apex = _jaw_frontal_angle(calc)
    iaa_jfa = abs(iaa - jfa)

    out.append(
        _pack(
            mid="iaa_jfa_diff",
            label="Разница IAA и JFA",
            category="структура",
            value=iaa_jfa,
            unit="°",
            ideal_min=0.0,
            ideal_max=2.5,
            explanation=(
                f"IAA ≈ {iaa:.1f}°, JFA ≈ {jfa:.1f}°. "
                f"FaceIQ идеал: разница 0–2.5° (у Elias ~10.6° → ~3.9/10, не ноль)."
            ),
            points=[
                _pt(calc, "lo", iaa_lo, "anchor", 33),
                _pt(calc, "ro", iaa_ro, "anchor", 263),
                _pt(calc, "tip", nose_tip, "anchor", 1),
                _pt(calc, "jl", jfa_lg, "anchor", 172),
                _pt(calc, "jr", jfa_rg, "anchor", 397),
                _pt(calc, "apex", jfa_apex, "anchor"),
            ],
            segments=[
                _seg(calc, iaa_lo, nose_tip, style="ref"),
                _seg(calc, iaa_ro, nose_tip, style="ref"),
                _seg(calc, jfa_lg, jfa_apex, style="primary"),
                _seg(calc, jfa_rg, jfa_apex, style="primary", label=f"{iaa_jfa:.1f}°"),
            ],
            formula={
                "type": "angle_diff",
                "a": ["lo", "tip", "ro"],
                "b": ["jl", "apex", "jr"],
            },
            # FaceIQ Elias 10.6° → 3.9/10; Sean 10° → 4.2/10.
            soft_margin=7.0,
            scale_pad=10.0,
        )
    )

    out.append(
        _pack(
            mid="iaa",
            label="IAA (ipsilateral alar)",
            category="структура",
            value=iaa,
            unit="°",
            ideal_min=86.5,
            ideal_max=92.5,
            explanation=(
                "Угол у кончика носа между внешними углами глаз (FaceIQ Ipsilateral Alar). "
                "Идеал 86.5–92.5° (как JFA)."
            ),
            points=[
                _pt(calc, "lo", iaa_lo, "anchor", 33),
                _pt(calc, "ro", iaa_ro, "anchor", 263),
                _pt(calc, "tip", nose_tip, "anchor", 1),
            ],
            segments=[
                _seg(calc, iaa_lo, nose_tip, style="primary"),
                _seg(calc, iaa_ro, nose_tip, style="primary", label=f"{iaa:.1f}°"),
            ],
            formula={"type": "angle", "points": ["lo", "tip", "ro"]},
            soft_margin=5.0,
            scale_pad=8.0,
        )
    )

    out.append(
        _pack(
            mid="jfa",
            label="JFA (jaw frontal angle)",
            category="структура",
            value=jfa,
            unit="°",
            ideal_min=86.5,
            ideal_max=92.5,
            explanation=(
                "Угол между линиями по контуру нижней челюсти (вершина обычно ниже menton). "
                "Идеал FaceIQ (мужчины): 86.5–92.5°, пик ~89.5°. "
                "Looksmax Tier1 шире: 84–95°. "
                "Квадратная «Low Jaw» ~78–80° (у FaceIQ у Sean O'Pry ~79° → ~6.7/10, не ноль)."
            ),
            points=[
                _pt(calc, "jl", jfa_lg, "anchor", 172),
                _pt(calc, "jm", jfa_lm, "anchor"),
                _pt(calc, "jr", jfa_rg, "anchor", 397),
                _pt(calc, "jrm", jfa_rm, "anchor"),
                _pt(calc, "apex", jfa_apex, "anchor"),
            ],
            segments=[
                _seg(calc, jfa_lg, jfa_apex, style="primary"),
                _seg(calc, jfa_rg, jfa_apex, style="primary", label=f"{jfa:.1f}°"),
            ],
            formula={"type": "jfa_intersect", "left": ["jl", "jm"], "right": ["jr", "jrm"]},
            # Calibrated to FaceIQ: ~79° ≈ 6.7/10 (Sean O'Pry). Scale UI ~80.5–98.5°.
            soft_margin=12.0,
            scale_pad=6.0,
        )
    )

    # Cheekbone height (FaceIQ): (menton→zygion) / (menton→pupil line) × 100.
    # Ideal 83–100%. NOT / full face height (that yields ~48% and is wrong).
    pupil_l = _midpoint(calc.px("left_eye_inner"), calc.px("left_eye_outer"))
    pupil_r = _midpoint(calc.px("right_eye_inner"), calc.px("right_eye_outer"))
    pupil_mid = _midpoint(pupil_l, pupil_r)
    cheek_mid = _midpoint(left_cheek, right_cheek)
    _, cheek_v = calc.axes.to_face(*cheek_mid)
    _, pupil_v = calc.axes.to_face(*pupil_mid)
    chin_to_cheek = abs(cv - cheek_v)
    chin_to_pupil = max(abs(cv - pupil_v), 1e-3)
    cheek_height_pct = 100.0 * chin_to_cheek / chin_to_pupil
    out.append(
        _pack(
            mid="cheekbone_height",
            label="Высота скул",
            category="скулы",
            value=cheek_height_pct,
            unit="%",
            ideal_min=83.0,
            ideal_max=100.0,
            explanation=(
                "Высота скул: расстояние подбородок→скулы / подбородок→линия зрачков. "
                "Идеал FaceIQ: 83–100% (высокие скулы)."
            ),
            points=[
                _pt(calc, "lc", left_cheek, "anchor", 234),
                _pt(calc, "rc", right_cheek, "anchor", 454),
                _pt(calc, "chin", chin, "ref", 152),
                _pt(calc, "pupil", pupil_mid, "ref"),
            ],
            segments=[
                _seg(calc, left_cheek, right_cheek, style="ref"),
                _seg(
                    calc,
                    cheek_mid,
                    chin,
                    style="primary",
                    label=f"{cheek_height_pct:.1f}%",
                ),
            ],
            formula={
                "type": "pct_ratio_pts",
                "num": ["chin", "lc"],
                "den": ["chin", "pupil"],
            },
            soft_margin=12.0,
        )
    )

    out.append(
        _pack(
            mid="upper_third",
            label="Верхняя треть",
            category="трети",
            value=upper_pct,
            unit="%",
            ideal_min=30.0,
            ideal_max=36.0,
            explanation="Доля лица от линии роста волос до бровей (glabella). Идеал ≈ 30–36% (FaceIQ ~33%).",
            points=midline,
            segments=[
                _seg(calc, hair_pt, chin_pt, style="ref"),
                _seg(calc, hair_pt, brow_pt, style="primary", label=f"{upper_pct:.1f}%"),
            ],
            formula={"type": "thirds_segment", "segment": "upper"},
            soft_margin=5.0,
        )
    )

    out.append(
        _pack(
            mid="mid_third",
            label="Средняя треть",
            category="трети",
            value=mid_pct,
            unit="%",
            ideal_min=30.0,
            ideal_max=36.0,
            explanation="Доля лица от бровей до основания носа. Идеал ≈ 30–36%.",
            points=midline,
            segments=[
                _seg(calc, hair_pt, chin_pt, style="ref"),
                _seg(calc, brow_pt, nose_pt, style="primary", label=f"{mid_pct:.1f}%"),
            ],
            formula={"type": "thirds_segment", "segment": "mid"},
            soft_margin=5.0,
        )
    )

    out.append(
        _pack(
            mid="lower_third",
            label="Нижняя треть",
            category="трети",
            value=lower_pct,
            unit="%",
            ideal_min=30.0,
            ideal_max=36.0,
            explanation="Доля лица от основания носа до подбородка. Идеал ≈ 30–36%.",
            points=midline,
            segments=[
                _seg(calc, hair_pt, chin_pt, style="ref"),
                _seg(calc, nose_pt, chin_pt, style="primary", label=f"{lower_pct:.1f}%"),
            ],
            formula={"type": "thirds_segment", "segment": "lower"},
            soft_margin=5.0,
        )
    )

    out.append(
        _pack(
            mid="mid_third_ratio",
            label="Соотношение средней трети",
            category="трети",
            value=mid_to_lower,
            unit="x",
            ideal_min=0.97,
            ideal_max=1.00,
            explanation="Средняя треть к нижней: баланс midface / lower face.",
            points=[
                _pt(calc, "li", left_inner, "anchor"),
                _pt(calc, "ri", right_inner, "anchor"),
                _pt(calc, "nb", nose_bottom, "anchor", 2),
            ],
            segments=[
                _seg(calc, left_inner, right_inner, style="primary"),
                _seg(
                    calc,
                    _midpoint(left_inner, right_inner),
                    nose_bottom,
                    style="primary",
                    label=f"{mid_to_lower:.2f}x",
                ),
            ],
            formula={"type": "ratio", "num": ["li", "ri"], "den_mode": "vertical_mid_to", "den": "nb"},
            soft_margin=0.08,
        )
    )

    # FWHR-style: cheek width / midface height (brow → nose). Ideal ≈ 1.96–2.00.
    # Using brow→chin gives ~1.0 and does not match FaceIQ / competitor ideals.
    midface_h = max(abs(nose_v - brow_v), 1.0)
    cheek_to_midface = cheek_w / midface_h
    out.append(
        _pack(
            mid="face_wh_cheek",
            label="Ширина/высота лица (скулы)",
            category="пропорции",
            value=cheek_to_midface,
            unit="x",
            ideal_min=1.90,
            ideal_max=2.10,
            explanation=(
                "Ширина по скулам к высоте средней зоны (брови → основание носа). "
                "Идеал ≈ 1.9–2.1 (FaceIQ FWHR)."
            ),
            points=[
                _pt(calc, "lc", left_cheek, "anchor", 234),
                _pt(calc, "rc", right_cheek, "anchor", 454),
                _pt(calc, "brow", brow, "anchor", 9),
                _pt(calc, "nb", nose_bottom, "anchor", 2),
            ],
            segments=[
                _seg(calc, left_cheek, right_cheek, style="primary"),
                _seg(calc, brow, nose_bottom, style="primary", label=f"{cheek_to_midface:.2f}x"),
            ],
            formula={"type": "ratio_hw", "h1": "lc", "h2": "rc", "v1": "brow", "v2": "nb"},
            soft_margin=0.28,
            scale_pad=0.35,
        )
    )

    forehead_w_pct = 100.0 * temple_w / max(cheek_w, 1e-3)
    out.append(
        _pack(
            mid="forehead_width",
            label="Ширина лба (височная)",
            category="пропорции",
            value=forehead_w_pct,
            unit="%",
            ideal_min=86.5,
            ideal_max=92.5,
            explanation="Ширина висков (bitemporal) относительно ширины скул. Идеал FaceIQ: 86.5–92.5%.",
            points=[
                _pt(calc, "lt", left_temple, "anchor", 54),
                _pt(calc, "rt", right_temple, "anchor", 284),
                _pt(calc, "lc", left_cheek, "ref", 234),
                _pt(calc, "rc", right_cheek, "ref", 454),
            ],
            segments=[
                _seg(calc, left_cheek, right_cheek, style="ref"),
                _seg(calc, left_temple, right_temple, style="primary", label=f"{forehead_w_pct:.1f}%"),
            ],
            formula={"type": "pct_ratio", "num": ["lt", "rt"], "den": ["lc", "rc"]},
            soft_margin=8.0,
        )
    )

    # Neck: MediaPipe has no true neck. Estimate below soft gonion, slightly more medial.
    # If the estimate is absurd (<75% or >110%), skip (handled by view filter + validity).
    gul, gvl = calc.axes.to_face(*soft_gonion_l)
    gur, gvr = calc.axes.to_face(*soft_gonion_r)
    neck_v = max(gvl, gvr) + 0.07 * face_h
    neck_l = calc.axes.to_image(gul * 0.96, neck_v)
    neck_r = calc.axes.to_image(gur * 0.96, neck_v)
    # Clamp to image bounds
    neck_l = (float(np.clip(neck_l[0], 0, calc.width - 1)), float(np.clip(neck_l[1], 0, calc.height - 1)))
    neck_r = (float(np.clip(neck_r[0], 0, calc.width - 1)), float(np.clip(neck_r[1], 0, calc.height - 1)))
    neck_w = _distance(neck_l, neck_r)
    neck_pct = 100.0 * neck_w / max(jaw_w, 1e-3)
    if 75.0 <= neck_pct <= 110.0:
        out.append(
            _pack(
                mid="neck_width",
                label="Ширина шеи",
                category="пропорции",
                value=neck_pct,
                unit="%",
                ideal_min=92.0,
                ideal_max=98.0,
                explanation="Ширина шеи к bigonial (оценка чуть ниже углов челюсти). Идеал FaceIQ ≈ 92–98%.",
                points=[
                    _pt(calc, "nl", neck_l, "anchor"),
                    _pt(calc, "nr", neck_r, "anchor"),
                    _pt(calc, "jl", soft_gonion_l, "ref", 132),
                    _pt(calc, "jr", soft_gonion_r, "ref", 361),
                ],
                segments=[
                    _seg(calc, soft_gonion_l, soft_gonion_r, style="ref"),
                    _seg(calc, neck_l, neck_r, style="primary", label=f"{neck_pct:.1f}%"),
                ],
                formula={"type": "pct_ratio", "num": ["nl", "nr"], "den": ["jl", "jr"]},
                soft_margin=10.0,
            )
        )

    jaw_pct = 100.0 * jaw_w / max(cheek_w, 1e-3)
    out.append(
        _pack(
            mid="jaw_width_bigonial",
            label="Ширина челюсти (bigonial)",
            category="челюсть",
            value=jaw_pct,
            unit="%",
            ideal_min=87.5,
            ideal_max=91.5,
            explanation="Bigonial (soft gonion) к bizygomatic — мягкий V-taper vs квадрат. Идеал FaceIQ ≈ 87.5–91.5%.",
            points=[
                _pt(calc, "jl", soft_gonion_l, "anchor", 132),
                _pt(calc, "jr", soft_gonion_r, "anchor", 361),
                _pt(calc, "lc", left_cheek, "ref", 234),
                _pt(calc, "rc", right_cheek, "ref", 454),
            ],
            segments=[
                _seg(calc, left_cheek, right_cheek, style="ref"),
                _seg(calc, soft_gonion_l, soft_gonion_r, style="primary", label=f"{jaw_pct:.1f}%"),
            ],
            formula={"type": "pct_ratio", "num": ["jl", "jr"], "den": ["lc", "rc"]},
            soft_margin=8.0,
            scale_pad=10.0,
        )
    )

    # Eyes — canthal tilt from named landmarks (mix understates tilt ~2° vs FaceIQ ~7°).
    eye_li = calc.px("left_eye_inner")
    eye_lo = calc.px("left_eye_outer")
    eye_ri = calc.px("right_eye_inner")
    eye_ro = calc.px("right_eye_outer")
    mean_tilt = (
        _tilt_deg(eye_li, eye_lo, ref_u, up)
        + _tilt_deg(eye_ri, eye_ro, ref_u, up)
    ) / 2.0
    out.append(
        _pack(
            mid="canthal_tilt",
            label="Кантальный наклон",
            category="глаза",
            value=mean_tilt,
            unit="°",
            ideal_min=5.0,
            ideal_max=8.5,
            explanation=(
                "Positive canthal tilt: внешний угол выше внутреннего. "
                "Идеал FaceIQ ≈ 5–8.5°."
            ),
            points=[
                _pt(calc, "li", eye_li, "anchor", 133),
                _pt(calc, "lo", eye_lo, "anchor", 33),
                _pt(calc, "ri", eye_ri, "anchor", 362),
                _pt(calc, "ro", eye_ro, "anchor", 263),
            ],
            segments=[
                _seg(calc, eye_li, eye_lo, style="primary", label=f"{mean_tilt:.1f}°"),
                _seg(calc, eye_ri, eye_ro, style="primary"),
            ],
            formula={"type": "canthal_tilt"},
            soft_margin=4.0,
            scale_pad=6.0,
        )
    )

    # Eye widths for aspect / one-eye-apart (same named canthi).
    le_w = _distance(eye_li, eye_lo)
    re_w = _distance(eye_ri, eye_ro)
    le_h = _distance(calc.px("left_eye_top"), calc.px("left_eye_bottom"))
    re_h = _distance(calc.px("right_eye_top"), calc.px("right_eye_bottom"))
    eye_aspect = ((le_w / max(le_h, 1e-3)) + (re_w / max(re_h, 1e-3))) / 2.0
    out.append(
        _pack(
            mid="eye_aspect",
            label="Пропорции глаз (ширина/высота)",
            category="глаза",
            value=eye_aspect,
            unit="x",
            ideal_min=3.00,
            ideal_max=3.50,
            explanation="Отношение ширины глазной щели к высоте (aperture). FaceIQ ideal ~2.8–3.6.",
            points=[
                _pt(calc, "li", eye_li, "anchor", 133),
                _pt(calc, "lo", eye_lo, "anchor", 33),
                _pt(calc, "lt", calc.px("left_eye_top"), "anchor", 159),
                _pt(calc, "lb", calc.px("left_eye_bottom"), "anchor", 145),
                _pt(calc, "ri", eye_ri, "anchor", 362),
                _pt(calc, "ro", eye_ro, "anchor", 263),
                _pt(calc, "rt", calc.px("right_eye_top"), "anchor", 386),
                _pt(calc, "rb", calc.px("right_eye_bottom"), "anchor", 374),
            ],
            segments=[
                _seg(calc, eye_li, eye_lo, style="primary"),
                _seg(calc, calc.px("left_eye_top"), calc.px("left_eye_bottom"), style="primary"),
                _seg(calc, eye_ri, eye_ro, style="primary", label=f"{eye_aspect:.2f}x"),
                _seg(calc, calc.px("right_eye_top"), calc.px("right_eye_bottom"), style="primary"),
            ],
            formula={"type": "eye_aspect"},
            soft_margin=0.45,
        )
    )

    intercanthal = _distance(eye_li, eye_ri)
    mean_eye_w = (le_w + re_w) / 2.0
    # FaceIQ «One Eye Apart»: intercanthal ÷ one eye width. Ideal ≈ 0.9–1.0.
    eye_spacing = intercanthal / max(mean_eye_w, 1e-3)
    out.append(
        _pack(
            mid="eye_spacing",
            label="Межглазье / ширина глаза",
            category="глаза",
            value=eye_spacing,
            unit="x",
            ideal_min=0.90,
            ideal_max=1.05,
            explanation=(
                "One Eye Apart (FaceIQ): межкантальное ≈ одной ширине глаза. "
                "Идеал 0.90–1.05 (пик ~0.97)."
            ),
            points=[
                _pt(calc, "li", eye_li, "anchor", 133),
                _pt(calc, "ri", eye_ri, "anchor", 362),
                _pt(calc, "lo", eye_lo, "ref", 33),
                _pt(calc, "ro", eye_ro, "ref", 263),
            ],
            segments=[
                _seg(calc, eye_li, eye_ri, style="primary", label=f"{eye_spacing:.2f}x"),
                _seg(calc, eye_li, eye_lo, style="ref"),
            ],
            formula={"type": "ratio_hw", "h1": "li", "h2": "ri", "v1": "li", "v2": "lo", "as_ratio": True},
            soft_margin=0.18,
            scale_pad=0.25,
        )
    )

    brow_l = calc.px_index(70)
    brow_r = calc.px_index(300)
    eye_top_mid = _midpoint(calc.px("left_eye_top"), calc.px("right_eye_top"))
    brow_mid = _midpoint(brow_l, brow_r)
    brow_gap = _distance(brow_mid, eye_top_mid) / face_h
    out.append(
        _pack(
            mid="brow_eye_gap",
            label="Расстояние бровь–глаз",
            category="брови",
            value=100.0 * brow_gap,
            unit="%",
            ideal_min=3.5,
            ideal_max=5.5,
            explanation="Зазор между бровью и верхним веком относительно высоты лица.",
            points=[
                _pt(calc, "bl", brow_l, "anchor", 70),
                _pt(calc, "br", brow_r, "anchor", 300),
                _pt(calc, "et", eye_top_mid, "anchor"),
            ],
            segments=[_seg(calc, brow_mid, eye_top_mid, style="primary", label=f"{100*brow_gap:.1f}%")],
            formula={"type": "vertical_pct_face", "a": "bl", "b": "et"},
            soft_margin=2.5,
        )
    )

    # Nose
    # FaceIQ: Nose Width ÷ Bridge Width (label «Nose Bridge to Nose Width»). Soft ≈ 1.98–2.22.
    nose_bridge_ratio = alar_w / max(bridge_w, 1e-3)
    out.append(
        _pack(
            mid="nose_width_bridge",
            label="Ширина носа / переносица",
            category="нос",
            value=nose_bridge_ratio,
            unit="x",
            ideal_min=2.00,
            ideal_max=2.20,
            explanation=(
                "Крылья носа к переносице (FaceIQ Nose Bridge→Nose Width). "
                "Идеал ~2.0–2.2; у Sean O'Pry ~2.24 → ~9/10."
            ),
            points=[
                _pt(calc, "bl", bridge_l, "anchor", 193),
                _pt(calc, "br", bridge_r, "anchor", 417),
                _pt(calc, "nl", left_nostril, "anchor", 98),
                _pt(calc, "nr", right_nostril, "anchor", 327),
            ],
            segments=[
                _seg(calc, bridge_l, bridge_r, style="ref"),
                _seg(calc, left_nostril, right_nostril, style="primary", label=f"{nose_bridge_ratio:.2f}x"),
            ],
            formula={"type": "ratio_hw", "h1": "nl", "h2": "nr", "v1": "bl", "v2": "br", "as_ratio": True},
            soft_margin=0.45,
            scale_pad=0.45,
        )
    )

    nose_len_pct = 100.0 * abs(nose_v - brow_v) / face_h
    out.append(
        _pack(
            mid="nose_length",
            label="Длина носа",
            category="нос",
            value=nose_len_pct,
            unit="%",
            ideal_min=28.0,
            ideal_max=36.0,
            explanation="Длина носа (брови → основание) как доля высоты лица.",
            points=[
                _pt(calc, "brow", brow, "anchor", 9),
                _pt(calc, "nb", nose_bottom, "anchor", 2),
            ],
            segments=[_seg(calc, brow, nose_bottom, style="primary", label=f"{nose_len_pct:.1f}%")],
            formula={"type": "vertical_pct_face", "a": "brow", "b": "nb"},
            soft_margin=6.0,
        )
    )

    alar_eye = alar_w / max(mean_eye_w, 1e-3)
    out.append(
        _pack(
            mid="nose_eye_width",
            label="Ширина носа / глаз",
            category="нос",
            value=alar_eye,
            unit="x",
            ideal_min=0.90,
            ideal_max=1.15,
            explanation=(
                "Неоклассика: крылья носа ≈ ширине одного глаза. "
                "Идеал 0.90–1.15."
            ),
            points=[
                _pt(calc, "nl", left_nostril, "anchor", 98),
                _pt(calc, "nr", right_nostril, "anchor", 327),
                _pt(calc, "li", eye_li, "ref", 133),
                _pt(calc, "lo", eye_lo, "ref", 33),
            ],
            segments=[
                _seg(calc, left_nostril, right_nostril, style="primary", label=f"{alar_eye:.2f}x"),
                _seg(calc, eye_li, eye_lo, style="ref"),
            ],
            formula={"type": "ratio_hw", "h1": "nl", "h2": "nr", "v1": "li", "v2": "lo", "as_ratio": True},
            soft_margin=0.22,
            scale_pad=0.30,
        )
    )

    mouth_nose = mouth_w / max(alar_w, 1e-3)
    out.append(
        _pack(
            mid="mouth_nose_width",
            label="Ширина рта к ширине носа",
            category="губы",
            value=mouth_nose,
            unit="x",
            ideal_min=1.38,
            ideal_max=1.55,
            explanation="FaceIQ mouth÷nose: идеал ~1.38–1.53 (пик ~1.45). Sean ~1.5 → 10/10.",
            points=[
                _pt(calc, "ml", mouth_l, "anchor", 61),
                _pt(calc, "mr", mouth_r, "anchor", 291),
                _pt(calc, "nl", left_nostril, "ref", 98),
                _pt(calc, "nr", right_nostril, "ref", 327),
            ],
            segments=[
                _seg(calc, left_nostril, right_nostril, style="ref"),
                _seg(calc, mouth_l, mouth_r, style="primary", label=f"{mouth_nose:.2f}x"),
            ],
            formula={"type": "ratio_hw", "h1": "ml", "h2": "mr", "v1": "nl", "v2": "nr", "as_ratio": True},
            soft_margin=0.22,
            scale_pad=0.30,
        )
    )

    # Lips: lower vermillion / upper (use 13 to 0 vs 14 to 17)
    lip_top = calc.px("upper_lip")
    lip_bot = calc.px("lower_lip")
    lip_stomion = philtrum
    lip_lower_edge = calc.px_index(17)
    upper_lip_h = _distance(lip_top, lip_stomion)
    lower_lip_h = _distance(lip_bot, lip_lower_edge)
    # competitor lower/upper ~1.55–1.85; use full lip heights 0–13 and 14–17
    upper_h = _distance(calc.px_index(0), calc.px_index(13))
    lower_h = _distance(calc.px_index(14), calc.px_index(17))
    lip_ratio = lower_h / max(upper_h, 1e-3)
    out.append(
        _pack(
            mid="lip_ratio",
            label="Нижняя / верхняя губа",
            category="губы",
            value=lip_ratio,
            unit="x",
            ideal_min=1.55,
            ideal_max=1.85,
            explanation="Классическая пропорция объёма нижней губы к верхней.",
            points=[
                _pt(calc, "u1", calc.px_index(0), "anchor", 0),
                _pt(calc, "u2", calc.px_index(13), "anchor", 13),
                _pt(calc, "l1", calc.px_index(14), "anchor", 14),
                _pt(calc, "l2", calc.px_index(17), "anchor", 17),
            ],
            segments=[
                _seg(calc, calc.px_index(0), calc.px_index(13), style="ref"),
                _seg(calc, calc.px_index(14), calc.px_index(17), style="primary", label=f"{lip_ratio:.2f}x"),
            ],
            formula={"type": "ratio_hw", "h1": "l1", "h2": "l2", "v1": "u1", "v2": "u2", "as_ratio": True},
            soft_margin=0.35,
        )
    )

    # Mouth corner height vs mouth center (commissure lift) — mm-like via % of face * scale
    mouth_mid = _midpoint(mouth_l, mouth_r)
    # vertical offset of corners vs center in face frame
    _, ml_v = calc.axes.to_face(*mouth_l)
    _, mr_v = calc.axes.to_face(*mouth_r)
    _, mm_v = calc.axes.to_face(*mouth_mid)
    # positive if corners higher than center (smile) — report absolute mm proxy
    corner_delta_px = abs(((ml_v + mr_v) / 2.0) - mm_v)
    # scale to "mm" assuming face height ~180mm
    corner_mm = corner_delta_px / face_h * 180.0
    out.append(
        _pack(
            mid="mouth_corners",
            label="Положение уголков рта",
            category="губы",
            value=corner_mm,
            unit="mm",
            ideal_min=0.0,
            ideal_max=4.0,
            explanation="Насколько уголки рта выше/ниже центра губ (в условных мм).",
            points=[
                _pt(calc, "ml", mouth_l, "anchor", 61),
                _pt(calc, "mr", mouth_r, "anchor", 291),
                _pt(calc, "mm", mouth_mid, "anchor"),
            ],
            segments=[_seg(calc, mouth_l, mouth_r, style="primary", label=f"{corner_mm:.1f} mm")],
            formula={"type": "mouth_corners"},
            soft_margin=3.0,
        )
    )

    philtrum_len = 100.0 * _distance(nose_bottom, philtrum) / face_h
    out.append(
        _pack(
            mid="philtrum",
            label="Длина фильтрума",
            category="губы",
            value=philtrum_len,
            unit="%",
            ideal_min=4.5,
            ideal_max=7.5,
            explanation="Расстояние от основания носа до верхней губы.",
            points=[
                _pt(calc, "nb", nose_bottom, "anchor", 2),
                _pt(calc, "ph", philtrum, "anchor", 0),
            ],
            segments=[_seg(calc, nose_bottom, philtrum, style="primary", label=f"{philtrum_len:.1f}%")],
            formula={"type": "vertical_pct_face", "a": "nb", "b": "ph"},
            soft_margin=2.5,
        )
    )

    mouth_face = 100.0 * mouth_w / max(cheek_w, 1e-3)
    out.append(
        _pack(
            mid="mouth_face_width",
            label="Ширина рта / лицо",
            category="губы",
            value=mouth_face,
            unit="%",
            ideal_min=35.0,
            ideal_max=45.0,
            explanation="Ширина рта относительно ширины лица по скулам.",
            points=[
                _pt(calc, "ml", mouth_l, "anchor", 61),
                _pt(calc, "mr", mouth_r, "anchor", 291),
                _pt(calc, "lc", left_cheek, "ref", 234),
                _pt(calc, "rc", right_cheek, "ref", 454),
            ],
            segments=[
                _seg(calc, left_cheek, right_cheek, style="ref"),
                _seg(calc, mouth_l, mouth_r, style="primary", label=f"{mouth_face:.1f}%"),
            ],
            formula={"type": "pct_ratio", "num": ["ml", "mr"], "den": ["lc", "rc"]},
            soft_margin=8.0,
        )
    )

    # Chin / jaw
    chin_share = 100.0 * lower_len / face_h
    out.append(
        _pack(
            mid="chin_share",
            label="Доля подбородка (нижняя треть)",
            category="подбородок",
            value=chin_share,
            unit="%",
            ideal_min=30.0,
            ideal_max=36.0,
            explanation="Нижняя треть как доля высоты лица.",
            points=midline,
            segments=[_seg(calc, nose_pt, chin_pt, style="primary", label=f"{chin_share:.1f}%")],
            formula={"type": "thirds_segment", "segment": "lower"},
            soft_margin=5.0,
        )
    )

    # Chin taper: chin width at 176/400 vs jaw
    chin_l, chin_r = calc.px_index(176), calc.px_index(400)
    chin_taper = _distance(chin_l, chin_r) / max(jaw_w, 1e-3)
    out.append(
        _pack(
            mid="chin_taper",
            label="Сужение подбородка",
            category="подбородок",
            value=chin_taper,
            unit="x",
            ideal_min=0.28,
            ideal_max=0.40,
            explanation="Ширина подбородка к ширине челюсти.",
            points=[
                _pt(calc, "cl", chin_l, "anchor", 176),
                _pt(calc, "cr", chin_r, "anchor", 400),
                _pt(calc, "jl", left_jaw, "ref", 172),
                _pt(calc, "jr", right_jaw, "ref", 397),
            ],
            segments=[
                _seg(calc, left_jaw, right_jaw, style="ref"),
                _seg(calc, chin_l, chin_r, style="primary", label=f"{chin_taper:.2f}x"),
            ],
            formula={"type": "ratio_hw", "h1": "cl", "h2": "cr", "v1": "jl", "v2": "jr", "as_ratio": True},
            soft_margin=0.12,
        )
    )

    # Chin projection via z
    cheek_z = (calc.landmarks[234]["z"] + calc.landmarks[454]["z"]) / 2.0
    chin_z = calc.landmarks[152]["z"]
    # MediaPipe z: more negative = closer to camera often — use relative
    chin_proj = abs(chin_z - cheek_z) * 100.0
    out.append(
        _pack(
            mid="chin_projection",
            label="Проекция подбородка (глубина)",
            category="подбородок",
            value=chin_proj,
            unit="x",
            ideal_min=1.5,
            ideal_max=3.5,
            explanation="Относительная глубина подбородка. Надёжно только в ПРОФИЛЕ / ¾.",
            points=[_pt(calc, "chin", chin, "anchor", 152), _pt(calc, "lc", left_cheek, "ref", 234)],
            segments=[_seg(calc, chin, _midpoint(left_cheek, right_cheek), style="primary")],
            formula={"type": "static"},
            soft_margin=2.0,
            view="profile",
        )
    )

    # Gonial angle is a PROFILE metric (ramus vs mandibular plane). Skip on frontal.
    gonial = (
        _angle_at(left_cheek, left_jaw, chin) + _angle_at(right_cheek, right_jaw, chin)
    ) / 2.0
    out.append(
        _pack(
            mid="gonial_angle",
            label="Угол нижней челюсти (gonial)",
            category="челюсть",
            value=gonial,
            unit="°",
            ideal_min=110.0,
            ideal_max=130.0,
            explanation="Настоящий gonial angle меряется в ПРОФИЛЕ (ветвь / тело челюсти). В анфасе 2D-контур ненадёжен.",
            points=[
                _pt(calc, "lc", left_cheek, "anchor", 234),
                _pt(calc, "jl", left_jaw, "anchor", 172),
                _pt(calc, "chin", chin, "anchor", 152),
            ],
            segments=[
                _seg(calc, left_cheek, left_jaw, style="primary"),
                _seg(calc, left_jaw, chin, style="primary", label=f"{gonial:.1f}°"),
            ],
            formula={"type": "angle", "points": ["lc", "jl", "chin"]},
            soft_margin=12.0,
            view="profile",
        )
    )

    # Ear protrusion needs profile (or top-down). Frontal angle is meaningless.
    ear_l = calc.px_index(234)
    ear_l_top = calc.px_index(127)
    ear_line_angle = abs(_tilt_deg(ear_l, ear_l_top, ref_u, up))
    out.append(
        _pack(
            mid="ear_protrusion",
            label="Угол отстояния ушей",
            category="уши",
            value=ear_line_angle,
            unit="°",
            ideal_min=10.0,
            ideal_max=11.5,
            explanation="Угол отстояния уха меряется в ПРОФИЛЕ. В анфасе не оценивается.",
            points=[
                _pt(calc, "el", ear_l, "anchor", 234),
                _pt(calc, "et", ear_l_top, "anchor", 127),
            ],
            segments=[
                _seg(calc, ear_l, ear_l_top, style="primary", label=f"{ear_line_angle:.1f}°"),
            ],
            formula={"type": "ear_angle"},
            soft_margin=6.0,
            view="profile",
        )
    )

    # Cheek / jaw = inverse of FaceIQ bigonial% (ideal jaw 87.5–91.5% → cheek/jaw ≈ 1.09–1.14).
    cheek_jaw = cheek_w / max(jaw_w, 1e-3)
    out.append(
        _pack(
            mid="cheek_jaw_ratio",
            label="Скулы / челюсть",
            category="скулы",
            value=cheek_jaw,
            unit="x",
            ideal_min=1.09,
            ideal_max=1.15,
            explanation=(
                "Bizygomatic ÷ bigonial. Эквивалент FaceIQ bigonial 87.5–91.5% "
                "(cheek/jaw ≈ 1.09–1.14). Ниже 1.09 — очень широкая челюсть."
            ),
            points=[
                _pt(calc, "lc", left_cheek, "anchor", 234),
                _pt(calc, "rc", right_cheek, "anchor", 454),
                _pt(calc, "jl", soft_gonion_l, "ref", 132),
                _pt(calc, "jr", soft_gonion_r, "ref", 361),
            ],
            segments=[
                _seg(calc, soft_gonion_l, soft_gonion_r, style="ref"),
                _seg(calc, left_cheek, right_cheek, style="primary", label=f"{cheek_jaw:.2f}x"),
            ],
            formula={"type": "ratio_hw", "h1": "lc", "h2": "rc", "v1": "jl", "v2": "jr", "as_ratio": True},
            soft_margin=0.10,
            scale_pad=0.15,
        )
    )

    # Cheek / temple = inverse of FaceIQ bitemporal% (ideal 86.5–92.5% → ≈ 1.08–1.16).
    cheek_temple = cheek_w / max(temple_w, 1e-3)
    out.append(
        _pack(
            mid="cheek_temple_ratio",
            label="Скулы / виски",
            category="скулы",
            value=cheek_temple,
            unit="x",
            ideal_min=1.08,
            ideal_max=1.16,
            explanation=(
                "Скулы ÷ височная ширина. Эквивалент FaceIQ bitemporal 86.5–92.5%. "
                "Раньше идеал 0.95–1.08 был перевёрнут (виски шире скул — неверно)."
            ),
            points=[
                _pt(calc, "lc", left_cheek, "anchor", 234),
                _pt(calc, "rc", right_cheek, "anchor", 454),
                _pt(calc, "lt", left_temple, "ref", 54),
                _pt(calc, "rt", right_temple, "ref", 284),
            ],
            segments=[
                _seg(calc, left_temple, right_temple, style="ref"),
                _seg(calc, left_cheek, right_cheek, style="primary", label=f"{cheek_temple:.2f}x"),
            ],
            formula={"type": "ratio_hw", "h1": "lc", "h2": "rc", "v1": "lt", "v2": "rt", "as_ratio": True},
            soft_margin=0.10,
            scale_pad=0.15,
        )
    )

    face_ratio = cheek_w / max(face_h, 1.0)
    out.append(
        _pack(
            mid="face_ratio",
            label="Ширина / высота лица",
            category="пропорции",
            value=face_ratio,
            unit="x",
            ideal_min=0.71,
            ideal_max=0.77,
            explanation=(
                "Скулы ÷ высота (волосы→подбородок). Обратное к FaceIQ Total Facial H/W "
                "(идеал H/W 1.30–1.40 → W/H 0.71–0.77)."
            ),
            points=[
                _pt(calc, "lc", left_cheek, "anchor", 234),
                _pt(calc, "rc", right_cheek, "anchor", 454),
                _pt(calc, "hairline", hair, "anchor"),
                _pt(calc, "chin", chin, "anchor", 152),
            ],
            segments=[
                _seg(calc, left_cheek, right_cheek, style="primary"),
                _seg(calc, hair, chin, style="primary", label=f"{face_ratio:.2f}x"),
            ],
            formula={"type": "ratio_hw", "h1": "lc", "h2": "rc", "v1": "hairline", "v2": "chin"},
            soft_margin=0.08,
            scale_pad=0.12,
        )
    )

    # Facial fifths: outer edges = side of head at eye level (127/356), not forehead temples.
    # Use named canthi for eye/mid strips (same as One Eye Apart), not soft _canthi mix.
    fifths = [
        ("fifth_left_outer", "Левая внешняя пятая", face_side_l, eye_lo),
        ("fifth_left_eye", "Левая глазная пятая", eye_lo, eye_li),
        ("fifth_mid", "Средняя пятая (нос)", eye_li, eye_ri),
        ("fifth_right_eye", "Правая глазная пятая", eye_ri, eye_ro),
        ("fifth_right_outer", "Правая внешняя пятая", eye_ro, face_side_r),
    ]
    face_span = _distance(face_side_l, face_side_r)
    for fid, flabel, a, b in fifths:
        pct = 100.0 * _distance(a, b) / max(face_span, 1e-3)
        out.append(
            _pack(
                mid=fid,
                label=flabel,
                category="пятины",
                value=pct,
                unit="%",
                ideal_min=16.0,
                ideal_max=24.0,
                explanation=(
                    "Пятины: лицо ≈ 5 равных полос (~20% каждая). "
                    "Допуск FaceIQ/looksmax шире классических ±2%."
                ),
                points=[_pt(calc, "a", a, "anchor"), _pt(calc, "b", b, "anchor")],
                segments=[_seg(calc, a, b, style="primary", label=f"{pct:.1f}%")],
                formula={"type": "pct_of_span", "a": "a", "b": "b"},
                soft_margin=6.0,
                scale_pad=8.0,
            )
        )

    # Symmetry extras
    def side_delta(l_key: str, r_key: str) -> float:
        lu, _ = calc.face(l_key)
        ru, _ = calc.face(r_key)
        return abs(abs(lu) - abs(ru)) / max(abs(lu), abs(ru), 1e-3) * 100.0

    for sid, slabel, lk, rk in [
        ("sym_eyes", "Симметрия глаз", "left_eye_outer", "right_eye_outer"),
        ("sym_mouth", "Симметрия рта", "mouth_left", "mouth_right"),
        ("sym_cheeks", "Симметрия скул", "left_cheek", "right_cheek"),
        ("sym_jaw", "Симметрия челюсти", "jaw_left", "jaw_right"),
        ("sym_brows", "Симметрия бровей", "left_temple", "right_temple"),
    ]:
        # use landmark keys via face(); brows use indices
        if sid == "sym_brows":
            lu, lv = calc.face_index(70)
            ru, rv = calc.face_index(300)
            val = abs(abs(lu) - abs(ru)) / max(abs(lu), abs(ru), 1e-3) * 100.0
            pts = [
                _pt(calc, "l", calc.px_index(70), "anchor", 70),
                _pt(calc, "r", calc.px_index(300), "anchor", 300),
            ]
        else:
            val = side_delta(lk, rk)
            lp, rp = calc.px(lk), calc.px(rk)
            pts = [
                _pt(calc, "l", lp, "anchor"),
                _pt(calc, "r", rp, "anchor"),
            ]
        p1 = (pts[0]["x"] * calc.width, pts[0]["y"] * calc.height)
        p2 = (pts[1]["x"] * calc.width, pts[1]["y"] * calc.height)
        out.append(
            _pack(
                mid=sid,
                label=slabel,
                category="симметрия",
                value=val,
                unit="%",
                ideal_min=0.0,
                ideal_max=6.0,
                explanation=(
                    "Разница |лево| vs |право| от средней оси лица. "
                    "0% = идеал; на фото с лёгким поворотом 5–10% — нормально, не «кривое лицо»."
                ),
                points=pts,
                segments=[_seg(calc, p1, p2, style="primary", label=f"{val:.1f}%")],
                formula={"type": "symmetry_pct"},
                soft_margin=10.0,
                scale_pad=12.0,
            )
        )

    # FaceIQ «Total Facial Width to Height» = height/bizygomatic, ideal 1.30–1.40 (не φ=1.618).
    golden = face_h / max(cheek_w, 1e-3)
    out.append(
        _pack(
            mid="golden_face",
            label="Высота / ширина лица",
            category="пропорции",
            value=golden,
            unit="x",
            ideal_min=1.30,
            ideal_max=1.40,
            explanation=(
                "Высота (волосы→подбородок) ÷ ширина скул. "
                "FaceIQ Total Facial H/W: 1.30–1.40 (не «золотое φ≈1.618» — это миф для лица)."
            ),
            points=[
                _pt(calc, "hairline", hair, "anchor"),
                _pt(calc, "chin", chin, "anchor", 152),
                _pt(calc, "lc", left_cheek, "anchor", 234),
                _pt(calc, "rc", right_cheek, "anchor", 454),
            ],
            segments=[
                _seg(calc, hair, chin, style="primary", label=f"{golden:.2f}x"),
                _seg(calc, left_cheek, right_cheek, style="ref"),
            ],
            formula={"type": "ratio_hw", "h1": "hairline", "h2": "chin", "v1": "lc", "v2": "rc", "as_ratio": True},
            soft_margin=0.18,
            scale_pad=0.20,
        )
    )

    # Midface ratio
    midface_r = mid_len / max(lower_len, 1e-3)
    out.append(
        _pack(
            mid="midface_ratio",
            label="Midface / lower face",
            category="пропорции",
            value=midface_r,
            unit="x",
            ideal_min=0.90,
            ideal_max=1.05,
            explanation="Средняя зона лица к нижней.",
            points=midline,
            segments=[
                _seg(calc, brow_pt, nose_pt, style="primary"),
                _seg(calc, nose_pt, chin_pt, style="ref", label=f"{midface_r:.2f}x"),
            ],
            formula={"type": "thirds_ratio", "a": "mid", "b": "lower"},
            soft_margin=0.12,
        )
    )

    # Nose deviation from midline
    nose_u, _ = calc.face("nose_tip")
    nose_dev = abs(nose_u) / max(cheek_w / 2.0, 1.0) * 100.0
    out.append(
        _pack(
            mid="nose_deviation",
            label="Отклонение носа от оси",
            category="нос",
            value=nose_dev,
            unit="%",
            ideal_min=0.0,
            ideal_max=3.0,
            explanation="Смещение кончика носа от срединной оси лица.",
            points=[
                _pt(calc, "tip", nose_tip, "anchor", 1),
                _pt(calc, "hairline", hair, "ref"),
                _pt(calc, "chin", chin, "ref", 152),
            ],
            segments=[
                _seg(calc, hair, chin, style="ref"),
                _seg(calc, nose_tip, _midpoint(hair, chin), style="primary", label=f"{nose_dev:.1f}%"),
            ],
            formula={"type": "static"},
            soft_margin=5.0,
        )
    )

    # Eye aperture absolute
    aperture = (le_h / max(le_w, 1e-3) + re_h / max(re_w, 1e-3)) / 2.0
    out.append(
        _pack(
            mid="eye_aperture",
            label="Высота глазной щели",
            category="глаза",
            value=aperture,
            unit="x",
            ideal_min=0.28,
            ideal_max=0.42,
            explanation="Высота глаза к его ширине (открытость).",
            points=[
                _pt(calc, "lt", calc.px("left_eye_top"), "anchor", 159),
                _pt(calc, "lb", calc.px("left_eye_bottom"), "anchor", 145),
                _pt(calc, "li", left_inner, "ref"),
                _pt(calc, "lo", left_outer, "ref"),
            ],
            segments=[
                _seg(calc, calc.px("left_eye_top"), calc.px("left_eye_bottom"), style="primary", label=f"{aperture:.2f}x"),
                _seg(calc, left_inner, left_outer, style="ref"),
            ],
            formula={"type": "eye_aperture"},
            soft_margin=0.1,
        )
    )

    # FaceIQ Eye Separation Ratio = interpupillary ÷ bizygomatic (ideal ~44.3–47.7%).
    ipd = _distance(pupil_l, pupil_r)
    esr_pct = 100.0 * ipd / max(cheek_w, 1e-3)
    out.append(
        _pack(
            mid="outer_eye_span",
            label="Eye Separation Ratio (IPD)",
            category="глаза",
            value=esr_pct,
            unit="%",
            ideal_min=44.3,
            ideal_max=47.7,
            explanation=(
                "FaceIQ ESR: межзрачковое ÷ ширина скул. Идеал 44.3–47.7% "
                "(узкая полоса FaceIQ ~45.7–46.8%). Не путать с шириной по внешним углам глаз."
            ),
            points=[
                _pt(calc, "pl", pupil_l, "anchor"),
                _pt(calc, "pr", pupil_r, "anchor"),
                _pt(calc, "lc", left_cheek, "ref", 234),
                _pt(calc, "rc", right_cheek, "ref", 454),
            ],
            segments=[
                _seg(calc, left_cheek, right_cheek, style="ref"),
                _seg(calc, pupil_l, pupil_r, style="primary", label=f"{esr_pct:.1f}%"),
            ],
            formula={"type": "pct_ratio", "num": ["pl", "pr"], "den": ["lc", "rc"]},
            soft_margin=2.2,
            scale_pad=4.0,
        )
    )

    # FaceIQ «Lower Third Proportion»: (nose→mouth) / (nose→chin). Ideal 31–33.5%.
    # NOT the same as facial lower-third % of full face height.
    mouth_mid = _midpoint(mouth_l, mouth_r)
    _, mouth_v = calc.axes.to_face(*mouth_mid)
    nose_to_mouth = abs(mouth_v - nose_v)
    nose_to_chin = max(abs(cv - nose_v), 1e-3)
    lower_prop = 100.0 * nose_to_mouth / nose_to_chin
    out.append(
        _pack(
            mid="lower_face_total",
            label="Lower third proportion",
            category="челюсть",
            value=lower_prop,
            unit="%",
            ideal_min=31.0,
            ideal_max=33.5,
            explanation=(
                "FaceIQ: (нос→рот) / (нос→подбородок). Идеал 31–33.5%. "
                "Не путать с нижней третью всего лица (~33–38%)."
            ),
            points=[
                _pt(calc, "nose", nose_pt, "anchor", 2),
                _pt(calc, "mouth", mouth_mid, "anchor"),
                _pt(calc, "chin", chin_pt, "anchor", 152),
            ],
            segments=[
                _seg(calc, nose_pt, chin_pt, style="ref"),
                _seg(calc, nose_pt, mouth_mid, style="primary", label=f"{lower_prop:.1f}%"),
            ],
            formula={"type": "pct_ratio", "num": ["nose", "mouth"], "den": ["nose", "chin"]},
            soft_margin=3.0,
            scale_pad=5.0,
        )
    )

    # Cheek vertical position within midface (eye → nose), NOT hairline→chin.
    # High cheekbones ≈ 18–28% of the eye–nose span (matches aesthetics.cheek_pos).
    eye_v = (calc.face("left_eye_outer")[1] + calc.face("right_eye_outer")[1]) / 2.0
    midface_span = max(nose_v - eye_v, 1e-3)
    cheek_pos_mid = (cheek_v - eye_v) / midface_span
    cheek_pos_pct = 100.0 * cheek_pos_mid
    out.append(
        _pack(
            mid="cheek_position",
            label="Положение скул по вертикали",
            category="скулы",
            value=cheek_pos_pct,
            unit="%",
            ideal_min=18.0,
            ideal_max=28.0,
            explanation=(
                "Положение скул в зоне глаз→нос (меньше % = выше к глазам). "
                "Идеал высоких скул ≈ 18–28%."
            ),
            points=[
                _pt(calc, "lc", left_cheek, "anchor", 234),
                _pt(calc, "rc", right_cheek, "anchor", 454),
                _pt(calc, "lo", calc.px("left_eye_outer"), "ref", 33),
                _pt(calc, "nb", nose_bottom, "ref", 2),
            ],
            segments=[
                _seg(calc, calc.px("left_eye_outer"), nose_bottom, style="ref"),
                _seg(calc, left_cheek, right_cheek, style="primary", label=f"{cheek_pos_pct:.1f}%"),
            ],
            formula={"type": "cheek_pos"},
            soft_margin=10.0,
        )
    )

    # Brow width
    brow_w = _distance(brow_l, brow_r)
    brow_pct = 100.0 * brow_w / max(cheek_w, 1e-3)
    out.append(
        _pack(
            mid="brow_width",
            label="Ширина бровей",
            category="брови",
            value=brow_pct,
            unit="%",
            ideal_min=70.0,
            ideal_max=85.0,
            explanation="Разлёт бровей относительно ширины скул.",
            points=[
                _pt(calc, "bl", brow_l, "anchor", 70),
                _pt(calc, "br", brow_r, "anchor", 300),
            ],
            segments=[_seg(calc, brow_l, brow_r, style="primary", label=f"{brow_pct:.1f}%")],
            formula={"type": "pct_ratio", "num": ["bl", "br"], "den": ["lc", "rc"]},
            soft_margin=10.0,
        )
    )

    # Pose gate: do not score profile-only metrics on frontal (and vice versa).
    pose = str(calc.pose_info.get("pose", "frontal"))
    if pose == "profile":
        allowed = {"profile", "any"}
    elif pose == "three_quarter":
        allowed = {"frontal", "three_quarter", "any"}
    else:
        allowed = {"frontal", "any"}

    out = [m for m in out if m.get("view", "frontal") in allowed]

    for i, m in enumerate(out):
        m["order"] = i + 1
        m["total"] = len(out)

    return out

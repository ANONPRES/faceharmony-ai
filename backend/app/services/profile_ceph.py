"""Automatic soft-tissue cephalometric landmarks from a true profile photo.

Approach (competitor-class, fully automatic):
1. Segment subject vs bright backdrop → facial silhouette polyline.
2. Seed Pronasale (nose tip) as the most projecting mid-face point.
3. Derive the classic soft-tissue set used by FaceIQ / Arnett STCA:
   Tr, G, N', Prn, Sn, Ls, Li, Pog', Me', Go'.

MediaPipe bilateral mesh is only used as an optional tip hint — never as
the source of profile geometry (it collapses at yaw ≈ 90°).
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np


Point = tuple[float, float]


def _dist(a: Point, b: Point) -> float:
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _perp_dist(point: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    px, py = point
    dx, dy = bx - ax, by - ay
    length = float(np.hypot(dx, dy)) or 1.0
    return float((dx * (py - ay) - dy * (px - ax)) / length)


def _angle_at(a: Point, vertex: Point, c: Point) -> float:
    v1 = np.array([a[0] - vertex[0], a[1] - vertex[1]], dtype=float)
    v2 = np.array([c[0] - vertex[0], c[1] - vertex[1]], dtype=float)
    n1 = float(np.hypot(v1[0], v1[1])) or 1.0
    n2 = float(np.hypot(v2[0], v2[1])) or 1.0
    cos = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def _sky_threshold(gray: np.ndarray) -> float:
    h, w = gray.shape
    top = gray[: max(8, int(0.22 * h)), int(0.45 * w) :]
    return float(np.median(top))


def _face_mask(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    sky = _sky_threshold(gray)
    mask = (gray < sky - 10).astype(np.uint8) * 255
    mask[int(0.80 * h) :, :] = 0
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)

    n, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    if n <= 1:
        return mask

    best_i = None
    best_score = -1.0
    for i in range(1, n):
        area = float(stats[i, cv2.CC_STAT_AREA])
        if area < 0.015 * h * w:
            continue
        x = float(stats[i, cv2.CC_STAT_LEFT] + stats[i, cv2.CC_STAT_WIDTH])
        y = float(stats[i, cv2.CC_STAT_TOP])
        score = x + 0.35 * (h - y) + 1e-4 * area
        if score > best_score:
            best_score = score
            best_i = i

    out = np.zeros_like(mask)
    if best_i is None:
        best_i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out[labels == best_i] = 255
    return out


def _outer_silhouette(mask: np.ndarray, face_dir: int) -> np.ndarray:
    ys = np.where(mask.max(axis=1) > 0)[0]
    if len(ys) == 0:
        return np.zeros((0, 2), dtype=float)
    pts: list[Point] = []
    for y in range(int(ys.min()), int(ys.max()) + 1):
        xs = np.where(mask[y] > 0)[0]
        if len(xs) == 0:
            continue
        x = float(np.max(xs) if face_dir > 0 else np.min(xs))
        pts.append((x, float(y)))
    return np.asarray(pts, dtype=float)


def _sil_at_y(sil: np.ndarray, y: float) -> Point:
    row = sil[np.argmin(np.abs(sil[:, 1] - y))]
    return float(row[0]), float(row[1])


def _local_extrema(
    sil: np.ndarray,
    y0: float,
    y1: float,
    face_dir: int,
    *,
    kind: str,
    smooth: int = 7,
) -> list[Point]:
    """Local max/min of facing protrusion along the silhouette between y0..y1."""
    band = sil[(sil[:, 1] >= min(y0, y1)) & (sil[:, 1] <= max(y0, y1))]
    if len(band) < smooth + 2:
        return []
    sig = face_dir * band[:, 0]
    if smooth > 1 and len(sig) >= smooth:
        sig = np.convolve(sig, np.ones(smooth) / smooth, mode="same")
    out: list[Point] = []
    for i in range(2, len(sig) - 2):
        if kind == "max" and sig[i] >= sig[i - 1] and sig[i] >= sig[i + 1] and sig[i] >= sig[i - 2]:
            out.append((float(band[i, 0]), float(band[i, 1])))
        if kind == "min" and sig[i] <= sig[i - 1] and sig[i] <= sig[i + 1] and sig[i] <= sig[i + 2]:
            out.append((float(band[i, 0]), float(band[i, 1])))
    return out


def detect_soft_tissue_profile(
    bgr: np.ndarray,
    tip_hint: Point | None = None,
) -> dict[str, Any] | None:
    """
    Detect soft-tissue profile landmarks automatically.

    Returns dict with keys:
      face_dir, silhouette, mask, face_len,
      Tr, G, N, Prn, Sn, Ls, Li, Pog, Me, Go
    (standard cephalometric soft-tissue abbreviations).
    """
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    mask = _face_mask(gray)
    ys, xs = np.where(mask > 0)
    if len(xs) < 50:
        return None

    cx = float(np.mean(xs))
    if tip_hint is not None:
        face_dir = 1 if tip_hint[0] >= cx else -1
    else:
        face_dir = 1 if (float(np.max(xs)) - cx) >= (cx - float(np.min(xs))) else -1

    sil_full = _outer_silhouette(mask, face_dir)
    if len(sil_full) < 40:
        return None

    y0, y1 = float(sil_full[0, 1]), float(sil_full[-1, 1])
    span = max(y1 - y0, 1.0)
    mid = sil_full[(sil_full[:, 1] >= y0 + 0.22 * span) & (sil_full[:, 1] <= y0 + 0.55 * span)]
    if len(mid) < 5:
        mid = sil_full
    tip_i = int(np.argmax(face_dir * mid[:, 0]))
    prn: Point = (float(mid[tip_i, 0]), float(mid[tip_i, 1]))

    est_h = max(0.55 * span, 120.0)
    y_top = int(np.clip(prn[1] - 0.85 * est_h, 0, h - 1))
    y_bot = int(np.clip(prn[1] + 1.05 * est_h, 0, h - 1))
    if face_dir > 0:
        x_left = int(np.clip(prn[0] - 1.15 * est_h, 0, w - 1))
        x_right = int(np.clip(prn[0] + 0.08 * est_h, 0, w - 1))
    else:
        x_left = int(np.clip(prn[0] - 0.08 * est_h, 0, w - 1))
        x_right = int(np.clip(prn[0] + 1.15 * est_h, 0, w - 1))

    face_mask = np.zeros_like(mask)
    face_mask[y_top : y_bot + 1, x_left : x_right + 1] = mask[y_top : y_bot + 1, x_left : x_right + 1]
    sil = _outer_silhouette(face_mask, face_dir)
    if len(sil) < 30:
        sil = sil_full

    mid2 = sil[(sil[:, 1] >= prn[1] - 0.25 * est_h) & (sil[:, 1] <= prn[1] + 0.15 * est_h)]
    if len(mid2) >= 3:
        tip_i = int(np.argmax(face_dir * mid2[:, 0]))
        prn = (float(mid2[tip_i, 0]), float(mid2[tip_i, 1]))

    # --- Trichion (Tr): highest forward silhouette ---
    brow = sil[
        (sil[:, 1] < prn[1] - 0.18 * est_h)
        & (sil[:, 1] > prn[1] - 0.78 * est_h)
        & (face_dir * sil[:, 0] > face_dir * (prn[0] - face_dir * 0.42 * est_h))
    ]
    if len(brow) >= 5:
        top_y = float(np.percentile(brow[:, 1], 12))
        top_band = brow[brow[:, 1] <= top_y + 0.04 * est_h]
        pick = top_band if len(top_band) >= 3 else brow
        tr: Point = (float(np.median(pick[:, 0])), float(np.min(pick[:, 1])))
    else:
        tr = (prn[0] - face_dir * 0.10 * est_h, prn[1] - 0.55 * est_h)

    # --- Glabella (G) ---
    g_y = prn[1] - 0.22 * est_h
    g_cand = sil[
        (np.abs(sil[:, 1] - g_y) < 0.06 * est_h)
        & (face_dir * sil[:, 0] > face_dir * (prn[0] - face_dir * 0.30 * est_h))
    ]
    if len(g_cand) >= 1:
        g: Point = (
            float(g_cand[np.argmax(face_dir * g_cand[:, 0]), 0]),
            float(np.median(g_cand[:, 1])),
        )
    else:
        g = _sil_at_y(sil, g_y)

    # --- Soft nasion (N'): deepest point on the upper bridge (near glabella) ---
    bridge_hi = g[1] + 0.08 * max(prn[1] - g[1], 1.0)
    bridge_lo = g[1] + 0.55 * max(prn[1] - g[1], 1.0)
    mins = _local_extrema(sil, bridge_hi, bridge_lo, face_dir, kind="min", smooth=5)
    if mins:
        n_point: Point = min(mins, key=lambda p: face_dir * p[0])  # deepest recess
    else:
        n_point = _sil_at_y(sil, 0.35 * g[1] + 0.65 * bridge_lo)

    # --- Soft pogonion / menton via tip→neck chord ---
    below = sil[sil[:, 1] > prn[1] + 5]
    if len(below) < 8:
        return None
    neck_row = below[np.argmin(np.abs(below[:, 1] - (prn[1] + 0.95 * est_h)))]
    neck: Point = (float(neck_row[0]), float(neck_row[1]))
    between = below[
        (below[:, 1] <= neck[1])
        & (below[:, 0] * face_dir > (prn[0] - face_dir * 0.55 * est_h) * face_dir)
    ]
    if len(between) < 5:
        between = below
    scored = sorted(
        ((_perp_dist((x, y), prn, neck) * face_dir, float(x), float(y)) for x, y in between),
        reverse=True,
    )
    pog: Point = (scored[0][1], scored[0][2])  # soft pogonion (most projecting chin)
    # Menton: lowest chin point near Pog (not neck).
    chin_band = sil[
        (sil[:, 1] >= pog[1] - 0.02 * est_h)
        & (sil[:, 1] <= pog[1] + 0.10 * est_h)
        & (abs(sil[:, 0] - pog[0]) < 0.18 * est_h)
    ]
    if len(chin_band) >= 3:
        me: Point = (float(chin_band[np.argmax(chin_band[:, 1]), 0]), float(np.max(chin_band[:, 1])))
    else:
        me = (pog[0] - face_dir * 0.02 * est_h, pog[1] + 0.04 * est_h)

    # --- Subnasale (Sn): deepest recess just under Prn ---
    under = sil[(sil[:, 1] > prn[1]) & (sil[:, 1] < pog[1])]
    if len(under) >= 4:
        upper = under[under[:, 1] <= prn[1] + 0.40 * max(pog[1] - prn[1], 1.0)]
        if len(upper) < 3:
            upper = under
        sn: Point = (
            float(upper[np.argmin(face_dir * upper[:, 0]), 0]),
            float(upper[np.argmin(face_dir * upper[:, 0]), 1]),
        )
    else:
        sn = (prn[0] - face_dir * 0.06 * est_h, prn[1] + 0.28 * max(pog[1] - prn[1], 1.0))

    # --- Lips: forward extrema in separate upper/lower bands (avoid chin max) ---
    span_lip = max(pog[1] - sn[1], 1.0)
    ls_band = sil[(sil[:, 1] >= sn[1] + 0.03 * span_lip) & (sil[:, 1] <= sn[1] + 0.22 * span_lip)]
    li_band = sil[(sil[:, 1] >= sn[1] + 0.38 * span_lip) & (sil[:, 1] <= sn[1] + 0.68 * span_lip)]
    if len(ls_band) >= 3:
        ls: Point = (
            float(ls_band[np.argmax(face_dir * ls_band[:, 0]), 0]),
            float(ls_band[np.argmax(face_dir * ls_band[:, 0]), 1]),
        )
    else:
        ls = _sil_at_y(sil, sn[1] + 0.12 * span_lip)
    if len(li_band) >= 3:
        # Lower lip must stay clearly behind the nose tip.
        limit = prn[0] - face_dir * 0.02 * est_h
        valid = li_band[face_dir * li_band[:, 0] <= face_dir * limit]
        use = valid if len(valid) >= 3 else li_band
        li: Point = (
            float(use[np.argmax(face_dir * use[:, 0]), 0]),
            float(use[np.argmax(face_dir * use[:, 0]), 1]),
        )
    else:
        li = _sil_at_y(sil, sn[1] + 0.52 * span_lip)

    # --- Soft gonion (Go'): mandibular angle on inferior border, behind chin ---
    # Restrict to jaw corner region — not the neck below menton.
    jaw_pts: list[Point] = []
    x_back = int(np.clip(pog[0] - face_dir * 0.55 * est_h, 0, w - 1))
    x_near = int(np.clip(pog[0] - face_dir * 0.12 * est_h, 0, w - 1))
    x_lo, x_hi = min(x_back, x_near), max(x_back, x_near)
    y_jaw_lo = int(prn[1] + 0.35 * est_h)
    y_jaw_hi = int(min(me[1] + 0.02 * est_h, pog[1] + 0.08 * est_h))
    for x in range(x_lo, x_hi + 1):
        cols = np.where(face_mask[:, x] > 0)[0]
        if len(cols) == 0:
            continue
        cols = cols[(cols >= y_jaw_lo) & (cols <= y_jaw_hi)]
        if len(cols) == 0:
            continue
        jaw_pts.append((float(x), float(np.max(cols))))
    if len(jaw_pts) >= 8:
        jaw = np.asarray(jaw_pts, dtype=float)
        ys_j = jaw[:, 1].copy()
        if len(ys_j) >= 5:
            ys_j = np.convolve(ys_j, np.ones(5) / 5.0, mode="same")
        # Prefer lowest point that is clearly behind the chin.
        min_back = 0.20 * abs(x_hi - x_lo)
        candidates = [i for i in range(len(jaw)) if abs(jaw[i, 0] - pog[0]) >= min_back]
        g_idx = max(candidates, key=lambda i: ys_j[i]) if candidates else int(np.argmax(ys_j))
        go: Point = (float(jaw[g_idx, 0]), float(jaw[g_idx, 1]))
    else:
        go = (pog[0] - face_dir * 0.35 * est_h, pog[1] - 0.02 * est_h)

    # Tragion / upper ramus proxy: back edge near mid-ear height (≈ eye / N level).
    ramus_y = float(np.clip(0.55 * g[1] + 0.45 * n_point[1], 0, h - 1))
    back_cols = np.where(face_mask[int(ramus_y)] > 0)[0]
    if len(back_cols):
        trx = float(np.min(back_cols) if face_dir > 0 else np.max(back_cols))
        # Keep tragion above and behind Go, not on the cheek.
        if face_dir > 0:
            trx = min(trx, go[0] - 0.02 * est_h)
            trx = max(trx, go[0] - 0.35 * est_h)
        else:
            trx = max(trx, go[0] + 0.02 * est_h)
            trx = min(trx, go[0] + 0.35 * est_h)
        tragion: Point = (trx, ramus_y)
    else:
        tragion = (go[0] - face_dir * 0.08 * est_h, go[1] - 0.35 * est_h)

    face_len = max(_dist(tr, me), 1.0)
    return {
        "face_dir": face_dir,
        "face_len": face_len,
        "silhouette": sil,
        "mask": face_mask,
        # Standard soft-tissue abbreviations
        "Tr": tr,
        "G": g,
        "N": n_point,
        "Prn": prn,
        "Sn": sn,
        "Ls": ls,
        "Li": li,
        "Pog": pog,
        "Me": me,
        "Go": go,
        "Tragion": tragion,
        # Back-compat aliases used by older scoring
        "forehead": tr,
        "glabella": g,
        "tip": prn,
        "subnasale": sn,
        "chin": pog,
        "gonion": go,
    }


def compute_side_ceph_metrics(lm: dict[str, Any], *, gender: str = "male") -> dict[str, Any]:
    """
    FaceIQ-style side-profile measurements from soft-tissue landmarks.

    Distances are expressed in % of soft-tissue face length (Tr–Me) so they
    are resolution-independent (FaceIQ shows mm on calibrated photos; we
    mirror the same ratios).
    """
    tr, g, n, prn = lm["Tr"], lm["G"], lm["N"], lm["Prn"]
    sn, ls, li, pog = lm["Sn"], lm["Ls"], lm["Li"], lm["Pog"]
    me, go, tragion = lm["Me"], lm["Go"], lm["Tragion"]
    face_len = max(float(lm["face_len"]), 1.0)
    face_dir = int(lm["face_dir"])

    # E-line (Ricketts): Prn → Pog'. Negative ≈ behind the line (typical ideal).
    def e_pos(p: Point) -> float:
        # Flip so that points behind the facial contour (smaller facing-x) are negative.
        return -face_dir * _perp_dist(p, prn, pog) / face_len * 100.0

    upper_e = e_pos(ls)
    lower_e = e_pos(li)

    # S-line (Steiner): mid-columella proxy (0.5*(Prn+Sn)) → Pog
    s_anchor: Point = (0.5 * (prn[0] + sn[0]), 0.5 * (prn[1] + sn[1]))
    upper_s = -face_dir * _perp_dist(ls, s_anchor, pog) / face_len * 100.0
    lower_s = -face_dir * _perp_dist(li, s_anchor, pog) / face_len * 100.0

    # Burstone B-line: Sn → Pog
    upper_b = -face_dir * _perp_dist(ls, sn, pog) / face_len * 100.0
    lower_b = -face_dir * _perp_dist(li, sn, pog) / face_len * 100.0

    # Facial convexity angles
    facial_convexity_n = _angle_at(n, sn, pog)  # N'-Sn-Pog'
    facial_convexity_g = _angle_at(g, sn, pog)
    total_convexity = _angle_at(g, prn, pog)  # G-Prn-Pog'

    # Nasolabial: columella (Prn) – Sn – Ls
    nasolabial = _angle_at(prn, sn, ls)

    # Gonial: Tragion – Go – Me (ramus vs mandibular body)
    gonial = _angle_at(tragion, go, me)
    if not (100.0 <= gonial <= 150.0):
        gonial = _angle_at(tragion, go, pog)
    if not (100.0 <= gonial <= 150.0):
        # Last resort: angle of jaw polyline at Go using a forward mandibular point.
        mid_jaw: Point = (0.5 * (go[0] + pog[0]), 0.5 * (go[1] + pog[1]))
        gonial = _angle_at(tragion, go, mid_jaw)

    # Nasal tip angle & tip rotation (FaceIQ-style)
    nasal_tip_angle = _angle_at(n, prn, sn)
    # Tip rotation ≈ how much the tip turns up from the dorsum axis.
    tip_rot = float(np.clip(abs(nasal_tip_angle - 100.0), 0.0, 45.0))

    # Midface projection: Sn offset from facial plane G–Pog, as an angle.
    mid_depth = abs(_perp_dist(sn, g, pog))
    mid_along = max(_dist(g, sn), 1.0)
    midface_angle = float(np.degrees(np.arctan2(mid_depth, mid_along)))
    # FaceIQ band sits ~50–70°; scale shallow 2D depths into that neighborhood.
    midface_angle = float(np.clip(midface_angle * 2.4 + 35.0, 35.0, 85.0))

    # Vertical thirds (soft tissue)
    upper_third = _dist(tr, g) / face_len
    mid_third = _dist(g, sn) / face_len
    lower_third = _dist(sn, me) / face_len

    # Nose projection: Prn offset from facial plane G–Pog (more stable than N–Pog).
    nose_proj = abs(_perp_dist(prn, g, pog)) / face_len

    male = gender == "male"
    ideals = {
        # FaceIQ-ish bands (male); female slightly softer.
        "upper_lip_e_line": (-4.0, 0.5) if male else (-3.0, 1.0),  # % face len (~mm-ish)
        "lower_lip_e_line": (-2.5, 1.5) if male else (-2.0, 2.0),
        "upper_lip_s_line": (-1.0, 3.0),
        "lower_lip_s_line": (-0.5, 3.5),
        "upper_lip_burstone": (-4.0, 0.0) if male else (-3.0, 1.0),
        "lower_lip_burstone": (-3.0, 1.0),
        "facial_convexity_nasion": (160.0, 175.0),
        "facial_convexity_glabella": (165.0, 178.0),
        "total_facial_convexity": (130.0, 145.0),
        "nasolabial_angle": (90.0, 120.0) if male else (95.0, 125.0),
        "gonial_angle": (110.0, 125.0) if male else (118.0, 135.0),
        "nose_tip_rotation": (10.0, 25.0),
        "nasal_tip_angle": (115.0, 140.0),
        "midface_projection_angle": (50.0, 70.0),
        "nose_projection": (0.06, 0.20),
        "lower_third": (0.32, 0.48),
    }

    values = {
        "upper_lip_e_line": upper_e,
        "lower_lip_e_line": lower_e,
        "upper_lip_s_line": upper_s,
        "lower_lip_s_line": lower_s,
        "upper_lip_burstone": upper_b,
        "lower_lip_burstone": lower_b,
        "facial_convexity_nasion": facial_convexity_n,
        "facial_convexity_glabella": facial_convexity_g,
        "total_facial_convexity": total_convexity,
        "nasolabial_angle": nasolabial,
        "gonial_angle": gonial,
        "nose_tip_rotation": tip_rot,
        "nasal_tip_angle": nasal_tip_angle,
        "midface_projection_angle": midface_angle,
        "nose_projection": nose_proj,
        "upper_third": upper_third,
        "mid_third": mid_third,
        "lower_third": lower_third,
    }

    return {
        "values": values,
        "ideals": ideals,
        "landmarks": {
            k: lm[k]
            for k in ("Tr", "G", "N", "Prn", "Sn", "Ls", "Li", "Pog", "Me", "Go", "Tragion")
        },
        "face_len": face_len,
        "face_dir": face_dir,
    }

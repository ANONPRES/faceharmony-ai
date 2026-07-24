"""Face geometry helpers: roll/tilt, hairline, and pose classification.

MediaPipe landmarks live in image space. For tilted heads we build a face
frame from the eye line so thirds/symmetry/measurements follow the face,
not the photo edges.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass
class FaceAxes:
    """
    2D face-aligned coordinate frame in pixel space.

    u runs left→right along the eye line (accounts for roll/tilt).
    v runs top→bottom perpendicular to the eye line (toward chin).
    """

    cx: float
    cy: float
    ux: float
    uy: float
    vx: float
    vy: float
    roll_rad: float

    @property
    def roll_deg(self) -> float:
        """Head roll in degrees (positive = clockwise in image coords)."""
        return float(np.degrees(self.roll_rad))

    def to_face(self, x: float, y: float) -> tuple[float, float]:
        """Convert image pixel to face-local (u, v)."""
        dx = x - self.cx
        dy = y - self.cy
        u = dx * self.ux + dy * self.uy
        v = dx * self.vx + dy * self.vy
        return float(u), float(v)

    def to_image(self, u: float, v: float) -> tuple[float, float]:
        """Convert face-local (u, v) back to image pixels."""
        x = self.cx + u * self.ux + v * self.vx
        y = self.cy + u * self.uy + v * self.vy
        return float(x), float(y)

    def line_across(
        self,
        v: float,
        half_span: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Horizontal-in-face line at constant v, spanning ±half_span in u."""
        return self.to_image(-half_span, v), self.to_image(half_span, v)

    def line_along(
        self,
        u: float,
        v0: float,
        v1: float,
    ) -> tuple[tuple[float, float], tuple[float, float]]:
        """Vertical-in-face line at constant u between v0 and v1."""
        return self.to_image(u, v0), self.to_image(u, v1)


def build_face_axes(
    landmarks: list[dict[str, Any]],
    width: int,
    height: int,
) -> FaceAxes:
    """
    Build face axes from eye corners and chin.

    Args:
        landmarks: Normalized MediaPipe landmarks.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        FaceAxes aligned to the subject's head roll.
    """
    left_eye = (
        (landmarks[33]["x"] + landmarks[133]["x"]) / 2.0 * width,
        (landmarks[33]["y"] + landmarks[133]["y"]) / 2.0 * height,
    )
    right_eye = (
        (landmarks[263]["x"] + landmarks[362]["x"]) / 2.0 * width,
        (landmarks[263]["y"] + landmarks[362]["y"]) / 2.0 * height,
    )
    chin = (landmarks[152]["x"] * width, landmarks[152]["y"] * height)
    glabella = (landmarks[9]["x"] * width, landmarks[9]["y"] * height)

    # Roll from inter-eye vector.
    dx = right_eye[0] - left_eye[0]
    dy = right_eye[1] - left_eye[1]
    roll = float(np.arctan2(dy, dx))

    ux, uy = float(np.cos(roll)), float(np.sin(roll))
    # Perpendicular "down" candidate.
    vx, vy = -uy, ux

    # Ensure +v points toward chin (down the face), not toward forehead.
    eye_mid = ((left_eye[0] + right_eye[0]) / 2.0, (left_eye[1] + right_eye[1]) / 2.0)
    to_chin = (chin[0] - eye_mid[0], chin[1] - eye_mid[1])
    if to_chin[0] * vx + to_chin[1] * vy < 0:
        vx, vy = -vx, -vy

    # Origin near glabella / mid-face for stable overlays.
    cx = (glabella[0] + eye_mid[0]) / 2.0
    cy = (glabella[1] + eye_mid[1]) / 2.0

    return FaceAxes(cx=cx, cy=cy, ux=ux, uy=uy, vx=vx, vy=vy, roll_rad=roll)


def _skin_mask(bgr: np.ndarray) -> np.ndarray:
    """Binary skin mask in YCrCb."""
    ycrcb = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    mask = cv2.inRange(ycrcb, (0, 133, 77), (255, 173, 127))
    return cv2.medianBlur(mask, 5)


def estimate_hairline_point(
    bgr_image: np.ndarray,
    landmarks: list[dict[str, Any]],
    axes: FaceAxes,
    width: int,
    height: int,
) -> tuple[float, float]:
    """
    Estimate trichion in image pixels, searching along the face midline.

    Walks forehead-up in face coordinates (against +v) so tilt is respected.
    Falls back to a geometric equal-thirds estimate when color is ambiguous.

    Returns:
        (x, y) hairline point in image pixels.
    """
    brow = (landmarks[9]["x"] * width, landmarks[9]["y"] * height)
    mesh_top = (landmarks[10]["x"] * width, landmarks[10]["y"] * height)
    nose = (landmarks[2]["x"] * width, landmarks[2]["y"] * height)
    chin = (landmarks[152]["x"] * width, landmarks[152]["y"] * height)

    _, brow_v = axes.to_face(*brow)
    _, mesh_v = axes.to_face(*mesh_top)
    _, nose_v = axes.to_face(*nose)
    _, chin_v = axes.to_face(*chin)

    middle = max(nose_v - brow_v, 1.0)
    # FaceIQ trichion ≈ upper third ~30–33% of hair→chin ≈ ~1.0× midface length above brow.
    geometric_v = brow_v - 1.00 * middle
    # MediaPipe landmark 10 sits mid-forehead — only a mild lift above it.
    mesh_lift_v = mesh_v - max(0.12 * middle, 4.0)

    mask = _skin_mask(bgr_image)
    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

    # Search from brow toward forehead along midline and nearby columns.
    search_start_v = brow_v - 2.0
    search_end_v = geometric_v - 0.35 * middle
    step = -1.5
    u_offsets = [-0.14 * middle, -0.07 * middle, 0.0, 0.07 * middle, 0.14 * middle]

    candidates: list[float] = []
    for u_off in u_offsets:
        skin_run = 0
        found_v = None
        v = search_start_v
        while v > search_end_v:
            x, y = axes.to_image(u_off, v)
            xi, yi = int(round(x)), int(round(y))
            if xi < 2 or yi < 2 or xi >= width - 2 or yi >= height - 2:
                v += step
                continue
            skin_ratio = float(np.mean(mask[yi - 1 : yi + 2, xi - 1 : xi + 2] > 0))
            if skin_ratio > 0.55:
                skin_run += 1
                v += step
                continue
            if skin_run >= 2:
                ref_x, ref_y = axes.to_image(u_off, brow_v)
                rxi, ryi = int(np.clip(ref_x, 2, width - 3)), int(np.clip(ref_y, 2, height - 3))
                ref = float(np.mean(gray[ryi - 1 : ryi + 2, rxi - 1 : rxi + 2]))
                local = float(np.mean(gray[yi - 1 : yi + 2, xi - 1 : xi + 2]))
                if local <= ref * 0.93 or skin_ratio < 0.35:
                    found_v = float(v)
                    break
            skin_run = 0
            v += step
        if found_v is not None:
            candidates.append(found_v)

    if candidates:
        hair_v = float(np.median(candidates))
    else:
        hair_v = geometric_v

    # Keep trichion near the geometric third — color search often climbs into hair
    # and overstates upper third (~40%+) vs FaceIQ (~30–33%).
    hair_v = float(np.clip(hair_v, geometric_v - 0.12 * middle, geometric_v + 0.18 * middle))
    hair_v = min(hair_v, mesh_lift_v)
    hair_v = min(hair_v, chin_v - 1.0)
    return axes.to_image(0.0, hair_v)


def estimate_pose(landmarks: list[dict[str, Any]], roll_deg: float = 0.0) -> dict[str, Any]:
    """
    Classify face orientation as frontal, three-quarter, or profile.

    Also reports head roll so the UI can show tilt awareness.
    """
    nose = landmarks[1]
    left_cheek = landmarks[234]
    right_cheek = landmarks[454]
    left_eye = landmarks[33]
    right_eye = landmarks[263]

    left_dist = abs(nose["x"] - left_cheek["x"])
    right_dist = abs(right_cheek["x"] - nose["x"])
    side_ratio = min(left_dist, right_dist) / max(left_dist, right_dist, 1e-6)

    face_width = abs(right_cheek["x"] - left_cheek["x"])
    center_x = (left_cheek["x"] + right_cheek["x"]) / 2.0
    yaw = (nose["x"] - center_x) / max(face_width, 1e-6)

    eye_span = abs(right_eye["x"] - left_eye["x"])
    eye_width_ratio = eye_span / max(face_width, 1e-6)

    abs_roll = abs(roll_deg)

    # True side profiles often keep a middling side_ratio in MediaPipe while
    # yaw explodes and eye span collapses — treat those as profile.
    if (
        side_ratio < 0.42
        or eye_width_ratio < 0.34
        or abs(yaw) > 0.55
    ):
        pose = "profile"
        label = "Профиль"
        confidence = float(
            np.clip(max(1.0 - side_ratio, abs(yaw) / 2.0, 1.0 - eye_width_ratio), 0.55, 0.98)
        )
    elif side_ratio < 0.68 or abs(yaw) > 0.12:
        pose = "three_quarter"
        label = "Три четверти"
        confidence = float(np.clip(0.55 + (0.68 - side_ratio), 0.5, 0.9))
    else:
        pose = "frontal"
        label = "Анфас"
        confidence = float(np.clip(side_ratio, 0.7, 0.99))

    if abs_roll >= 4.0 and pose == "frontal":
        label = f"Анфас · наклон {abs_roll:.0f}°"
    elif abs_roll >= 4.0:
        label = f"{label} · наклон {abs_roll:.0f}°"

    return {
        "pose": pose,
        "pose_label": label,
        "yaw": float(yaw),
        "roll_deg": float(roll_deg),
        "side_ratio": float(side_ratio),
        "confidence": confidence,
    }

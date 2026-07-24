"""MediaPipe Face Landmarker detection service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision


# Common MediaPipe Face Mesh / Face Landmarker indices used for geometry.
LANDMARK = {
    "forehead": 10,
    "glabella": 9,
    "nose_bridge": 6,
    "nose_tip": 1,
    "nose_bottom": 2,
    "chin": 152,
    "left_cheek": 234,
    "right_cheek": 454,
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "left_eye_top": 159,
    "left_eye_bottom": 145,
    "right_eye_top": 386,
    "right_eye_bottom": 374,
    "left_nostril": 98,
    "right_nostril": 327,
    "mouth_left": 61,
    "mouth_right": 291,
    "upper_lip": 13,
    "lower_lip": 14,
    "philtrum": 0,
    "jaw_left": 172,
    "jaw_right": 397,
    "left_temple": 127,
    "right_temple": 356,
}


def _bbox_area(face: Any) -> float:
    xs = [float(p.x) for p in face]
    ys = [float(p.y) for p in face]
    return max(max(xs) - min(xs), 0.0) * max(max(ys) - min(ys), 0.0)


def _pick_best_face(faces: list[Any]) -> Any:
    """Prefer the largest detected face (profile shots can yield weak extras)."""
    return max(faces, key=_bbox_area)


def _clahe_bgr(bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
    lightness, a, b = cv2.split(lab)
    lightness = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lightness)
    return cv2.cvtColor(cv2.merge([lightness, a, b]), cv2.COLOR_LAB2BGR)


class FaceLandmarkDetector:
    """
    Detect 478 facial landmarks using MediaPipe Face Landmarker.

    Extreme profile photos often fail at default confidence / num_faces=1.
    We keep a sensitive landmarker and retry with light preprocessing.
    """

    def __init__(self, model_path: str | Path | None = None) -> None:
        """
        Initialize the Face Landmarker.

        Args:
            model_path: Optional path to face_landmarker.task. Defaults to backend/models/.
        """
        root = Path(__file__).resolve().parents[2]
        path = Path(model_path) if model_path else root / "models" / "face_landmarker.task"
        if not path.exists():
            raise FileNotFoundError(
                f"Face landmarker model not found at {path}. "
                "Download face_landmarker.task into backend/models/."
            )

        base_options = mp_python.BaseOptions(model_asset_path=str(path))
        # num_faces=2 + very low confidence is required for many true side
        # profiles (num_faces=1 returns empty even at conf=0.05).
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=2,
            min_face_detection_confidence=0.05,
            min_face_presence_confidence=0.05,
            min_tracking_confidence=0.1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

    def _detect_raw(self, bgr_image: np.ndarray) -> list[Any] | None:
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)
        faces = list(result.face_landmarks or [])
        return faces or None

    def detect(self, bgr_image: np.ndarray) -> list[dict[str, Any]]:
        """
        Detect facial landmarks in a BGR image.

        Args:
            bgr_image: OpenCV BGR image.

        Returns:
            List of landmark dicts with keys: index, x, y, z (normalized 0-1).

        Raises:
            ValueError: When no face is detected.
        """
        h0, w0 = bgr_image.shape[:2]

        # meta: flip (bool), crop origin/size in original pixels (None = full frame)
        attempts: list[tuple[str, np.ndarray, bool, tuple[int, int, int, int] | None]] = [
            ("orig", bgr_image, False, None),
            ("clahe", _clahe_bgr(bgr_image), False, None),
            ("flip", cv2.flip(bgr_image, 1), True, None),
            ("clahe_flip", cv2.flip(_clahe_bgr(bgr_image), 1), True, None),
        ]

        if max(h0, w0) < 1400:
            up = cv2.resize(bgr_image, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            up_c = cv2.resize(
                _clahe_bgr(bgr_image), None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC
            )
            # Uniform scale keeps normalized coords identical to original.
            attempts.extend(
                [
                    ("up2", up, False, None),
                    ("up2_clahe", up_c, False, None),
                    ("up2_flip", cv2.flip(up, 1), True, None),
                ]
            )

        y0, y1 = int(h0 * 0.04), int(h0 * 0.96)
        x0, x1 = int(w0 * 0.12), int(w0 * 0.92)
        if y1 > y0 + 40 and x1 > x0 + 40:
            crop = bgr_image[y0:y1, x0:x1]
            crop_box = (x0, y0, crop.shape[1], crop.shape[0])
            attempts.append(("crop", crop, False, crop_box))
            attempts.append(("crop_clahe", _clahe_bgr(crop), False, crop_box))

        faces = None
        flipped = False
        crop_box: tuple[int, int, int, int] | None = None
        for _name, variant, is_flip, box in attempts:
            found = self._detect_raw(variant)
            if found:
                faces = found
                flipped = is_flip
                crop_box = box
                break

        if not faces:
            raise ValueError(
                "Лицо не найдено. Для профиля нужен чёткий боковой ракурс "
                "(нос/подбородок/лоб в кадре, без сильного блюра). "
                "Попробуйте другой кадр или чуть ближе обрезать лицо."
            )

        face = _pick_best_face(faces)
        landmarks: list[dict[str, Any]] = []
        for index, point in enumerate(face):
            vx = float(point.x)
            vy = float(point.y)
            if flipped:
                vx = 1.0 - vx

            if crop_box is None:
                ox, oy = vx, vy
            else:
                cx0, cy0, cw, ch = crop_box
                ox = (cx0 + vx * cw) / max(w0, 1)
                oy = (cy0 + vy * ch) / max(h0, 1)

            landmarks.append(
                {
                    "index": index,
                    "x": float(ox),
                    "y": float(oy),
                    "z": float(point.z),
                }
            )
        return landmarks

    def point(
        self,
        landmarks: list[dict[str, Any]],
        key: str,
        width: int,
        height: int,
    ) -> tuple[float, float]:
        """
        Return a landmark in pixel coordinates.

        Args:
            landmarks: Detected landmark list.
            key: Named key from LANDMARK.
            width: Image width in pixels.
            height: Image height in pixels.

        Returns:
            (x, y) pixel coordinates.
        """
        lm = landmarks[LANDMARK[key]]
        return lm["x"] * width, lm["y"] * height

    def close(self) -> None:
        """Release MediaPipe resources."""
        self._landmarker.close()

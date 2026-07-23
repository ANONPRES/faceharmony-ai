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


class FaceLandmarkDetector:
    """
    Detect 478 facial landmarks using MediaPipe Face Landmarker.

    The detector is created once and reused across requests for performance.
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
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_faces=1,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._landmarker = vision.FaceLandmarker.create_from_options(options)

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
        rgb = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect(mp_image)

        if not result.face_landmarks:
            raise ValueError(
                "Лицо не найдено. Загрузите чёткое фото с хорошим освещением "
                "(анфас или профиль)."
            )

        face = result.face_landmarks[0]
        landmarks: list[dict[str, Any]] = []
        for index, point in enumerate(face):
            landmarks.append(
                {
                    "index": index,
                    "x": float(point.x),
                    "y": float(point.y),
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

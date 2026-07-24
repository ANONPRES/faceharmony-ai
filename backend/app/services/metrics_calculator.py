"""Facial geometry metrics in a roll-aware face coordinate frame."""

from __future__ import annotations

from typing import Any

import numpy as np

from .geometry import FaceAxes, build_face_axes, estimate_hairline_point, estimate_pose
from .landmark_detector import LANDMARK
from .scoring import combine_scores, distance as _distance, midpoint as _midpoint, power_mean, soft_score
from . import aesthetics as aesthetic_features
from .detailed_measurements import build_detailed_measurements

PHI = 1.6180339887

# Aesthetic-feature-first weights (анфас).
FRONTAL_WEIGHTS = {
    "cheekbones": 0.16,
    "eye_cut": 0.15,
    "nose": 0.11,
    "face_shape": 0.12,
    "jaw": 0.11,
    "symmetry": 0.09,
    "chin": 0.09,
    "lips": 0.05,
    "midface": 0.05,
    "eyes": 0.04,
    "brow": 0.03,
}

PROFILE_WEIGHTS = {
    "nose": 0.20,
    "chin": 0.16,
    "jaw": 0.16,
    "cheekbones": 0.12,
    "face_shape": 0.10,
    "midface": 0.10,
    "lips": 0.08,
    "golden_ratio": 0.08,
}

THREE_QUARTER_WEIGHTS = {
    "cheekbones": 0.14,
    "nose": 0.12,
    "jaw": 0.12,
    "face_shape": 0.12,
    "eye_cut": 0.10,
    "chin": 0.10,
    "symmetry": 0.08,
    "midface": 0.08,
    "lips": 0.06,
    "eyes": 0.04,
    "brow": 0.04,
}



def _norm_point(x: float, y: float, width: int, height: int) -> dict[str, float]:
    """Normalize a pixel point to 0–1 image coordinates."""
    return {"x": float(x / max(width, 1)), "y": float(y / max(height, 1))}


def _norm_segment(
    p1: tuple[float, float],
    p2: tuple[float, float],
    width: int,
    height: int,
    label: str = "",
    sid: str = "",
) -> dict[str, Any]:
    """Build a normalized line segment payload for the frontend overlay."""
    return {
        "id": sid,
        "label": label,
        "x1": float(p1[0] / max(width, 1)),
        "y1": float(p1[1] / max(height, 1)),
        "x2": float(p2[0] / max(width, 1)),
        "y2": float(p2[1] / max(height, 1)),
    }


# True profile photo feature weights (silhouette metrics only).
TRUE_PROFILE_WEIGHTS = {
    "nose": 0.28,
    "chin": 0.22,
    "jaw": 0.22,
    "face_shape": 0.16,
    "midface": 0.08,
    "lips": 0.04,
}


class MetricsCalculator:
    """
    Compute facial proportion metrics in a face-aligned frame.

    Head roll/tilt is measured from the eye line so symmetry, thirds, and
    overlay guides follow the face instead of the photo edges.
    """

    def __init__(
        self,
        landmarks: list[dict[str, Any]],
        width: int,
        height: int,
        bgr_image: np.ndarray | None = None,
        gender: str = "male",
    ) -> None:
        """Initialize axes, pose, gender, and corrected hairline."""
        self.landmarks = landmarks
        self.width = width
        self.height = height
        self.bgr_image = bgr_image
        self.gender = gender if gender in ("male", "female") else "male"
        self.axes: FaceAxes = build_face_axes(landmarks, width, height)
        self.pose_info = estimate_pose(landmarks, roll_deg=self.axes.roll_deg)
        if bgr_image is not None:
            self.hairline_xy = estimate_hairline_point(
                bgr_image, landmarks, self.axes, width, height
            )
        else:
            brow = self.px("glabella")
            nose = self.px("nose_bottom")
            _, brow_v = self.axes.to_face(*brow)
            _, nose_v = self.axes.to_face(*nose)
            self.hairline_xy = self.axes.to_image(0.0, brow_v - max(nose_v - brow_v, 1.0))

    def px(self, key: str) -> tuple[float, float]:
        """Named landmark in image pixels."""
        lm = self.landmarks[LANDMARK[key]]
        return lm["x"] * self.width, lm["y"] * self.height

    def px_index(self, index: int) -> tuple[float, float]:
        """Indexed landmark in image pixels."""
        lm = self.landmarks[index]
        return lm["x"] * self.width, lm["y"] * self.height

    def face(self, key: str) -> tuple[float, float]:
        """Named landmark in face-local (u, v)."""
        return self.axes.to_face(*self.px(key))

    def face_index(self, index: int) -> tuple[float, float]:
        """Indexed landmark in face-local (u, v)."""
        return self.axes.to_face(*self.px_index(index))

    def calculate_all(self, *, as_profile_view: bool = False) -> dict[str, Any]:
        """Compute all metrics, pose-aware overall, and rotated overlay guides.

        Args:
            as_profile_view: When True, treat this calculator as the dedicated
                profile photo and score via silhouette metrics (not frontal widths).
        """
        from .profile_silhouette import build_profile_silhouette_scores

        # Dedicated profile upload: force profile pose so feature branches
        # and measurement gates use the profile path.
        if as_profile_view:
            self.pose_info = {
                **self.pose_info,
                "pose": "profile",
                "pose_label": "Профиль",
                "confidence": max(float(self.pose_info.get("confidence", 0.7)), 0.75),
            }

        if as_profile_view:
            sil = build_profile_silhouette_scores(self)
            # Keep unused frontal keys mild/neutral so appeal doesn't explode.
            metric_map = {
                "cheekbones": {
                    "score": 70.0,
                    "label": "Скулы",
                    "explanation": "На чистом профиле ширина скул не оценивается.",
                    "ratio": None,
                },
                "eye_cut": {
                    "score": 70.0,
                    "label": "Вырез глаз",
                    "explanation": "На чистом профиле canthal tilt малоинформативен.",
                    "ratio": None,
                },
                "symmetry": {
                    "score": 70.0,
                    "label": "Симметрия",
                    "explanation": "Симметрия по одному боку профиля не считается.",
                    "ratio": None,
                },
                "nose": sil["nose"],
                "face_shape": sil["face_shape"],
                "jaw": sil["jaw"],
                "chin": sil["chin"],
                "lips": sil["lips"],
                "midface": sil["midface"],
                "eyes": {
                    "score": 70.0,
                    "label": "Глаза",
                    "explanation": "Межглазье на профиле не оценивается.",
                    "ratio": None,
                },
                "brow": {
                    "score": 70.0,
                    "label": "Брови",
                    "explanation": "На профиле брови оцениваются ограниченно.",
                    "ratio": None,
                },
                "golden_ratio": {
                    "score": 70.0,
                    "label": "Золотое сечение",
                    "explanation": "На профиле используется контурная гармония вместо φ-ширин.",
                    "ratio": None,
                },
                "thirds": {
                    "score": 72.0,
                    "label": "Трети (профиль)",
                    "explanation": "Вертикальные трети на профиле оцениваются по силуэту отдельно.",
                    "ratio": None,
                },
                "fifths": {
                    "score": 70.0,
                    "label": "Пятые",
                    "explanation": "Горизонтальные пятые на профиле не считаются.",
                    "ratio": None,
                },
                "face_ratio": {
                    "score": float(sil["face_shape"]["score"]),
                    "label": "Пропорции",
                    "explanation": "На профиле ≈ контурная гармония силуэта.",
                    "ratio": sil["face_shape"].get("ratio"),
                },
            }
        else:
            symmetry = self.symmetry_score()
            thirds = self.facial_thirds_score()
            fifths = self.facial_fifths_score()
            eyes = self.eye_spacing_score()
            nose = aesthetic_features.nose_aesthetics_score(self)
            lips = self.lip_score()
            jaw = self.jaw_score()
            chin = self.chin_score()
            face_ratio = self.face_width_height_score()
            golden = self.golden_ratio_score()
            cheekbones = aesthetic_features.cheekbone_score(self)
            eye_cut = aesthetic_features.eye_cut_score(self)
            face_shape = aesthetic_features.face_shape_score(self)
            brow = aesthetic_features.brow_score(self)
            midface = aesthetic_features.midface_score(self)

            metric_map = {
                "cheekbones": cheekbones,
                "eye_cut": eye_cut,
                "nose": nose,
                "face_shape": face_shape,
                "jaw": jaw,
                "symmetry": symmetry,
                "chin": chin,
                "lips": lips,
                "midface": midface,
                "eyes": eyes,
                "brow": brow,
                "golden_ratio": golden,
                "thirds": thirds,
                "fifths": fifths,
                "face_ratio": face_ratio,
            }

        score_values = {key: float(metric_map[key]["score"]) for key in metric_map}

        # FaceIQ mode: Harmony only (A/D/F / Overall Coming Soon on FaceIQ).
        # Build frontal measurements first, then Harmony from ratio scores.
        measurements = build_detailed_measurements(self, gender=self.gender)
        if as_profile_view:
            from .profile_silhouette import build_profile_ceph_measurements

            ceph_rows = build_profile_ceph_measurements(self)
            if ceph_rows:
                frontal_only = [m for m in measurements if m.get("view") != "profile"]
                measurements = frontal_only + ceph_rows

        from .faceiq_harmony import harmony_from_measurements

        harmony_info = harmony_from_measurements(measurements)
        harmony_100 = float(harmony_info["score"])
        harmony_10 = float(harmony_info["score_10"])

        pillars = {
            "harmony": harmony_100,
            # FaceIQ: Coming Soon — keep keys for API compat, UI hides them.
            "angularity": None,
            "dimorphism": None,
            "features": None,
        }
        metric_map["harmony"] = {
            "score": round(harmony_100, 1),
            "label": "Harmony",
            "explanation": harmony_info["explanation"],
            "ratio": harmony_10,
        }
        score_values["harmony"] = harmony_100

        # Appeal temporarily mirrors Harmony /10 (FaceIQ has no separate Appeal).
        metric_map["appeal"] = {
            "score": round(harmony_100, 1),
            "label": "Harmony",
            "explanation": "Пока = FaceIQ Harmony (столпы A/D/F скоро).",
            "ratio": harmony_10,
        }
        score_values["appeal"] = harmony_100

        frontal_overall = harmony_100
        three_q_overall = harmony_100 * 0.98

        pose = self.pose_info["pose"]
        if as_profile_view:
            raw_profile = self._weighted_overall(score_values, TRUE_PROFILE_WEIGHTS)
            profile_overall = self._calibrate_overall(raw_profile, score_values, "profile")
            overall = profile_overall
            profile_score: float | None = float(np.clip(round(profile_overall, 1), 0.0, 100.0))
        else:
            profile_score = None
            if pose == "three_quarter":
                overall = three_q_overall
            elif pose == "profile":
                raw_profile = self._weighted_overall(score_values, TRUE_PROFILE_WEIGHTS)
                profile_overall = self._calibrate_overall(raw_profile, score_values, "profile")
                overall = profile_overall
                profile_score = float(np.clip(round(profile_overall, 1), 0.0, 100.0))
            else:
                overall = frontal_overall

        return {
            "overall": float(np.clip(round(overall, 1), 0.0, 100.0)),
            "frontal_score": float(np.clip(round(frontal_overall, 1), 0.0, 100.0)),
            "profile_score": profile_score,
            "pose": pose,
            "pose_label": self.pose_info["pose_label"],
            "pose_confidence": float(round(self.pose_info["confidence"], 3)),
            "roll_deg": float(round(self.axes.roll_deg, 2)),
            "gender": self.gender,
            "appeal": float(round(harmony_100, 1)),
            "appeal_10": float(harmony_10),
            "harmony": float(round(harmony_100, 1)),
            "angularity": None,
            "dimorphism": None,
            "features_pillar": None,
            "pillars": {
                "harmony": float(round(harmony_100, 1)),
                "angularity": None,
                "dimorphism": None,
                "features": None,
            },
            "harmony_breakdown": harmony_info.get("breakdown", {}),
            "scores": {k: float(round(v, 1)) for k, v in score_values.items()},
            "metrics": metric_map,
            "measurements": measurements,
            "overlay": self.build_overlay(),
        }

    def _weighted_overall(self, scores: dict[str, float], weights: dict[str, float]) -> float:
        """Pose-specific overall with weak-metric penalty (power mean)."""
        parts = [(scores[k], w) for k, w in weights.items() if k in scores]
        # Profile silhouette metrics are noisier — less brutal power mean.
        p = 0.72 if weights is TRUE_PROFILE_WEIGHTS else 0.50
        return power_mean(parts, p=p)

    def _calibrate_overall(self, raw: float, scores: dict[str, float], pose: str) -> float:
        """
        Profile / legacy path: weak-link blend + rarity curve (no 72→78 squash).

        Target bands:
        - uneven / average → mid 50s–65
        - solid → ~68–78
        - consistently elite → ~85–94
        """
        from .pillars import rarity_curve

        core_keys = (
            ["cheekbones", "eye_cut", "nose", "face_shape", "jaw", "symmetry"]
            if pose != "profile"
            else ["nose", "chin", "jaw", "face_shape", "midface"]
        )
        core = [scores[k] for k in core_keys if k in scores]
        if not core:
            return rarity_curve(raw)

        ordered = sorted(core)
        weak = float(np.mean(ordered[:2]))
        mean_core = float(np.mean(core))
        spread = float(np.std(core))
        top = float(np.mean(ordered[-2:]))

        blended = 0.40 * raw + 0.40 * weak + 0.20 * top
        if spread >= 14:
            blended -= min(8.0, (spread - 14.0) * 0.5)
        if weak >= 92 and mean_core >= 93 and spread <= 4.5:
            blended += 2.0
        elif weak >= 88 and mean_core >= 90:
            blended += 1.0

        return rarity_curve(blended)


    def symmetry_score(self) -> dict[str, Any]:
        """Left/right balance vs face midline in roll-aware coordinates."""
        pairs = [
            ("left_cheek", "right_cheek"),
            ("left_eye_outer", "right_eye_outer"),
            ("left_eye_inner", "right_eye_inner"),
            ("mouth_left", "mouth_right"),
            ("left_nostril", "right_nostril"),
            ("jaw_left", "jaw_right"),
            ("left_temple", "right_temple"),
        ]
        pose = self.pose_info["pose"]
        reliability = 1.0 if pose == "frontal" else (0.6 if pose == "three_quarter" else 0.3)

        diffs: list[float] = []
        for left_key, right_key in pairs:
            lu, lv = self.face(left_key)
            ru, rv = self.face(right_key)
            # Mirror across face midline u=0.
            if max(abs(lu), abs(ru)) < 1e-3:
                continue
            diffs.append(abs(abs(lu) - abs(ru)) / max(abs(lu), abs(ru)))
            # Along-face vertical alignment of mirrored pair.
            face_h = abs(self.face("chin")[1] - self.axes.to_face(*self.hairline_xy)[1])
            diffs.append(abs(lv - rv) / max(face_h, 1.0))

        mean_diff = float(np.mean(diffs)) if diffs else 1.0
        base = float(np.clip(100.0 * np.exp(-mean_diff * 3.8), 0.0, 100.0))
        score = reliability * base + (1.0 - reliability) * 74.0

        return {
            "score": round(score, 1),
            "label": "Симметрия",
            "explanation": (
                "Симметрия считается относительно оси лица с учётом наклона головы "
                f"({abs(self.axes.roll_deg):.0f}°)."
            ),
            "ratio": round(1.0 - mean_diff, 4),
        }

    def facial_thirds_score(self) -> dict[str, Any]:
        """Vertical thirds along the face axis (hairline → chin)."""
        _, hair_v = self.axes.to_face(*self.hairline_xy)
        _, brow_v = self.face("glabella")
        _, nose_v = self.face("nose_bottom")
        _, chin_v = self.face("chin")

        upper = abs(brow_v - hair_v)
        middle = abs(nose_v - brow_v)
        lower = abs(chin_v - nose_v)
        total = upper + middle + lower
        ratios = (
            (1 / 3, 1 / 3, 1 / 3)
            if total < 1e-3
            else (upper / total, middle / total, lower / total)
        )
        score = float(np.mean([soft_score(r, 1 / 3, 0.055) for r in ratios]))
        return {
            "score": round(score, 1),
            "label": "Лицевые трети",
            "explanation": (
                "Трети измеряются вдоль оси лица (с учётом наклона), "
                "от линии роста волос до подбородка."
            ),
            "ratio": round(float(np.std(ratios)), 4),
        }

    def facial_fifths_score(self) -> dict[str, Any]:
        """Horizontal fifths along the eye-line axis."""
        if self.pose_info["pose"] == "profile":
            return {
                "score": 72.0,
                "label": "Лицевые пятые",
                "explanation": "На профиле горизонтальные пятые малоинформативны.",
                "ratio": None,
            }

        left_u, _ = self.face("left_temple")
        right_u, _ = self.face("right_temple")
        le_o, _ = self.face("left_eye_outer")
        le_i, _ = self.face("left_eye_inner")
        re_i, _ = self.face("right_eye_inner")
        re_o, _ = self.face("right_eye_outer")

        face_w = abs(right_u - left_u)
        if face_w < 1e-3:
            return {
                "score": 50.0,
                "label": "Лицевые пятые",
                "explanation": "Не удалось измерить горизонтальные пятые.",
                "ratio": None,
            }

        # Sort so left→right in face u.
        order = sorted([left_u, le_o, le_i, re_i, re_o, right_u])
        segments = [order[i + 1] - order[i] for i in range(5)]
        ratios = [s / face_w for s in segments]
        score = float(np.mean([soft_score(r, 0.2, 0.045) for r in ratios]))
        if self.pose_info["pose"] == "three_quarter":
            score = 0.7 * score + 0.3 * 76.0
        return {
            "score": round(score, 1),
            "label": "Лицевые пятые",
            "explanation": "Пятые считаются вдоль линии глаз, а не горизонтали кадра.",
            "ratio": round(float(np.std(ratios)), 4),
        }

    def eye_spacing_score(self) -> dict[str, Any]:
        """Intercanthal distance vs average eye width (face-aligned)."""
        if self.pose_info["pose"] == "profile":
            return {
                "score": 72.0,
                "label": "Расстояние между глазами",
                "explanation": "На профиле межглазье оценивается ограниченно.",
                "ratio": None,
            }
        le_o = self.px("left_eye_outer")
        le_i = self.px("left_eye_inner")
        re_i = self.px("right_eye_inner")
        re_o = self.px("right_eye_outer")
        avg_eye = (_distance(le_o, le_i) + _distance(re_o, re_i)) / 2.0
        spacing = _distance(le_i, re_i)
        ratio = spacing / avg_eye if avg_eye > 1e-3 else 1.0
        # Real faces often sit near ~1.1–1.3; classic "1.0" is too harsh on photos.
        score = soft_score(ratio, 1.15, 0.32)
        return {
            "score": round(score, 1),
            "label": "Расстояние между глазами",
            "explanation": "Межглазье сравнивается со средней шириной глаза (мягкий ориентир ≈ 1.1–1.2).",
            "ratio": round(ratio, 4),
        }

    def nose_score(self) -> dict[str, Any]:
        """Nose proportions using face-aligned widths where helpful."""
        nose_width = _distance(self.px("left_nostril"), self.px("right_nostril"))
        mouth_width = _distance(self.px("mouth_left"), self.px("mouth_right"))
        face_width = abs(self.face("right_cheek")[0] - self.face("left_cheek")[0])
        nose_length = abs(self.face("nose_bottom")[1] - self.face("nose_bridge")[1])

        if self.pose_info["pose"] == "profile":
            tip_z = abs(self.landmarks[LANDMARK["nose_tip"]]["z"])
            cheek_z = abs(
                (
                    self.landmarks[LANDMARK["left_cheek"]]["z"]
                    + self.landmarks[LANDMARK["right_cheek"]]["z"]
                )
                / 2.0
            )
            projection = tip_z / max(cheek_z, 1e-4)
            length_ratio = nose_length / max(face_width, 1e-3)
            score = combine_scores(
                [
                    (soft_score(projection, 1.35, 0.45), 0.55),
                    (soft_score(length_ratio, 0.34, 0.10), 0.45),
                ]
            )
            return {
                "score": round(score, 1),
                "label": "Нос (профиль)",
                "explanation": "На профиле оцениваются проекция и длина носа.",
                "ratio": round(projection, 4),
            }

        width_vs_mouth = nose_width / mouth_width if mouth_width > 1e-3 else 0.7
        width_vs_face = nose_width / face_width if face_width > 1e-3 else 0.25
        length_vs_face = nose_length / face_width if face_width > 1e-3 else 0.35
        score = combine_scores(
            [
                (soft_score(width_vs_mouth, 0.70, 0.15), 0.4),
                (soft_score(width_vs_face, 0.24, 0.07), 0.3),
                (soft_score(length_vs_face, 0.34, 0.10), 0.3),
            ]
        )
        return {
            "score": round(score, 1),
            "label": "Пропорции носа",
            "explanation": "Ширина и длина носа относительно рта и лица.",
            "ratio": round(width_vs_mouth, 4),
        }

    def lip_score(self) -> dict[str, Any]:
        """Lip vermilion ratio and mouth-to-face width."""
        upper_height = abs(self.face_index(0)[1] - self.face_index(13)[1])
        lower_height = abs(self.face_index(17)[1] - self.face_index(14)[1])
        lip_ratio = 1.25 if upper_height < 1e-3 else max(lower_height / upper_height, 0.35)
        mouth_width = _distance(self.px("mouth_left"), self.px("mouth_right"))
        face_width = abs(self.face("right_cheek")[0] - self.face("left_cheek")[0])
        mouth_face = mouth_width / face_width if face_width > 1e-3 else 0.4
        # Women: fuller lower lip ideal; men: slightly flatter.
        lip_ideal = 1.65 if self.gender == "female" else 1.35
        score = combine_scores(
            [
                (soft_score(lip_ratio, lip_ideal, 0.45), 0.4),
                (soft_score(mouth_face, 0.40, 0.10), 0.6),
            ]
        )
        return {
            "score": round(score, 1),
            "label": "Пропорции губ",
            "explanation": "Высота губ и ширина рта относительно лица.",
            "ratio": round(lip_ratio, 4),
        }

    def jaw_score(self) -> dict[str, Any]:
        """Jaw width vs cheek width in face coordinates."""
        jaw_w = abs(self.face("jaw_right")[0] - self.face("jaw_left")[0])
        cheek_w = abs(self.face("right_cheek")[0] - self.face("left_cheek")[0])
        jaw_cheek = jaw_w / cheek_w if cheek_w > 1e-3 else 0.88
        mouth_v = (self.face("mouth_left")[1] + self.face("mouth_right")[1]) / 2.0
        jaw_depth = abs(self.face("chin")[1] - mouth_v) / max(cheek_w, 1e-3)

        if self.pose_info["pose"] == "profile":
            mand = abs(self.face("chin")[1] - self.face("nose_bottom")[1]) / max(cheek_w, 1e-3)
            return {
                "score": round(soft_score(mand, 0.55, 0.18), 1),
                "label": "Линия челюсти",
                "explanation": "На профиле оценивается геометрия нижней трети.",
                "ratio": round(mand, 4),
            }

        # Men: slightly tapered square jaw (~0.80–0.84); women: stronger V-taper.
        # Elite male meshes often sit ~0.80–0.82 — don't peak at 0.86.
        jaw_ideal = 0.81 if self.gender == "male" else 0.77
        score = combine_scores(
            [
                (soft_score(jaw_cheek, jaw_ideal, 0.08), 0.62),
                (soft_score(jaw_depth, 0.33, 0.09), 0.38),
            ]
        )
        return {
            "score": round(score, 1),
            "label": "Челюсть",
            "explanation": (
                f"Отношение ширины челюсти к скулам ≈ {jaw_cheek:.2f} "
                f"(идеал для {'мужчин' if self.gender == 'male' else 'женщин'} ≈ {jaw_ideal:.2f})."
            ),
            "ratio": round(jaw_cheek, 4),
        }

    def chin_score(self) -> dict[str, Any]:
        """Chin share, projection, taper, and midline centering in face frame."""
        _, hair_v = self.axes.to_face(*self.hairline_xy)
        _, brow_v = self.face("glabella")
        _, nose_v = self.face("nose_bottom")
        chin_u, chin_v = self.face("chin")
        mouth_v = (self.face("mouth_left")[1] + self.face("mouth_right")[1]) / 2.0

        # Tight crop often puts the estimated hairline above the frame →
        # inflated face_h and broken chin_share. Fall back to brow-based height.
        hair_y_img = self.hairline_xy[1]
        chin_y_img = self.px("chin")[1]
        tight = hair_y_img < 0.08 * self.height or chin_y_img > 0.92 * self.height
        if tight:
            face_h = abs(chin_v - brow_v) / 0.68  # brow→chin ≈ 68% of trichion→chin
        else:
            face_h = abs(chin_v - hair_v)

        chin_share = abs(chin_v - nose_v) / face_h if face_h > 1e-3 else 0.33
        # Longer menton (mouth→chin vs nose→mouth). Male models often ~2.2–3.0.
        projection = abs(chin_v - mouth_v) / max(abs(mouth_v - nose_v), 1e-3)
        proj_ideal = 2.45 if self.gender == "male" else 2.15
        proj_sigma = 0.65 if self.gender == "male" else 0.55
        cheek_w = abs(self.face("right_cheek")[0] - self.face("left_cheek")[0])
        chin_left = self.face_index(176) if len(self.landmarks) > 400 else (chin_u - cheek_w * 0.17, chin_v)
        chin_right = self.face_index(400) if len(self.landmarks) > 400 else (chin_u + cheek_w * 0.17, chin_v)
        chin_w = abs(chin_right[0] - chin_left[0])
        chin_taper = chin_w / max(cheek_w, 1e-3)
        center_pen = float(np.clip(100.0 * np.exp(-abs(chin_u) / max(cheek_w, 1e-3) * 10.0), 0, 100))

        score = combine_scores(
            [
                (soft_score(chin_share, 0.33, 0.08), 0.22),
                (soft_score(projection, proj_ideal, proj_sigma), 0.43),
                (soft_score(chin_taper, 0.34, 0.09), 0.20),
                (center_pen, 0.15),
            ]
        )
        return {
            "score": round(score, 1),
            "label": "Подбородок",
            "explanation": (
                "Доля нижней трети, проекция и сужение подбородка по оси лица."
                + (" Tight-crop: высота лица оценена от бровей." if tight else "")
            ),
            "ratio": round(projection, 4),
        }

    def face_width_height_score(self) -> dict[str, Any]:
        """Face width/height using hairline→chin along face axes."""
        width = abs(self.face("right_cheek")[0] - self.face("left_cheek")[0])
        _, hair_v = self.axes.to_face(*self.hairline_xy)
        height = abs(self.face("chin")[1] - hair_v)
        ratio = width / height if height > 1e-3 else 0.75
        return {
            "score": round(soft_score(ratio, 0.72, 0.10), 1),
            "label": "Ширина / высота лица",
            "explanation": "Соотношение ширины и высоты в системе координат лица.",
            "ratio": round(ratio, 4),
        }

    def golden_ratio_score(self) -> dict[str, Any]:
        """φ approximations along face axes."""
        _, hair_v = self.axes.to_face(*self.hairline_xy)
        _, nose_v = self.face("nose_bottom")
        _, chin_v = self.face("chin")
        face_h = abs(chin_v - hair_v)
        face_w = abs(self.face("right_cheek")[0] - self.face("left_cheek")[0])
        upper = abs(nose_v - hair_v)
        lower = abs(chin_v - nose_v)
        mouth_w = _distance(self.px("mouth_left"), self.px("mouth_right"))
        eye_span = _distance(self.px("left_eye_outer"), self.px("right_eye_outer"))

        scored: list[float] = []
        ratios: list[float] = []
        if lower > 1e-3:
            r = upper / lower
            ratios.append(r)
            scored.append(soft_score(r, PHI, 0.35))
        if face_w > 1e-3:
            r = face_h / face_w
            ratios.append(r)
            scored.append(soft_score(r, PHI, 0.40))
        if mouth_w > 1e-3 and self.pose_info["pose"] != "profile":
            r = eye_span / mouth_w
            ratios.append(r)
            scored.append(soft_score(r, PHI, 0.45))

        score = float(np.mean(scored)) if scored else 72.0
        return {
            "score": round(score, 1),
            "label": "Золотое сечение",
            "explanation": "Учебное сравнение длин с φ ≈ 1.618 в осях лица.",
            "ratio": round(float(np.mean(ratios)) if ratios else PHI, 4),
        }

    def build_overlay(self) -> dict[str, Any]:
        """
        Build overlay guides rotated to the face roll.

        All guides are line segments in normalized image coordinates so the
        frontend can draw tilt-aware geometry without guessing angles.
        """
        w, h = self.width, self.height
        axes = self.axes

        _, hair_v = axes.to_face(*self.hairline_xy)
        _, brow_v = self.face("glabella")
        _, nose_v = self.face("nose_bottom")
        _, chin_v = self.face("chin")
        left_u, _ = self.face("left_temple")
        right_u, _ = self.face("right_temple")
        cheek_span = abs(self.face("right_cheek")[0] - self.face("left_cheek")[0])
        half = max(cheek_span * 0.75, abs(right_u - left_u) * 0.55)

        # Symmetry / face-height axis.
        sym_a, sym_b = axes.line_along(0.0, hair_v - 0.05 * (chin_v - hair_v), chin_v + 0.02 * (chin_v - hair_v))
        symmetry_line = _norm_segment(sym_a, sym_b, w, h, "Ось лица", "symmetry")

        thirds_lines = []
        for v, label in [
            (hair_v, "Линия роста волос"),
            (brow_v, "Брови"),
            (nose_v, "Основание носа"),
            (chin_v, "Подбородок"),
        ]:
            a, b = axes.line_across(v, half)
            thirds_lines.append(_norm_segment(a, b, w, h, label, "thirds"))

        # Legacy thirds_y for older clients: project face-line midpoints.
        thirds_y = [
            ((a[1] + b[1]) / 2.0) / h
            for (a, b), _ in zip(
                [axes.line_across(v, half) for v in (hair_v, brow_v, nose_v, chin_v)],
                range(4),
            )
        ]

        face_left_u = min(left_u, right_u)
        face_right_u = max(left_u, right_u)
        span = face_right_u - face_left_u
        mid_v = (brow_v + nose_v) / 2.0
        fifths_lines = []
        fifths_x = []
        for i in range(6):
            u = face_left_u + span * i / 5.0
            a, b = axes.line_along(u, hair_v, chin_v)
            fifths_lines.append(_norm_segment(a, b, w, h, f"Пятая {i}", "fifths"))
            fifths_x.append(float(((a[0] + b[0]) / 2.0) / w))

        # Golden rectangle corners in face space, mapped to image.
        box_w = cheek_span
        box_h = box_w * PHI
        face_h = abs(chin_v - hair_v)
        if box_h > face_h * 1.15:
            box_h = face_h
            box_w = face_h / PHI
        cy_v = (hair_v + chin_v) / 2.0
        corners_face = [
            (-box_w / 2, cy_v - box_h / 2),
            (box_w / 2, cy_v - box_h / 2),
            (box_w / 2, cy_v + box_h / 2),
            (-box_w / 2, cy_v + box_h / 2),
        ]
        corners = [_norm_point(*axes.to_image(u, v), w, h) for u, v in corners_face]
        golden_boxes = [
            {
                "corners": corners,
                # Axis-aligned fallback bbox for older clients.
                "x": min(c["x"] for c in corners),
                "y": min(c["y"] for c in corners),
                "w": max(c["x"] for c in corners) - min(c["x"] for c in corners),
                "h": max(c["y"] for c in corners) - min(c["y"] for c in corners),
            }
        ]

        measurements = [
            _norm_segment(self.px("left_cheek"), self.px("right_cheek"), w, h, "Ширина лица", "face_width"),
            _norm_segment(self.hairline_xy, self.px("chin"), w, h, "Высота лица", "face_height"),
            _norm_segment(self.px("left_eye_inner"), self.px("right_eye_inner"), w, h, "Межглазье", "eye_spacing"),
            _norm_segment(self.px("left_nostril"), self.px("right_nostril"), w, h, "Ширина носа", "nose_width"),
            _norm_segment(self.px("mouth_left"), self.px("mouth_right"), w, h, "Ширина рта", "mouth_width"),
            _norm_segment(*axes.line_across(hair_v, half * 0.55), w, h, "Линия роста волос", "hairline"),
        ]

        # Midline_x legacy: average x of symmetry line.
        midline_x = (symmetry_line["x1"] + symmetry_line["x2"]) / 2.0

        return {
            "roll_deg": float(round(axes.roll_deg, 2)),
            "origin": _norm_point(axes.cx, axes.cy, w, h),
            "midline_x": float(midline_x),
            "thirds_y": [float(v) for v in thirds_y],
            "fifths_x": fifths_x,
            "symmetry_line": symmetry_line,
            "thirds_lines": thirds_lines,
            "fifths_lines": fifths_lines,
            "golden_boxes": golden_boxes,
            "measurements": measurements,
        }

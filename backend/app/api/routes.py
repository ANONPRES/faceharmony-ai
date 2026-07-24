"""FastAPI route handlers."""

from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.schemas import (
    AnalysisResponse,
    DetailedMeasurement,
    LandmarkPoint,
    MetricDetail,
    OverlayGuides,
)
from app.services.landmark_detector import FaceLandmarkDetector
from app.services.metrics_calculator import MetricsCalculator
from app.services.recommendations import generate_recommendations
from app.utils.image import decode_image, read_image_bytes

router = APIRouter()
_detector: FaceLandmarkDetector | None = None


def get_detector() -> FaceLandmarkDetector:
    """Lazy-load and cache the FaceLandmarkDetector singleton."""
    global _detector
    if _detector is None:
        _detector = FaceLandmarkDetector()
    return _detector


@router.get("/health")
def health() -> dict[str, str]:
    """Simple health check endpoint."""
    return {"status": "ok", "service": "FaceHarmony AI"}


def _detect_or_raise(image):
    try:
        return get_detector().detect(image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Детектор лиц не смог запуститься: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(
            status_code=500,
            detail=f"Анализ не удался: {exc}",
        ) from exc


def _to_response(
    *,
    result: dict,
    landmarks: list,
    width: int,
    height: int,
) -> AnalysisResponse:
    recommendations = generate_recommendations(
        result["metrics"],
        pose_label=result["pose_label"],
    )
    metric_models = {
        key: MetricDetail(**value) for key, value in result["metrics"].items()
    }
    landmark_models = [LandmarkPoint(**lm) for lm in landmarks]
    overlay = OverlayGuides(**result["overlay"])
    scores = result["scores"]

    return AnalysisResponse(
        overall=result["overall"],
        frontal_score=result["frontal_score"],
        profile_score=result.get("profile_score"),
        gender=result.get("gender", "male"),
        appeal=float(result.get("appeal") or scores.get("appeal") or 0.0),
        appeal_10=float(result.get("appeal_10") or 0.0),
        harmony=float(result.get("harmony") or scores.get("harmony") or 0.0),
        angularity=result.get("angularity"),
        dimorphism=result.get("dimorphism"),
        features_pillar=result.get("features_pillar"),
        pillars=dict(result.get("pillars") or {}),
        harmony_breakdown=dict(result.get("harmony_breakdown") or {}),
        pose=result["pose"],
        pose_label=result["pose_label"],
        pose_confidence=result["pose_confidence"],
        roll_deg=result.get("roll_deg", 0.0),
        cheekbones=float(
            scores.get("cheekbones")
            or result["metrics"].get("cheekbones", {}).get("score")
            or 0.0
        ),
        eye_cut=float(
            scores.get("eye_cut") or result["metrics"].get("eye_cut", {}).get("score") or 0.0
        ),
        face_shape=float(
            scores.get("face_shape")
            or result["metrics"].get("face_shape", {}).get("score")
            or 0.0
        ),
        midface=float(
            scores.get("midface") or result["metrics"].get("midface", {}).get("score") or 0.0
        ),
        brow=float(scores.get("brow") or result["metrics"].get("brow", {}).get("score") or 0.0),
        symmetry=scores["symmetry"],
        golden_ratio=scores["golden_ratio"],
        eyes=scores["eyes"],
        nose=scores["nose"],
        lips=scores["lips"],
        jaw=scores["jaw"],
        chin=scores["chin"],
        thirds=scores["thirds"],
        fifths=scores["fifths"],
        face_ratio=scores["face_ratio"],
        face_shape_label=(
            result["metrics"].get("face_shape", {}).get("face_shape")
            if isinstance(result["metrics"].get("face_shape"), dict)
            else None
        ),
        recommendations=recommendations,
        metrics=metric_models,
        measurements=[
            DetailedMeasurement(**m) for m in result.get("measurements", [])
        ],
        landmarks=landmark_models,
        overlay=overlay,
        image_width=width,
        image_height=height,
    )


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_face(
    file: UploadFile = File(..., description="Frontal (or primary) face photo"),
    gender: str = Form("male"),
    profile_file: UploadFile | None = File(None, description="Optional true profile photo"),
) -> AnalysisResponse:
    """
    Analyze face photo(s).

    - `gender`: male | female — shifts ideal bands / dimorphism scoring
    - `file`: required front (or main) photo
    - `profile_file`: optional dedicated profile photo for real profile_score
    """
    gender_norm = (gender or "male").strip().lower()
    if gender_norm not in ("male", "female"):
        raise HTTPException(status_code=422, detail="gender must be 'male' or 'female'")

    data = await read_image_bytes(file)
    image = decode_image(data)
    height, width = image.shape[:2]
    landmarks = _detect_or_raise(image)

    front_calc = MetricsCalculator(
        landmarks, width, height, bgr_image=image, gender=gender_norm
    )
    front = front_calc.calculate_all(as_profile_view=False)

    profile_score = None
    profile_measurements: list[dict] = []
    if profile_file is not None and profile_file.filename:
        pdata = await read_image_bytes(profile_file)
        pimage = decode_image(pdata)
        ph, pw = pimage.shape[:2]
        plandmarks = _detect_or_raise(pimage)
        profile_calc = MetricsCalculator(
            plandmarks, pw, ph, bgr_image=pimage, gender=gender_norm
        )
        profile = profile_calc.calculate_all(as_profile_view=True)
        profile_score = profile.get("profile_score")
        # Keep profile-view atomic measurements from the profile photo.
        profile_measurements = [
            m for m in profile.get("measurements", []) if m.get("view") == "profile"
        ]
        # Blend overall when both views exist.
        if profile_score is not None:
            front["overall"] = round(
                0.65 * float(front["frontal_score"]) + 0.35 * float(profile_score),
                1,
            )
            # Prefer higher appeal of the two for display.
            front["appeal"] = max(float(front.get("appeal") or 0), float(profile.get("appeal") or 0))
            front["appeal_10"] = round(front["appeal"] / 10.0, 1)
            if "appeal" in front["metrics"] and "appeal" in profile["metrics"]:
                if profile["metrics"]["appeal"]["score"] > front["metrics"]["appeal"]["score"]:
                    front["metrics"]["appeal"] = profile["metrics"]["appeal"]

    front["profile_score"] = profile_score
    if profile_measurements:
        # Merge: frontal measurements + profile-only from profile photo.
        front_meas = [m for m in front.get("measurements", []) if m.get("view") != "profile"]
        merged = front_meas + profile_measurements
        for i, m in enumerate(merged):
            m["order"] = i + 1
            m["total"] = len(merged)
        front["measurements"] = merged

    return _to_response(result=front, landmarks=landmarks, width=width, height=height)

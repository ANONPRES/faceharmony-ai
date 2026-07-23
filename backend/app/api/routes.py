"""FastAPI route handlers."""

from __future__ import annotations

from fastapi import APIRouter, File, HTTPException, UploadFile

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


@router.post("/analyze", response_model=AnalysisResponse)
async def analyze_face(file: UploadFile = File(...)) -> AnalysisResponse:
    """
    Analyze an uploaded face photo and return a facial harmony report.

    Args:
        file: Image upload (JPEG, PNG, WEBP, or BMP).

    Returns:
        Structured analysis JSON with scores, landmarks, overlay guides, and tips.
    """
    data = await read_image_bytes(file)
    image = decode_image(data)
    height, width = image.shape[:2]

    try:
        landmarks = get_detector().detect(image)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Детектор лиц не смог запуститься: {exc}",
        ) from exc
    except Exception as exc:  # noqa: BLE001 — surface unexpected analyze failures to client
        raise HTTPException(
            status_code=500,
            detail=f"Анализ не удался: {exc}",
        ) from exc

    calculator = MetricsCalculator(landmarks, width, height, bgr_image=image)
    result = calculator.calculate_all()
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
        profile_score=result["profile_score"],
        pose=result["pose"],
        pose_label=result["pose_label"],
        pose_confidence=result["pose_confidence"],
        roll_deg=result.get("roll_deg", 0.0),
        cheekbones=float(scores.get("cheekbones") or result["metrics"].get("cheekbones", {}).get("score") or 0.0),
        eye_cut=float(scores.get("eye_cut") or result["metrics"].get("eye_cut", {}).get("score") or 0.0),
        face_shape=float(scores.get("face_shape") or result["metrics"].get("face_shape", {}).get("score") or 0.0),
        midface=float(scores.get("midface") or result["metrics"].get("midface", {}).get("score") or 0.0),
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

"""Pydantic response schemas for FaceHarmony AI."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class MetricDetail(BaseModel):
    """Detailed score and explanation for a single facial metric."""

    score: float = Field(..., ge=0, le=100)
    label: str
    explanation: str
    ratio: float | None = None
    face_shape: str | None = None


class MeasurementPoint(BaseModel):
    """Editable overlay point for a detailed measurement."""

    id: str
    x: float
    y: float
    role: str = "anchor"
    landmark: int | None = None


class MeasurementSegment(BaseModel):
    """Line segment drawn for a detailed measurement."""

    x1: float
    y1: float
    x2: float
    y2: float
    style: str = "primary"
    label: str = ""


class DetailedMeasurement(BaseModel):
    """Atomic facial measurement with ideal band (competitor-style)."""

    id: str
    label: str
    category: str
    value: float
    unit: str
    display: str
    ideal_min: float
    ideal_max: float
    score: float = Field(..., ge=0, le=100)
    score_10: float = Field(..., ge=0, le=10)
    explanation: str
    scale_min: float
    scale_max: float
    soft_margin: float | None = None
    points: list[MeasurementPoint] = Field(default_factory=list)
    segments: list[MeasurementSegment] = Field(default_factory=list)
    formula: dict[str, Any] = Field(default_factory=dict)
    view: Literal["frontal", "profile", "three_quarter", "any"] = "frontal"
    order: int = 0
    total: int = 0


class LandmarkPoint(BaseModel):
    """Normalized facial landmark."""

    x: float
    y: float
    z: float = 0.0
    index: int


class OverlayGuides(BaseModel):
    """Guide geometry for the facial overlay (tilt-aware segments)."""

    roll_deg: float = 0.0
    origin: dict[str, float] | None = None
    midline_x: float
    thirds_y: list[float]
    fifths_x: list[float]
    symmetry_line: dict[str, Any] | None = None
    thirds_lines: list[dict[str, Any]] = Field(default_factory=list)
    fifths_lines: list[dict[str, Any]] = Field(default_factory=list)
    golden_boxes: list[dict[str, Any]]
    measurements: list[dict[str, Any]]


class AnalysisResponse(BaseModel):
    """Full facial harmony analysis payload."""

    overall: float
    frontal_score: float
    profile_score: float | None = None
    gender: Literal["male", "female"] = "male"
    appeal: float = 0.0
    appeal_10: float = 0.0
    harmony: float = 0.0
    angularity: float | None = None
    dimorphism: float | None = None
    features_pillar: float | None = None
    pillars: dict[str, float | None] = Field(default_factory=dict)
    harmony_breakdown: dict[str, float] = Field(default_factory=dict)
    pose: Literal["frontal", "three_quarter", "profile"]
    pose_label: str
    pose_confidence: float
    roll_deg: float = 0.0
    cheekbones: float = 0.0
    eye_cut: float = 0.0
    face_shape: float = 0.0
    midface: float = 0.0
    brow: float = 0.0
    symmetry: float
    golden_ratio: float
    eyes: float
    nose: float
    lips: float
    jaw: float
    chin: float
    thirds: float
    fifths: float
    face_ratio: float
    face_shape_label: str | None = None
    recommendations: list[str]
    metrics: dict[str, MetricDetail]
    measurements: list[DetailedMeasurement] = Field(default_factory=list)
    landmarks: list[LandmarkPoint]
    overlay: OverlayGuides
    image_width: int
    image_height: int
    disclaimer: str = (
        "FaceHarmony AI оценивает геометрию черт: скулы, вырез глаз, нос, форму лица, "
        "челюсть и баланс. Appeal — образовательный proxy, не вердикт о человеке."
    )

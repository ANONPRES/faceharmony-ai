/**
 * Shared TypeScript types for FaceHarmony AI analysis payloads.
 */

export interface MetricDetail {
  score: number;
  label: string;
  explanation: string;
  ratio: number | null;
  face_shape?: string | null;
}

export interface MeasurementPoint {
  id: string;
  x: number;
  y: number;
  role: string;
  landmark?: number | null;
}

export interface MeasurementSegment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  style?: string;
  label?: string;
}

export interface DetailedMeasurement {
  id: string;
  label: string;
  category: string;
  value: number;
  unit: string;
  display: string;
  ideal_min: number;
  ideal_max: number;
  score: number;
  score_10: number;
  explanation: string;
  scale_min: number;
  scale_max: number;
  soft_margin?: number | null;
  points: MeasurementPoint[];
  segments: MeasurementSegment[];
  formula: Record<string, unknown>;
  view?: "frontal" | "profile" | "three_quarter" | "any";
  order: number;
  total: number;
}

export interface LandmarkPoint {
  x: number;
  y: number;
  z: number;
  index: number;
}

export interface MeasurementGuide {
  id: string;
  label: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface GoldenBox {
  x: number;
  y: number;
  w: number;
  h: number;
  corners?: { x: number; y: number }[];
}

export interface OverlaySegment {
  id?: string;
  label?: string;
  x1: number;
  y1: number;
  x2: number;
  y2: number;
}

export interface OverlayGuides {
  roll_deg?: number;
  origin?: { x: number; y: number };
  midline_x: number;
  thirds_y: number[];
  fifths_x: number[];
  symmetry_line?: OverlaySegment;
  thirds_lines?: OverlaySegment[];
  fifths_lines?: OverlaySegment[];
  golden_boxes: GoldenBox[];
  measurements: MeasurementGuide[];
}

export type FacePose = "frontal" | "three_quarter" | "profile";
export type Gender = "male" | "female";

export interface AnalysisResult {
  overall: number;
  frontal_score: number;
  profile_score: number | null;
  gender?: Gender;
  appeal?: number;
  appeal_10?: number;
  pose: FacePose;
  pose_label: string;
  pose_confidence: number;
  roll_deg?: number;
  cheekbones?: number;
  eye_cut?: number;
  face_shape?: number;
  face_shape_label?: string | null;
  midface?: number;
  brow?: number;
  symmetry: number;
  golden_ratio: number;
  eyes: number;
  nose: number;
  lips: number;
  jaw: number;
  chin: number;
  thirds: number;
  fifths: number;
  face_ratio: number;
  recommendations: string[];
  metrics: Record<string, MetricDetail>;
  measurements?: DetailedMeasurement[];
  landmarks: LandmarkPoint[];
  overlay: OverlayGuides;
  image_width: number;
  image_height: number;
  disclaimer: string;
}

export interface HistoryEntry {
  id: string;
  createdAt: string;
  imageDataUrl: string;
  result: AnalysisResult;
}

export interface OverlayToggles {
  landmarks: boolean;
  symmetry: boolean;
  thirds: boolean;
  fifths: boolean;
  golden: boolean;
  measurements: boolean;
}

/**
 * Client-side helpers to re-score a measurement after point edits.
 */

import type { DetailedMeasurement, MeasurementPoint, MeasurementSegment } from "./types";

function dist(a: { x: number; y: number }, b: { x: number; y: number }): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function byId(points: MeasurementPoint[], id: string): MeasurementPoint | undefined {
  return points.find((p) => p.id === id);
}

function formatValue(value: number, unit: string): string {
  if (unit === "%") return `${value.toFixed(1)}%`;
  if (unit === "°") return `${value.toFixed(1)}°`;
  if (unit === "x") return `${value.toFixed(2)}x`;
  if (unit === "mm") return `${value.toFixed(1)} mm`;
  return value.toFixed(2);
}

function rangeScore(
  value: number,
  idealMin: number,
  idealMax: number,
  softMargin?: number | null,
): number {
  const lo = Math.min(idealMin, idealMax);
  const hi = Math.max(idealMin, idealMax);
  const center = 0.5 * (lo + hi);
  const half = Math.max(0.5 * (hi - lo), 1e-6);
  const sigma =
    softMargin != null && softMargin > 0 ? softMargin : half * 1.15;
  const z = (value - center) / sigma;
  return Math.max(0, Math.min(100, 100 * Math.exp(-0.5 * z * z)));
}

function angleDeg(
  a: { x: number; y: number },
  vertex: { x: number; y: number },
  c: { x: number; y: number },
): number {
  const v1x = a.x - vertex.x;
  const v1y = a.y - vertex.y;
  const v2x = c.x - vertex.x;
  const v2y = c.y - vertex.y;
  const n1 = Math.hypot(v1x, v1y) || 1;
  const n2 = Math.hypot(v2x, v2y) || 1;
  const cos = Math.max(-1, Math.min(1, (v1x * v2x + v1y * v2y) / (n1 * n2)));
  return (Math.acos(cos) * 180) / Math.PI;
}

function lineIntersect(
  p1: { x: number; y: number },
  p2: { x: number; y: number },
  q1: { x: number; y: number },
  q2: { x: number; y: number },
): { x: number; y: number } | null {
  const d1x = p2.x - p1.x;
  const d1y = p2.y - p1.y;
  const d2x = q2.x - q1.x;
  const d2y = q2.y - q1.y;
  const den = d1x * d2y - d1y * d2x;
  if (Math.abs(den) < 1e-9) return null;
  const t = ((q1.x - p1.x) * d2y - (q1.y - p1.y) * d2x) / den;
  return { x: p1.x + t * d1x, y: p1.y + t * d1y };
}

function rebuildSegments(
  m: DetailedMeasurement,
  points: MeasurementPoint[],
  display: string,
): MeasurementSegment[] {
  if (!m.segments.length) return [];
  // Keep topology: map endpoints to nearest current points by original coords order.
  // Simpler: if exactly 2 points, one primary segment; else preserve count using point pairs heuristically.
  const primary = m.segments.filter((s) => (s.style ?? "primary") === "primary");
  const refs = m.segments.filter((s) => s.style === "ref");
  const out: MeasurementSegment[] = [];
  const formulaType = String(m.formula?.type ?? "");

  // JFA / IAA−JFA: draw jaw lines to intersection apex below chin.
  const jl = byId(points, "jl");
  const jr = byId(points, "jr");
  const apex = byId(points, "apex");
  if (jl && jr && apex && (formulaType === "jfa_intersect" || formulaType === "angle_diff")) {
    out.push({
      x1: jl.x,
      y1: jl.y,
      x2: apex.x,
      y2: apex.y,
      style: "primary",
    });
    out.push({
      x1: jr.x,
      y1: jr.y,
      x2: apex.x,
      y2: apex.y,
      style: "primary",
      label: display,
    });
    if (formulaType === "angle_diff") {
      const lo = byId(points, "lo");
      const tip = byId(points, "tip");
      const ro = byId(points, "ro");
      if (lo && tip && ro) {
        out.unshift(
          { x1: lo.x, y1: lo.y, x2: tip.x, y2: tip.y, style: "ref" },
          { x1: ro.x, y1: ro.y, x2: tip.x, y2: tip.y, style: "ref" },
        );
      }
    }
    return out;
  }

  if (points.length >= 2 && primary.length) {
    // Re-draw primary as consecutive pairs / first-last depending on count
    if (points.length === 2) {
      out.push({
        x1: points[0].x,
        y1: points[0].y,
        x2: points[1].x,
        y2: points[1].y,
        style: "primary",
        label: display,
      });
    } else if (points.length >= 4 && primary.length >= 2) {
      out.push({
        x1: points[0].x,
        y1: points[0].y,
        x2: points[1].x,
        y2: points[1].y,
        style: "primary",
        label: display,
      });
      out.push({
        x1: points[2].x,
        y1: points[2].y,
        x2: points[3].x,
        y2: points[3].y,
        style: "primary",
      });
    } else {
      out.push({
        x1: points[0].x,
        y1: points[0].y,
        x2: points[1].x,
        y2: points[1].y,
        style: "primary",
        label: display,
      });
    }
  }

  if (refs.length && points.length >= 4) {
    const a = points[points.length - 2];
    const b = points[points.length - 1];
    out.unshift({
      x1: a.x,
      y1: a.y,
      x2: b.x,
      y2: b.y,
      style: "ref",
    });
  } else if (refs.length && points.length >= 2) {
    // keep original ref geometry if we can't remap cleanly
    out.unshift(...refs);
  }

  return out.length ? out : m.segments.map((s) => ({ ...s, label: s.label ? display : s.label }));
}

/**
 * Recompute value/score after the user dragged overlay points.
 */
export function recomputeMeasurement(
  original: DetailedMeasurement,
  points: MeasurementPoint[],
): DetailedMeasurement {
  const formula = original.formula ?? {};
  const type = String(formula.type ?? "static");
  let value = original.value;
  let nextPoints = points;

  const p = (id: string) => byId(nextPoints, id);

  try {
    if (type === "pct_ratio") {
      const num = (formula.num as string[]) ?? [];
      const den = (formula.den as string[]) ?? [];
      const a = p(num[0]);
      const b = p(num[1]);
      const c = p(den[0]);
      const d = p(den[1]);
      if (a && b && c && d) {
        value = (100 * dist(a, b)) / Math.max(dist(c, d), 1e-6);
      }
    } else if (type === "ratio_hw") {
      const h1 = p(String(formula.h1));
      const h2 = p(String(formula.h2));
      const v1 = p(String(formula.v1));
      const v2 = p(String(formula.v2));
      if (h1 && h2 && v1 && v2) {
        value = dist(h1, h2) / Math.max(dist(v1, v2), 1e-6);
      }
    } else if (type === "angle") {
      const ids = (formula.points as string[]) ?? [];
      const a = p(ids[0]);
      const b = p(ids[1]);
      const c = p(ids[2]);
      if (a && b && c) value = angleDeg(a, b, c);
    } else if (type === "jfa_intersect") {
      const left = (formula.left as string[]) ?? ["jl", "jm"];
      const right = (formula.right as string[]) ?? ["jr", "jrm"];
      const jl = p(left[0]);
      const jm = p(left[1]);
      const jr = p(right[0]);
      const jrm = p(right[1]);
      if (jl && jm && jr && jrm) {
        const apex = lineIntersect(jl, jm, jr, jrm);
        if (apex) {
          nextPoints = nextPoints.map((pt) =>
            pt.id === "apex" ? { ...pt, x: apex.x, y: apex.y } : pt,
          );
          value = angleDeg(jl, apex, jr);
        }
      }
    } else if (type === "angle_diff") {
      const A = (formula.a as string[]) ?? [];
      const B = (formula.b as string[]) ?? [];
      const a0 = p(A[0]);
      const a1 = p(A[1]);
      const a2 = p(A[2]);
      const b0 = p(B[0]);
      const b1 = p(B[1]);
      const b2 = p(B[2]);
      if (a0 && a1 && a2 && b0 && b1 && b2) {
        value = Math.abs(angleDeg(a0, a1, a2) - angleDeg(b0, b1, b2));
      }
    } else if (type === "eye_aspect" && points.length >= 8) {
      const [li, lo, lt, lb, ri, ro, rt, rb] = points;
      const left = dist(li, lo) / Math.max(dist(lt, lb), 1e-6);
      const right = dist(ri, ro) / Math.max(dist(rt, rb), 1e-6);
      value = (left + right) / 2;
    } else if (type === "canthal_tilt" && points.length >= 4) {
      const [li, lo, ri, ro] = points;
      const tilt = (a: MeasurementPoint, b: MeasurementPoint) => {
        const dx = b.x - a.x;
        const dy = b.y - a.y;
        return (Math.atan2(-dy, Math.abs(dx)) * 180) / Math.PI;
      };
      value = (tilt(li, lo) + tilt(ri, ro)) / 2;
    } else if (points.length === 2) {
      // generic distance — keep unit; for % metrics leave original unless formula known
      if (original.unit === "x") {
        // can't infer denominator
      }
    }
  } catch {
    value = original.value;
  }

  const score = rangeScore(
    value,
    original.ideal_min,
    original.ideal_max,
    original.soft_margin,
  );
  const display = formatValue(value, original.unit);
  return {
    ...original,
    points: nextPoints,
    value: Number(value.toFixed(4)),
    display,
    score: Number(score.toFixed(1)),
    score_10: Number((score / 10).toFixed(1)),
    segments: rebuildSegments(original, nextPoints, display),
  };
}

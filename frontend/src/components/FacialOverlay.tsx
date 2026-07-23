"use client";

import { useEffect, useRef } from "react";
import type { AnalysisResult, OverlayToggles } from "@/lib/types";

interface FacialOverlayProps {
  imageUrl: string;
  result: AnalysisResult;
  toggles: OverlayToggles;
}

interface Segment {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  label?: string;
}

/**
 * Draw a normalized line segment on the canvas.
 */
function drawSegment(
  ctx: CanvasRenderingContext2D,
  seg: Segment,
  width: number,
  height: number,
  withLabel = false,
) {
  ctx.beginPath();
  ctx.moveTo(seg.x1 * width, seg.y1 * height);
  ctx.lineTo(seg.x2 * width, seg.y2 * height);
  ctx.stroke();
  if (withLabel && seg.label) {
    const mx = ((seg.x1 + seg.x2) / 2) * width + 4;
    const my = ((seg.y1 + seg.y2) / 2) * height - 4;
    ctx.fillText(seg.label, mx, my);
  }
}

/**
 * Canvas overlay with roll-aware (tilted) facial guides.
 */
export function FacialOverlay({ imageUrl, result, toggles }: FacialOverlayProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;

    const image = new Image();
    image.onload = () => {
      const maxWidth = 720;
      const scale = Math.min(1, maxWidth / image.width);
      const width = Math.round(image.width * scale);
      const height = Math.round(image.height * scale);
      canvas.width = width;
      canvas.height = height;
      ctx.clearRect(0, 0, width, height);
      ctx.drawImage(image, 0, 0, width, height);

      const { overlay, landmarks } = result;

      if (toggles.landmarks) {
        ctx.fillStyle = "rgba(167, 139, 250, 0.85)";
        for (const lm of landmarks) {
          if (lm.index % 3 !== 0) continue;
          ctx.beginPath();
          ctx.arc(lm.x * width, lm.y * height, 1.4, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      if (toggles.symmetry) {
        ctx.strokeStyle = "rgba(56, 189, 248, 0.9)";
        ctx.lineWidth = 1.6;
        ctx.setLineDash([6, 6]);
        if (overlay.symmetry_line) {
          drawSegment(ctx, overlay.symmetry_line, width, height);
        } else {
          ctx.beginPath();
          ctx.moveTo(overlay.midline_x * width, 0);
          ctx.lineTo(overlay.midline_x * width, height);
          ctx.stroke();
        }
        ctx.setLineDash([]);
      }

      if (toggles.thirds) {
        ctx.strokeStyle = "rgba(196, 181, 253, 0.9)";
        ctx.lineWidth = 1.5;
        if (overlay.thirds_lines?.length) {
          for (const line of overlay.thirds_lines) {
            drawSegment(ctx, line, width, height);
          }
        } else {
          for (const y of overlay.thirds_y) {
            ctx.beginPath();
            ctx.moveTo(0, y * height);
            ctx.lineTo(width, y * height);
            ctx.stroke();
          }
        }
      }

      if (toggles.fifths) {
        ctx.strokeStyle = "rgba(125, 211, 252, 0.8)";
        ctx.lineWidth = 1.25;
        if (overlay.fifths_lines?.length) {
          for (const line of overlay.fifths_lines) {
            drawSegment(ctx, line, width, height);
          }
        } else {
          for (const x of overlay.fifths_x) {
            ctx.beginPath();
            ctx.moveTo(x * width, 0);
            ctx.lineTo(x * width, height);
            ctx.stroke();
          }
        }
      }

      if (toggles.golden) {
        ctx.strokeStyle = "rgba(251, 191, 36, 0.9)";
        ctx.lineWidth = 1.75;
        for (const box of overlay.golden_boxes) {
          const corners = box.corners as { x: number; y: number }[] | undefined;
          if (corners?.length === 4) {
            ctx.beginPath();
            ctx.moveTo(corners[0].x * width, corners[0].y * height);
            for (let i = 1; i < 4; i += 1) {
              ctx.lineTo(corners[i].x * width, corners[i].y * height);
            }
            ctx.closePath();
            ctx.stroke();
          } else {
            ctx.strokeRect(box.x * width, box.y * height, box.w * width, box.h * height);
          }
        }
      }

      if (toggles.measurements) {
        ctx.strokeStyle = "rgba(52, 211, 153, 0.95)";
        ctx.fillStyle = "rgba(167, 243, 208, 0.95)";
        ctx.lineWidth = 1.5;
        ctx.font = "11px sans-serif";
        for (const m of overlay.measurements) {
          drawSegment(ctx, m, width, height, true);
        }
      }
    };
    image.src = imageUrl;
  }, [imageUrl, result, toggles]);

  return (
    <canvas
      ref={canvasRef}
      className="mx-auto max-w-full rounded-3xl border border-white/10 shadow-2xl"
    />
  );
}

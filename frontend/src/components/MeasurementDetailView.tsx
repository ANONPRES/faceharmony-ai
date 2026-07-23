"use client";

import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { X } from "lucide-react";
import type { DetailedMeasurement, MeasurementPoint } from "@/lib/types";
import { recomputeMeasurement } from "@/lib/recomputeMeasurement";

interface MeasurementDetailViewProps {
  imageUrl: string;
  measurements: DetailedMeasurement[];
  initialIndex?: number;
  onClose: () => void;
}

function IdealSlider({ m }: { m: DetailedMeasurement }) {
  const span = Math.max(m.scale_max - m.scale_min, 1e-6);
  const idealLo = ((m.ideal_min - m.scale_min) / span) * 100;
  const idealHi = ((m.ideal_max - m.scale_min) / span) * 100;
  const pos = Math.max(0, Math.min(100, ((m.value - m.scale_min) / span) * 100));
  const inIdeal = m.value >= m.ideal_min && m.value <= m.ideal_max;

  return (
    <div className="relative px-1 pt-8 pb-2">
      <div
        className="absolute -translate-x-1/2 rounded-md px-2 py-0.5 text-xs font-semibold text-white shadow"
        style={{
          left: `${pos}%`,
          top: 0,
          background: inIdeal ? "#16a34a" : "#ca8a04",
        }}
      >
        {m.display}
        <span
          className="absolute left-1/2 top-full -translate-x-1/2 border-4 border-transparent"
          style={{ borderTopColor: inIdeal ? "#16a34a" : "#ca8a04" }}
        />
      </div>
      <div
        className="relative h-3 overflow-hidden rounded-full"
        style={{
          background:
            "linear-gradient(90deg, #ef4444 0%, #f97316 18%, #22c55e 42%, #22c55e 58%, #f97316 82%, #ef4444 100%)",
        }}
      >
        <div
          className="absolute top-0 h-full bg-emerald-400/35"
          style={{ left: `${idealLo}%`, width: `${Math.max(2, idealHi - idealLo)}%` }}
        />
        <div
          className="absolute top-1/2 h-4 w-0.5 -translate-y-1/2 bg-white"
          style={{ left: `${pos}%` }}
        />
      </div>
    </div>
  );
}

/**
 * Competitor-style per-measurement carousel: value / ideal / score + editable overlay.
 */
export function MeasurementDetailView({
  imageUrl,
  measurements,
  initialIndex = 0,
  onClose,
}: MeasurementDetailViewProps) {
  const [index, setIndex] = useState(initialIndex);
  const [local, setLocal] = useState<DetailedMeasurement[]>(measurements);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ id: string } | null>(null);
  const sizeRef = useRef({ w: 1, h: 1 });

  useEffect(() => {
    setLocal(measurements);
    setIndex(Math.min(initialIndex, Math.max(0, measurements.length - 1)));
  }, [measurements, initialIndex]);

  const current = local[index];
  const total = local.length;

  const fmtIdeal = useMemo(() => {
    if (!current) return "";
    const u = current.unit;
    const a = current.ideal_min;
    const b = current.ideal_max;
    if (u === "%") return `${a.toFixed(2)}–${b.toFixed(2)}`;
    if (u === "°") return `${a.toFixed(2)}–${b.toFixed(2)}`;
    if (u === "x") return `${a.toFixed(2)} – ${b.toFixed(2)}`;
    if (u === "mm") return `${a.toFixed(2)} – ${b.toFixed(2)}`;
    return `${a} – ${b}`;
  }, [current]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas || !current) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const image = new Image();
    image.onload = () => {
      const maxW = 560;
      const scale = Math.min(1, maxW / image.width);
      const w = Math.round(image.width * scale);
      const h = Math.round(image.height * scale);
      sizeRef.current = { w, h };
      canvas.width = w;
      canvas.height = h;
      ctx.clearRect(0, 0, w, h);
      ctx.drawImage(image, 0, 0, w, h);

      for (const seg of current.segments) {
        const primary = (seg.style ?? "primary") === "primary";
        ctx.strokeStyle = primary ? "#22c55e" : "rgba(255,255,255,0.75)";
        ctx.lineWidth = primary ? 2.4 : 1.4;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(seg.x1 * w, seg.y1 * h);
        ctx.lineTo(seg.x2 * w, seg.y2 * h);
        ctx.stroke();
        if (seg.label) {
          ctx.fillStyle = "#4ade80";
          ctx.font = "600 13px system-ui, sans-serif";
          ctx.fillText(
            seg.label,
            ((seg.x1 + seg.x2) / 2) * w + 6,
            ((seg.y1 + seg.y2) / 2) * h - 8,
          );
        }
      }

      for (const pt of current.points) {
        const x = pt.x * w;
        const y = pt.y * h;
        ctx.beginPath();
        ctx.arc(x, y, 7, 0, Math.PI * 2);
        ctx.fillStyle = "rgba(255,255,255,0.95)";
        ctx.fill();
        ctx.lineWidth = 2;
        ctx.strokeStyle = "#16a34a";
        ctx.stroke();
      }
    };
    image.src = imageUrl;
  }, [current, imageUrl]);

  useEffect(() => {
    draw();
  }, [draw]);

  const updatePoint = (id: string, x: number, y: number) => {
    setLocal((prev) => {
      const copy = [...prev];
      const m = copy[index];
      if (!m) return prev;
      const points: MeasurementPoint[] = m.points.map((p) =>
        p.id === id ? { ...p, x, y } : p,
      );
      copy[index] = recomputeMeasurement(m, points);
      return copy;
    });
  };

  const pointerToNorm = (e: React.PointerEvent<HTMLCanvasElement>) => {
    const canvas = canvasRef.current;
    if (!canvas) return null;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX - rect.left) / rect.width;
    const y = (e.clientY - rect.top) / rect.height;
    return {
      x: Math.max(0, Math.min(1, x)),
      y: Math.max(0, Math.min(1, y)),
    };
  };

  const onPointerDown = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!current) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const norm = pointerToNorm(e);
    if (!norm) return;
    const { w, h } = sizeRef.current;
    let best: { id: string; d: number } | null = null;
    for (const pt of current.points) {
      const dx = (pt.x - norm.x) * w;
      const dy = (pt.y - norm.y) * h;
      const d = Math.hypot(dx, dy);
      if (d < 18 && (!best || d < best.d)) best = { id: pt.id, d };
    }
    if (best) {
      dragRef.current = { id: best.id };
      canvas.setPointerCapture(e.pointerId);
    }
  };

  const onPointerMove = (e: React.PointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    const norm = pointerToNorm(e);
    if (!norm) return;
    updatePoint(dragRef.current.id, norm.x, norm.y);
  };

  const onPointerUp = () => {
    dragRef.current = null;
  };

  if (!current) return null;

  const scoreTone =
    current.score_10 >= 9
      ? "text-emerald-600 bg-emerald-50"
      : current.score_10 >= 7
        ? "text-lime-700 bg-lime-50"
        : "text-amber-700 bg-amber-50";

  return (
    <div className="fixed inset-0 z-50 flex items-end justify-center bg-black/70 p-3 sm:items-center">
      <div className="flex max-h-[96vh] w-full max-w-md flex-col overflow-hidden rounded-3xl bg-[#f4f5f7] text-[#111] shadow-2xl">
        <div className="flex items-start justify-between gap-3 px-4 pb-2 pt-4">
          <h2 className="text-lg font-semibold leading-tight">{current.label}</h2>
          <button
            type="button"
            onClick={onClose}
            className="rounded-full p-1.5 text-neutral-500 hover:bg-black/5"
            aria-label="Закрыть"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="grid grid-cols-3 gap-2 px-4">
          <div className="rounded-2xl bg-white px-3 py-2.5 shadow-sm">
            <div className="text-[10px] font-semibold uppercase tracking-wide text-neutral-400">
              Ваше значение
            </div>
            <div className="mt-1 text-xl font-bold">{current.display}</div>
          </div>
          <div className="rounded-2xl bg-white px-3 py-2.5 shadow-sm">
            <div className="text-[10px] font-semibold uppercase tracking-wide text-neutral-400">
              Идеал
            </div>
            <div className="mt-1 text-sm font-semibold text-emerald-700">{fmtIdeal}</div>
          </div>
          <div className={`rounded-2xl px-3 py-2.5 shadow-sm ${scoreTone}`}>
            <div className="text-[10px] font-semibold uppercase tracking-wide opacity-70">
              Оценка
            </div>
            <div className="mt-1 text-xl font-bold">{current.score_10.toFixed(1)} / 10</div>
          </div>
        </div>

        <div className="mt-3 overflow-y-auto px-4 pb-2">
          <div className="overflow-hidden rounded-2xl bg-neutral-200">
            <canvas
              ref={canvasRef}
              className="mx-auto block max-h-[48vh] w-full touch-none cursor-crosshair object-contain"
              onPointerDown={onPointerDown}
              onPointerMove={onPointerMove}
              onPointerUp={onPointerUp}
              onPointerCancel={onPointerUp}
            />
          </div>
          <p className="mt-2 text-center text-[11px] text-neutral-500">
            Нажмите на точку на фото, чтобы поменять её положение и пересчитать метрики.
          </p>
          <IdealSlider m={current} />
          <p className="mt-1 text-xs leading-relaxed text-neutral-500">{current.explanation}</p>
          <p className="mt-1 text-[11px] uppercase tracking-wide text-neutral-400">
            {current.category}
          </p>
        </div>

        <div className="mt-auto flex items-center justify-between gap-2 border-t border-black/5 bg-white px-3 py-3">
          <button
            type="button"
            disabled={index <= 0}
            onClick={() => setIndex((i) => Math.max(0, i - 1))}
            className="rounded-xl bg-neutral-100 px-4 py-2.5 text-sm font-medium disabled:opacity-40"
          >
            ‹ Назад
          </button>
          <div className="text-sm font-medium text-neutral-500">
            {index + 1} / {total}
          </div>
          <button
            type="button"
            disabled={index >= total - 1}
            onClick={() => setIndex((i) => Math.min(total - 1, i + 1))}
            className="rounded-xl bg-neutral-900 px-4 py-2.5 text-sm font-medium text-white disabled:opacity-40"
          >
            Далее ›
          </button>
        </div>
      </div>
    </div>
  );
}

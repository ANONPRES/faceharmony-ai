"use client";

import { useMemo, useState } from "react";
import type { DetailedMeasurement } from "@/lib/types";
import { MeasurementDetailView } from "./MeasurementDetailView";

interface MeasurementsPanelProps {
  imageUrl: string;
  measurements: DetailedMeasurement[];
}

/** FaceIQ frontal category order (matches backend CAT_* labels). */
const FACEIQ_CATEGORY_ORDER = [
  "Трети лица",
  "Форма лица",
  "Глаза",
  "Нос",
  "Рот",
  "Челюсть",
  "Прочее",
];

/**
 * Grid of all atomic measurements; opens competitor-style detail carousel.
 */
export function MeasurementsPanel({ imageUrl, measurements }: MeasurementsPanelProps) {
  const [openIndex, setOpenIndex] = useState<number | null>(null);

  const byCategory = useMemo(() => {
    const map = new Map<string, DetailedMeasurement[]>();
    for (const m of measurements) {
      const list = map.get(m.category) ?? [];
      list.push(m);
      map.set(m.category, list);
    }
    const ordered: [string, DetailedMeasurement[]][] = [];
    for (const cat of FACEIQ_CATEGORY_ORDER) {
      const items = map.get(cat);
      if (items?.length) ordered.push([cat, items]);
      map.delete(cat);
    }
    for (const [cat, items] of map.entries()) {
      ordered.push([cat, items]);
    }
    return ordered;
  }, [measurements]);

  if (!measurements.length) {
    return (
      <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-6 text-sm text-white/50">
        Детальные измерения появятся после нового анализа (перезагрузите фото).
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-end justify-between gap-3">
        <div>
          <h2 className="font-[family-name:var(--font-display)] text-2xl text-white">
            Детальные измерения
          </h2>
          <p className="mt-1 text-sm text-white/50">
            Группы как в FaceIQ: трети, форма, глаза, нос, рот, челюсть.{" "}
            {measurements.length} метрик. Профиль (gonial и т.п.) — только с бокового фото.
          </p>
        </div>
        <button
          type="button"
          onClick={() => setOpenIndex(0)}
          className="rounded-full bg-gradient-to-r from-violet-500 to-sky-500 px-4 py-2 text-sm font-medium text-white"
        >
          Смотреть все по очереди
        </button>
      </div>

      {byCategory.map(([category, items]) => {
        const groupAvg =
          items.reduce((s, m) => s + m.score_10, 0) / Math.max(items.length, 1);
        return (
          <div key={category} className="space-y-3">
            <div className="flex items-baseline justify-between gap-3">
              <h3 className="text-xs font-semibold uppercase tracking-[0.18em] text-violet-200/70">
                {category}
              </h3>
              <span className="text-sm text-white/45">
                {groupAvg.toFixed(1)}
                <span className="text-white/30">/10</span>
              </span>
            </div>
            <div className="grid gap-2 sm:grid-cols-2">
              {items.map((m) => {
                const globalIndex = measurements.findIndex((x) => x.id === m.id);
                const tone =
                  m.score_10 >= 9
                    ? "text-emerald-300"
                    : m.score_10 >= 7
                      ? "text-sky-300"
                      : "text-amber-300";
                return (
                  <button
                    key={m.id}
                    type="button"
                    onClick={() => setOpenIndex(globalIndex)}
                    className="flex items-center justify-between gap-3 rounded-2xl border border-white/10 bg-white/[0.04] px-4 py-3 text-left transition hover:border-violet-400/40 hover:bg-white/[0.07]"
                  >
                    <div className="min-w-0">
                      <div className="truncate text-sm font-medium text-white">{m.label}</div>
                      <div className="mt-0.5 text-xs text-white/45">
                        {m.display} · идеал {m.ideal_min}–{m.ideal_max}
                      </div>
                    </div>
                    <div className={`shrink-0 text-lg font-semibold ${tone}`}>
                      {m.score_10.toFixed(1)}
                      <span className="text-xs font-normal text-white/35">/10</span>
                    </div>
                  </button>
                );
              })}
            </div>
          </div>
        );
      })}

      {openIndex !== null && (
        <MeasurementDetailView
          imageUrl={imageUrl}
          measurements={measurements}
          initialIndex={openIndex}
          onClose={() => setOpenIndex(null)}
        />
      )}
    </div>
  );
}

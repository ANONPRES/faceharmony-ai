"use client";

import type { OverlayToggles } from "@/lib/types";

interface OverlayControlsProps {
  toggles: OverlayToggles;
  onChange: (next: OverlayToggles) => void;
}

const LABELS: { key: keyof OverlayToggles; label: string }[] = [
  { key: "landmarks", label: "Точки" },
  { key: "symmetry", label: "Ось симметрии" },
  { key: "thirds", label: "Трети" },
  { key: "fifths", label: "Пятые" },
  { key: "golden", label: "Золотое сечение" },
  { key: "measurements", label: "Измерения" },
];

/**
 * Переключатели слоёв оверлея.
 */
export function OverlayControls({ toggles, onChange }: OverlayControlsProps) {
  return (
    <div className="flex flex-wrap gap-2">
      {LABELS.map(({ key, label }) => {
        const active = toggles[key];
        return (
          <button
            key={key}
            type="button"
            onClick={() => onChange({ ...toggles, [key]: !active })}
            className={`rounded-full border px-3 py-1.5 text-xs transition ${
              active
                ? "border-violet-300/40 bg-violet-500/20 text-violet-100"
                : "border-white/10 bg-white/5 text-white/50 hover:text-white"
            }`}
          >
            {label}
          </button>
        );
      })}
    </div>
  );
}

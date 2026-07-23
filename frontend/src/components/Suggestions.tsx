"use client";

import { Lightbulb } from "lucide-react";
import { GlassCard } from "./GlassCard";

interface SuggestionsProps {
  recommendations: string[];
}

/**
 * Neutral educational suggestions list derived from metric scores.
 */
export function Suggestions({ recommendations }: SuggestionsProps) {
  return (
    <GlassCard delay={0.15}>
      <div className="mb-4 flex items-center gap-2">
        <Lightbulb className="h-5 w-5 text-sky-300" />
        <h2 className="font-[family-name:var(--font-display)] text-xl text-white">
          Наблюдения
        </h2>
      </div>
      <ul className="space-y-3">
        {recommendations.map((tip) => (
          <li
            key={tip}
            className="rounded-2xl border border-white/8 bg-white/[0.03] px-4 py-3 text-sm leading-relaxed text-white/70"
          >
            {tip}
          </li>
        ))}
      </ul>
    </GlassCard>
  );
}

"use client";

import { CheckCircle2 } from "lucide-react";
import { GlassCard } from "./GlassCard";
import type { MetricDetail } from "@/lib/types";

interface MetricCardProps {
  metric: MetricDetail;
  delay?: number;
}

/**
 * Breakdown card showing a metric score, bar, and short explanation.
 */
export function MetricCard({ metric, delay = 0 }: MetricCardProps) {
  const tone =
    metric.score >= 85
      ? "from-emerald-400 to-cyan-400"
      : metric.score >= 70
        ? "from-violet-400 to-sky-400"
        : "from-amber-300 to-orange-400";

  return (
    <GlassCard delay={delay} className="flex h-full flex-col gap-3">
      <div className="flex items-start justify-between gap-3">
        <div>
          <div className="flex items-center gap-2">
            <CheckCircle2 className="h-4 w-4 text-violet-300" />
            <h3 className="font-medium text-white">{metric.label}</h3>
          </div>
          <p className="mt-2 text-sm leading-relaxed text-white/55">{metric.explanation}</p>
        </div>
        <div className="shrink-0 text-right">
          <div className="font-[family-name:var(--font-display)] text-3xl text-white">
            {Math.round(metric.score)}
          </div>
          <div className="text-[11px] uppercase tracking-wider text-white/40">балл</div>
        </div>
      </div>
      <div className="mt-auto h-2 overflow-hidden rounded-full bg-white/10">
        <div
          className={`h-full rounded-full bg-gradient-to-r ${tone} transition-all duration-700`}
          style={{ width: `${Math.max(4, Math.min(100, metric.score))}%` }}
        />
      </div>
    </GlassCard>
  );
}

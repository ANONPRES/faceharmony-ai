"use client";

import {
  PolarAngleAxis,
  PolarGrid,
  Radar,
  RadarChart,
  ResponsiveContainer,
  Tooltip,
} from "recharts";
import { GlassCard } from "./GlassCard";
import type { AnalysisResult } from "@/lib/types";

interface MetricsChartProps {
  result: AnalysisResult;
}

/**
 * Радар по ключевым эстетическим чертам.
 */
export function MetricsChart({ result }: MetricsChartProps) {
  const data = [
    { metric: "Appeal", score: result.appeal ?? result.metrics.appeal?.score ?? 0 },
    { metric: "Скулы", score: result.cheekbones ?? result.metrics.cheekbones?.score ?? 0 },
    { metric: "Вырез", score: result.eye_cut ?? result.metrics.eye_cut?.score ?? 0 },
    { metric: "Нос", score: result.nose },
    { metric: "Форма", score: result.face_shape ?? result.metrics.face_shape?.score ?? 0 },
    { metric: "Челюсть", score: result.jaw },
    { metric: "Подбородок", score: result.chin },
    { metric: "Губы", score: result.lips },
    { metric: "Симметрия", score: result.symmetry },
    { metric: "Midface", score: result.midface ?? result.metrics.midface?.score ?? 0 },
  ];

  return (
    <GlassCard delay={0.1}>
      <h2 className="mb-4 font-[family-name:var(--font-display)] text-xl text-white">
        Радар черт
      </h2>
      <div className="h-72 w-full">
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={data}>
            <PolarGrid stroke="rgba(255,255,255,0.12)" />
            <PolarAngleAxis
              dataKey="metric"
              tick={{ fill: "rgba(255,255,255,0.55)", fontSize: 11 }}
            />
            <Radar
              name="Балл"
              dataKey="score"
              stroke="#8b5cf6"
              fill="#8b5cf6"
              fillOpacity={0.35}
            />
            <Tooltip
              contentStyle={{
                background: "#12122a",
                border: "1px solid rgba(255,255,255,0.1)",
                borderRadius: 12,
                color: "#fff",
              }}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>
    </GlassCard>
  );
}

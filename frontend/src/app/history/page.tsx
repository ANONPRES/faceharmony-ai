"use client";

import { useEffect, useMemo, useState } from "react";
import Link from "next/link";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { GlassCard } from "@/components/GlassCard";
import { clearHistory, loadHistory, removeHistoryEntry } from "@/lib/history";
import type { HistoryEntry } from "@/lib/types";
import { Trash2 } from "lucide-react";

/**
 * История анализов и сравнение двух записей.
 */
export default function HistoryPage() {
  const [entries, setEntries] = useState<HistoryEntry[]>([]);
  const [selected, setSelected] = useState<string[]>([]);

  useEffect(() => {
    setEntries(loadHistory());
  }, []);

  const toggleSelect = (id: string) => {
    setSelected((prev) => {
      if (prev.includes(id)) return prev.filter((x) => x !== id);
      if (prev.length >= 2) return [prev[1], id];
      return [...prev, id];
    });
  };

  const comparisonData = useMemo(() => {
    const chosen = selected
      .map((id) => entries.find((e) => e.id === id))
      .filter(Boolean) as HistoryEntry[];
    if (chosen.length < 2) return [];

    const keys = [
      "overall",
      "frontal_score",
      "profile_score",
      "symmetry",
      "golden_ratio",
      "thirds",
      "fifths",
      "eyes",
      "nose",
      "lips",
      "jaw",
      "chin",
    ] as const;

    const labels: Record<(typeof keys)[number], string> = {
      overall: "overall",
      frontal_score: "анфас",
      profile_score: "профиль",
      symmetry: "симметрия",
      golden_ratio: "золото",
      thirds: "трети",
      fifths: "пятые",
      eyes: "глаза",
      nose: "нос",
      lips: "губы",
      jaw: "челюсть",
      chin: "подбородок",
    };

    return keys.map((key) => ({
      metric: labels[key],
      A: Number(chosen[0].result[key] ?? 0),
      B: Number(chosen[1].result[key] ?? 0),
    }));
  }, [entries, selected]);

  const refresh = () => setEntries(loadHistory());

  return (
    <div className="space-y-8">
      <div className="flex flex-wrap items-end justify-between gap-4">
        <div>
          <h1 className="font-[family-name:var(--font-display)] text-4xl text-white sm:text-5xl">
            История
          </h1>
          <p className="mt-2 max-w-xl text-sm text-white/55">
            Анализы хранятся локально в браузере. Выберите две записи для
            сравнения — лучше один ракурс с одним (анфас↔анфас).
          </p>
        </div>
        {entries.length > 0 && (
          <button
            type="button"
            onClick={() => {
              clearHistory();
              setSelected([]);
              refresh();
            }}
            className="rounded-full border border-white/15 px-4 py-2 text-sm text-white/70 hover:bg-white/5"
          >
            Очистить всё
          </button>
        )}
      </div>

      {entries.length === 0 ? (
        <GlassCard className="text-center">
          <p className="text-white/60">Пока нет сохранённых анализов.</p>
          <Link
            href="/upload"
            className="mt-4 inline-flex rounded-full bg-gradient-to-r from-violet-500 to-sky-500 px-5 py-2.5 text-sm font-medium text-white"
          >
            Сделать первый анализ
          </Link>
        </GlassCard>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {entries.map((entry) => {
            const active = selected.includes(entry.id);
            const poseLabel = entry.result.pose_label ?? "Анфас";
            return (
              <GlassCard key={entry.id} className="space-y-3 !p-4">
                <button
                  type="button"
                  onClick={() => toggleSelect(entry.id)}
                  className={`w-full overflow-hidden rounded-2xl border text-left transition ${
                    active
                      ? "border-violet-400/60 ring-2 ring-violet-400/30"
                      : "border-white/10"
                  }`}
                >
                  {/* eslint-disable-next-line @next/next/no-img-element */}
                  <img
                    src={entry.imageDataUrl}
                    alt="Превью анализа"
                    className="h-40 w-full object-cover"
                  />
                </button>
                <div className="flex items-start justify-between gap-2">
                  <div>
                    <div className="font-[family-name:var(--font-display)] text-2xl text-white">
                      {Number(entry.result.overall).toFixed(1)}
                      <span className="text-sm text-white/40"> / 100</span>
                    </div>
                    <div className="mt-1 text-xs text-violet-200/80">{poseLabel}</div>
                    <div className="text-xs text-white/40">
                      {new Date(entry.createdAt).toLocaleString("ru-RU")}
                    </div>
                  </div>
                  <button
                    type="button"
                    aria-label="Удалить"
                    onClick={() => {
                      removeHistoryEntry(entry.id);
                      setSelected((prev) => prev.filter((id) => id !== entry.id));
                      refresh();
                    }}
                    className="rounded-full border border-white/10 p-2 text-white/50 hover:text-rose-300"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-2 text-center text-[11px] text-white/55">
                  <div className="rounded-xl bg-white/5 py-1.5">
                    анфас{" "}
                    {Number(entry.result.frontal_score ?? entry.result.overall).toFixed(1)}
                  </div>
                  <div className="rounded-xl bg-white/5 py-1.5">
                    профиль{" "}
                    {entry.result.profile_score == null
                      ? "—"
                      : Number(entry.result.profile_score).toFixed(1)}
                  </div>
                </div>
                <div className="flex gap-2">
                  <Link
                    href={`/results?id=${entry.id}`}
                    className="flex-1 rounded-full bg-white/10 py-2 text-center text-xs text-white"
                  >
                    Открыть
                  </Link>
                  <button
                    type="button"
                    onClick={() => toggleSelect(entry.id)}
                    className="flex-1 rounded-full border border-white/10 py-2 text-xs text-white/70"
                  >
                    {active ? "Выбрано" : "Сравнить"}
                  </button>
                </div>
              </GlassCard>
            );
          })}
        </div>
      )}

      {comparisonData.length > 0 && (
        <GlassCard>
          <h2 className="mb-4 font-[family-name:var(--font-display)] text-xl text-white">
            Сравнение
          </h2>
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={comparisonData}>
                <CartesianGrid stroke="rgba(255,255,255,0.08)" vertical={false} />
                <XAxis
                  dataKey="metric"
                  tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                />
                <YAxis
                  domain={[0, 100]}
                  tick={{ fill: "rgba(255,255,255,0.5)", fontSize: 11 }}
                />
                <Tooltip
                  contentStyle={{
                    background: "#12122a",
                    border: "1px solid rgba(255,255,255,0.1)",
                    borderRadius: 12,
                  }}
                />
                <Legend />
                <Bar dataKey="A" fill="#8b5cf6" radius={[6, 6, 0, 0]} />
                <Bar dataKey="B" fill="#38bdf8" radius={[6, 6, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </GlassCard>
      )}
    </div>
  );
}

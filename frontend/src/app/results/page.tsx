"use client";

import { useEffect, useMemo, useState, Suspense } from "react";
import Link from "next/link";
import { useSearchParams } from "next/navigation";
import { ScoreRing } from "@/components/ScoreRing";
import { MetricCard } from "@/components/MetricCard";
import { Suggestions } from "@/components/Suggestions";
import { FacialOverlay } from "@/components/FacialOverlay";
import { OverlayControls } from "@/components/OverlayControls";
import { MetricsChart } from "@/components/MetricsChart";
import { GlassCard } from "@/components/GlassCard";
import { MeasurementsPanel } from "@/components/MeasurementsPanel";
import { loadLatestAnalysis } from "@/lib/session";
import { getHistoryEntry } from "@/lib/history";
import type { AnalysisResult, OverlayToggles } from "@/lib/types";

/**
 * Контент отчёта с поддержкой ?id= из истории.
 */
function ResultsContent() {
  const searchParams = useSearchParams();
  const historyId = searchParams.get("id");
  const [imageUrl, setImageUrl] = useState<string | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [toggles, setToggles] = useState<OverlayToggles>({
    landmarks: true,
    symmetry: true,
    thirds: true,
    fifths: false,
    golden: false,
    measurements: false,
  });

  useEffect(() => {
    let cancelled = false;
    const load = async () => {
      if (historyId) {
        const entry = getHistoryEntry(historyId);
        if (entry) {
          if (!cancelled) {
            setImageUrl(entry.imageDataUrl);
            setResult(entry.result);
          }
          return;
        }
      }
      const latest = await loadLatestAnalysis();
      if (!cancelled) {
        setImageUrl(latest.imageDataUrl);
        setResult(latest.result);
      }
    };
    void load();
    return () => {
      cancelled = true;
    };
  }, [historyId]);

  const cards = useMemo(() => {
    if (!result) return [];
    const order = [
      "harmony",
      "cheekbones",
      "eye_cut",
      "nose",
      "face_shape",
      "jaw",
      "chin",
      "lips",
      "midface",
      "brow",
      "symmetry",
      "eyes",
      "thirds",
      "fifths",
      "golden_ratio",
      "face_ratio",
    ];
    return order.map((key) => result.metrics[key]).filter(Boolean);
  }, [result]);

  if (!result) {
    return (
      <GlassCard className="mx-auto max-w-lg text-center">
        <h1 className="font-[family-name:var(--font-display)] text-2xl text-white">
          Пока нет анализа
        </h1>
        <p className="mt-2 text-sm text-white/55">
          Загрузите фото, чтобы получить отчёт о гармонии лица.
        </p>
        <Link
          href="/upload"
          className="mt-6 inline-flex rounded-full bg-gradient-to-r from-violet-500 to-sky-500 px-5 py-2.5 text-sm font-medium text-white"
        >
          Анализировать лицо
        </Link>
      </GlassCard>
    );
  }

  const previewUrl = imageUrl ?? "";

  const poseHint =
    result.pose === "profile"
      ? "Профиль: оценка по силуэту (нос, подбородок, челюсть), не по ширинам анфаса."
      : result.pose === "three_quarter"
        ? "Ракурс ¾: часть метрик менее надёжна — лучше строгое анфас + отдельный профиль."
        : "Overall = FaceIQ Harmony (анфас). Angularity / Dimorphism / Features — скоро, как на FaceIQ.";

  return (
    <div className="space-y-8">
      <div className="text-center">
        <h1 className="font-[family-name:var(--font-display)] text-4xl text-white sm:text-5xl">
          Эстетика лица
        </h1>
        <p className="mx-auto mt-3 max-w-2xl text-sm text-white/55">
          {result.disclaimer}
        </p>
        {result.face_shape_label && (
          <p className="mt-3 text-sm text-violet-200/90">
            Форма лица: {result.face_shape_label}
          </p>
        )}
        {result.gender && (
          <p className="mt-2 text-sm text-white/50">
            Идеалы: {result.gender === "female" ? "женщина" : "мужчина"}
          </p>
        )}
      </div>

      <div className="grid gap-6 lg:grid-cols-[1.05fr_0.95fr]">
        <GlassCard className="flex flex-col items-center justify-center gap-6">
          <ScoreRing score={result.overall} label="Harmony" />
          {typeof result.appeal_10 === "number" && (
            <div className="text-center">
              <div className="text-[11px] uppercase tracking-wider text-amber-200/70">
                Harmony
              </div>
              <div className="mt-1 font-[family-name:var(--font-display)] text-3xl text-amber-100">
                {result.appeal_10.toFixed(1)}
                <span className="text-base text-white/40"> / 10</span>
              </div>
            </div>
          )}

          <div className="grid w-full max-w-md grid-cols-2 gap-2">
            {(
              [
                ["Harmony", result.pillars?.harmony ?? result.harmony],
                ["Angularity", null],
                ["Dimorphism", null],
                ["Features", null],
              ] as const
            ).map(([label, value]) => (
              <div
                key={label}
                className="rounded-xl border border-white/10 bg-white/[0.03] px-3 py-2 text-center"
              >
                <div className="text-[10px] uppercase tracking-wider text-white/45">
                  {label}
                </div>
                <div className="mt-0.5 font-[family-name:var(--font-display)] text-lg text-white">
                  {typeof value === "number" ? (
                    <>
                      {(value / 10).toFixed(1)}
                      <span className="text-xs text-white/35">/10</span>
                    </>
                  ) : (
                    <span className="text-sm text-white/35">скоро</span>
                  )}
                </div>
              </div>
            ))}
          </div>
          <div className="flex flex-wrap items-center justify-center gap-2">
            <span className="rounded-full border border-violet-300/30 bg-violet-500/15 px-3 py-1 text-xs text-violet-100">
              {result.pose_label}
            </span>
            {typeof result.roll_deg === "number" && Math.abs(result.roll_deg) >= 1 && (
              <span className="rounded-full border border-amber-300/30 bg-amber-500/10 px-3 py-1 text-xs text-amber-100">
                наклон {result.roll_deg.toFixed(1)}°
              </span>
            )}
            <span className="rounded-full border border-white/10 bg-white/5 px-3 py-1 text-xs text-white/55">
              уверенность {(result.pose_confidence * 100).toFixed(0)}%
            </span>
          </div>
          <p className="max-w-sm text-center text-sm text-white/50">{poseHint}</p>

          <div className="grid w-full max-w-md grid-cols-2 gap-3">
            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-center">
              <div className="text-[11px] uppercase tracking-wider text-sky-200/70">
                Анфас
              </div>
              <div className="mt-1 font-[family-name:var(--font-display)] text-3xl text-white">
                {Number(result.frontal_score).toFixed(1)}
              </div>
            </div>
            <div className="rounded-2xl border border-white/10 bg-white/[0.03] p-4 text-center">
              <div className="text-[11px] uppercase tracking-wider text-violet-200/70">
                Профиль
              </div>
              <div className="mt-1 font-[family-name:var(--font-display)] text-3xl text-white">
                {result.profile_score == null
                  ? "—"
                  : Number(result.profile_score).toFixed(1)}
              </div>
              {result.profile_score == null && (
                <p className="mt-1 text-[10px] text-white/35">нет фото профиля</p>
              )}
            </div>
          </div>
          <p className="max-w-md text-center text-xs text-white/40">
            Profile score считается только по отдельному фото профиля. Overall —
            анфас, либо смесь 65/35 если профиль загружен.
          </p>
        </GlassCard>

        <GlassCard delay={0.08} className="space-y-4">
          <h2 className="font-[family-name:var(--font-display)] text-xl text-white">
            Оверлей лица
          </h2>
          <OverlayControls toggles={toggles} onChange={setToggles} />
          <FacialOverlay imageUrl={previewUrl} result={result} toggles={toggles} />
        </GlassCard>
      </div>

      <div className="grid gap-4 md:grid-cols-2">
        {cards.map((metric, index) => (
          <MetricCard key={metric.label} metric={metric} delay={0.05 * index} />
        ))}
      </div>

      <MeasurementsPanel
        imageUrl={previewUrl}
        measurements={result.measurements ?? []}
      />

      <div className="grid gap-6 lg:grid-cols-2">
        <MetricsChart result={result} />
        <Suggestions recommendations={result.recommendations} />
      </div>

      <div className="flex flex-wrap justify-center gap-3">
        <Link
          href="/upload"
          className="rounded-full bg-white px-5 py-2.5 text-sm font-medium text-[#0b0b18]"
        >
          Другое фото
        </Link>
        <Link
          href="/history"
          className="rounded-full border border-white/15 bg-white/5 px-5 py-2.5 text-sm text-white/80"
        >
          Сравнить в истории
        </Link>
      </div>
    </div>
  );
}

/**
 * Страница результатов.
 */
export default function ResultsPage() {
  return (
    <Suspense
      fallback={
        <div className="py-20 text-center text-white/50">Загрузка отчёта…</div>
      }
    >
      <ResultsContent />
    </Suspense>
  );
}

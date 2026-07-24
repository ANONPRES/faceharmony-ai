"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ImageUploader, type AnalyzePayload } from "@/components/ImageUploader";
import { GlassCard } from "@/components/GlassCard";
import { analyzeFace, compressImageForStorage } from "@/lib/api";
import { addHistoryEntry } from "@/lib/history";
import { saveLatestAnalysis } from "@/lib/session";

/**
 * Страница загрузки фото и запуска анализа.
 */
export default function UploadPage() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  const handleAnalyze = async (payload: AnalyzePayload) => {
    setBusy(true);
    try {
      const result = await analyzeFace(payload.file, {
        gender: payload.gender,
        profileFile: payload.profileFile,
      });
      const preview = await compressImageForStorage(payload.file, {
        maxEdge: 960,
        quality: 0.58,
      });
      const thumb = await compressImageForStorage(payload.file, {
        maxEdge: 360,
        quality: 0.55,
      });
      await saveLatestAnalysis(preview, result);
      addHistoryEntry(thumb, result);
      router.push("/results");
    } catch (err) {
      console.error(err);
      alert(err instanceof Error ? err.message : "Анализ не удался.");
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="mx-auto max-w-3xl space-y-6">
      <div className="text-center">
        <h1 className="font-[family-name:var(--font-display)] text-4xl text-white sm:text-5xl">
          Анализ лица
        </h1>
        <p className="mx-auto mt-3 max-w-xl text-sm text-white/55 sm:text-base">
          Укажите пол (для идеалов метрик), загрузите анфас и при желании —
          отдельное фото профиля для настоящего profile score.
        </p>
      </div>

      <GlassCard>
        <ImageUploader onAnalyze={handleAnalyze} busy={busy} />
      </GlassCard>
    </div>
  );
}

"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ImageUploader } from "@/components/ImageUploader";
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

  const handleAnalyze = async (file: File, _previewUrl: string) => {
    setBusy(true);
    try {
      const result = await analyzeFace(file);
      // Keep storage payloads tiny — gallery shots are often 5–15MB.
      const preview = await compressImageForStorage(file, {
        maxEdge: 960,
        quality: 0.58,
      });
      const thumb = await compressImageForStorage(file, {
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
          Загрузите чёткое фото. Система определит ракурс (анфас / профиль),
          линию роста волос и посчитает образовательные метрики гармонии.
        </p>
      </div>

      <GlassCard>
        <ImageUploader onAnalyze={handleAnalyze} busy={busy} />
      </GlassCard>
    </div>
  );
}

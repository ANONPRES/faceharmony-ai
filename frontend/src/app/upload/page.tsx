"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";
import { ImageUploader } from "@/components/ImageUploader";
import { GlassCard } from "@/components/GlassCard";
import { analyzeFace, fileToDataUrl } from "@/lib/api";
import { addHistoryEntry } from "@/lib/history";
import { saveLatestAnalysis } from "@/lib/session";

/**
 * Страница загрузки фото и запуска анализа.
 */
export default function UploadPage() {
  const router = useRouter();
  const [busy, setBusy] = useState(false);

  const handleAnalyze = async (file: File, previewUrl: string) => {
    setBusy(true);
    try {
      const result = await analyzeFace(file);
      const dataUrl = previewUrl.startsWith("blob:")
        ? await fileToDataUrl(file)
        : previewUrl;
      saveLatestAnalysis(dataUrl, result);
      addHistoryEntry(dataUrl, result);
      router.push("/results");
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

/**
 * Клиентские хелперы для FastAPI.
 */

import type { AnalysisResult } from "./types";

const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") || "/fh-api";

/**
 * Отправить фото на анализ.
 */
export async function analyzeFace(file: File): Promise<AnalysisResult> {
  const form = new FormData();
  form.append("file", file);

  const response = await fetch(`${API_BASE}/analyze`, {
    method: "POST",
    body: form,
  });

  if (!response.ok) {
    let detail = "Анализ не удался. Попробуйте другое фото.";
    try {
      const payload = await response.json();
      if (typeof payload.detail === "string") {
        detail = payload.detail;
      } else if (Array.isArray(payload.detail)) {
        detail = payload.detail.map((d: { msg?: string }) => d.msg).join(", ");
      }
    } catch {
      // keep default
    }
    throw new Error(detail);
  }

  return response.json() as Promise<AnalysisResult>;
}

/**
 * File → data URL для превью и истории.
 */
export function fileToDataUrl(file: File | Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Не удалось прочитать файл."));
    reader.readAsDataURL(file);
  });
}

export type CompressOptions = {
  /** Max width/height in px. */
  maxEdge?: number;
  /** JPEG quality 0–1. */
  quality?: number;
};

/**
 * Downscale + JPEG-compress an image so it fits browser storage quotas (~5 MB).
 * Large phone photos as raw data URLs often exceed sessionStorage/localStorage.
 */
export async function compressImageForStorage(
  source: File | Blob | string,
  options: CompressOptions = {},
): Promise<string> {
  const maxEdge = options.maxEdge ?? 960;
  const quality = options.quality ?? 0.58;

  let blob: Blob;
  if (typeof source === "string") {
    const res = await fetch(source);
    blob = await res.blob();
  } else {
    blob = source;
  }

  let bitmap: ImageBitmap;
  try {
    bitmap = await createImageBitmap(blob);
  } catch {
    // Unsupported decode (rare) — last resort tiny placeholder, never full-res.
    const canvas = document.createElement("canvas");
    canvas.width = 8;
    canvas.height = 8;
    return canvas.toDataURL("image/jpeg", 0.5);
  }

  try {
    const attempts: Array<{ edge: number; q: number }> = [
      { edge: maxEdge, q: quality },
      { edge: Math.min(maxEdge, 720), q: Math.min(quality, 0.5) },
      { edge: 480, q: 0.45 },
      { edge: 320, q: 0.4 },
    ];

    let last = "";
    for (const attempt of attempts) {
      const scale = Math.min(1, attempt.edge / Math.max(bitmap.width, bitmap.height));
      const width = Math.max(1, Math.round(bitmap.width * scale));
      const height = Math.max(1, Math.round(bitmap.height * scale));
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      if (!ctx) continue;
      ctx.drawImage(bitmap, 0, 0, width, height);
      last = canvas.toDataURL("image/jpeg", attempt.q);
      // ~350KB data URL is safe for any leftover Web Storage usage.
      if (last.length < 350_000) return last;
    }
    return last;
  } finally {
    bitmap.close();
  }
}

/**
 * Валидация файла на клиенте.
 */
export function validateImageFile(file: File): string | null {
  const allowed = ["image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"];
  if (!allowed.includes(file.type)) {
    return "Загрузите JPEG, PNG, WEBP или BMP.";
  }
  if (file.size > 12 * 1024 * 1024) {
    return "Размер изображения должен быть не больше 12 МБ.";
  }
  return null;
}

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
export function fileToDataUrl(file: File): Promise<string> {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result));
    reader.onerror = () => reject(new Error("Не удалось прочитать файл."));
    reader.readAsDataURL(file);
  });
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

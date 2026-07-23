"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Camera, ImagePlus, Loader2, Upload, X } from "lucide-react";
import { motion } from "framer-motion";
import { validateImageFile } from "@/lib/api";

interface ImageUploaderProps {
  onAnalyze: (file: File, previewUrl: string) => Promise<void>;
  busy?: boolean;
}

/**
 * Загрузка фото: drag-and-drop, камера, превью.
 */
export function ImageUploader({ onAnalyze, busy = false }: ImageUploaderProps) {
  const [preview, setPreview] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragging, setDragging] = useState(false);
  const [cameraOpen, setCameraOpen] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);

  const applyFile = useCallback((next: File) => {
    const validationError = validateImageFile(next);
    if (validationError) {
      setError(validationError);
      return;
    }
    setError(null);
    setFile(next);
    const url = URL.createObjectURL(next);
    setPreview((prev) => {
      if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
      return url;
    });
  }, []);

  const onDrop = useCallback(
    (event: React.DragEvent<HTMLDivElement>) => {
      event.preventDefault();
      setDragging(false);
      const dropped = event.dataTransfer.files?.[0];
      if (dropped) applyFile(dropped);
    },
    [applyFile],
  );

  const stopCamera = useCallback(() => {
    streamRef.current?.getTracks().forEach((track) => track.stop());
    streamRef.current = null;
    setCameraOpen(false);
  }, []);

  const startCamera = async () => {
    setError(null);
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "user" },
        audio: false,
      });
      streamRef.current = stream;
      setCameraOpen(true);
    } catch {
      setError("Нет доступа к камере или она недоступна в этом браузере.");
    }
  };

  useEffect(() => {
    if (cameraOpen && videoRef.current && streamRef.current) {
      videoRef.current.srcObject = streamRef.current;
      void videoRef.current.play();
    }
  }, [cameraOpen]);

  useEffect(() => () => stopCamera(), [stopCamera]);

  const captureFrame = () => {
    const video = videoRef.current;
    if (!video) return;
    const canvas = document.createElement("canvas");
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    canvas.toBlob(
      (blob) => {
        if (!blob) return;
        const captured = new File([blob], `capture-${Date.now()}.jpg`, {
          type: "image/jpeg",
        });
        applyFile(captured);
        stopCamera();
      },
      "image/jpeg",
      0.92,
    );
  };

  const clearSelection = () => {
    setFile(null);
    setPreview((prev) => {
      if (prev?.startsWith("blob:")) URL.revokeObjectURL(prev);
      return null;
    });
    setError(null);
  };

  return (
    <div className="space-y-4">
      <div
        onDragOver={(e) => {
          e.preventDefault();
          setDragging(true);
        }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        className={`relative overflow-hidden rounded-[2rem] border border-dashed p-6 transition ${
          dragging
            ? "border-violet-300 bg-violet-500/10"
            : "border-white/15 bg-white/[0.03]"
        }`}
      >
        {!preview && !cameraOpen && (
          <div className="flex flex-col items-center gap-4 py-10 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-3xl bg-gradient-to-br from-violet-500/30 to-sky-500/30">
              <ImagePlus className="h-7 w-7 text-violet-200" />
            </div>
            <div>
              <p className="font-[family-name:var(--font-display)] text-2xl text-white">
                Перетащите фото лица
              </p>
              <p className="mt-2 max-w-md text-sm text-white/50">
                Лучше всего анфас или чистый профиль при ровном свете. JPEG, PNG,
                WEBP или BMP до 12 МБ.
              </p>
            </div>
            <div className="flex flex-wrap items-center justify-center gap-3">
              <button
                type="button"
                onClick={() => inputRef.current?.click()}
                className="inline-flex items-center gap-2 rounded-full bg-white px-5 py-2.5 text-sm font-medium text-[#0b0b18] transition hover:bg-violet-100"
              >
                <Upload className="h-4 w-4" />
                Выбрать файл
              </button>
              <button
                type="button"
                onClick={startCamera}
                className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-5 py-2.5 text-sm text-white transition hover:bg-white/10"
              >
                <Camera className="h-4 w-4" />
                Камера
              </button>
            </div>
          </div>
        )}

        {cameraOpen && (
          <div className="space-y-4">
            <video
              ref={videoRef}
              playsInline
              muted
              className="mx-auto max-h-[420px] w-full rounded-3xl object-cover"
            />
            <div className="flex justify-center gap-3">
              <button
                type="button"
                onClick={captureFrame}
                className="rounded-full bg-gradient-to-r from-violet-500 to-sky-500 px-5 py-2.5 text-sm font-medium text-white"
              >
                Сделать снимок
              </button>
              <button
                type="button"
                onClick={stopCamera}
                className="rounded-full border border-white/15 px-5 py-2.5 text-sm text-white/80"
              >
                Отмена
              </button>
            </div>
          </div>
        )}

        {preview && !cameraOpen && (
          <div className="relative">
            <motion.img
              initial={{ opacity: 0, scale: 0.98 }}
              animate={{ opacity: 1, scale: 1 }}
              src={preview}
              alt="Превью выбранного фото"
              className="mx-auto max-h-[460px] rounded-3xl object-contain"
            />
            <button
              type="button"
              onClick={clearSelection}
              className="absolute right-3 top-3 rounded-full border border-white/15 bg-black/50 p-2 text-white backdrop-blur"
              aria-label="Очистить"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        )}

        <input
          ref={inputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp,image/bmp"
          className="hidden"
          onChange={(e) => {
            const selected = e.target.files?.[0];
            if (selected) applyFile(selected);
          }}
        />
      </div>

      {error && (
        <p className="rounded-2xl border border-rose-400/30 bg-rose-500/10 px-4 py-3 text-sm text-rose-200">
          {error}
        </p>
      )}

      <button
        type="button"
        disabled={!file || busy}
        onClick={async () => {
          if (!file || !preview) return;
          try {
            await onAnalyze(file, preview);
          } catch (err) {
            setError(err instanceof Error ? err.message : "Ошибка анализа.");
          }
        }}
        className="flex w-full items-center justify-center gap-2 rounded-full bg-gradient-to-r from-violet-500 via-indigo-500 to-sky-500 px-6 py-3.5 text-sm font-semibold text-white shadow-lg shadow-violet-500/25 transition enabled:hover:brightness-110 disabled:cursor-not-allowed disabled:opacity-40"
      >
        {busy ? (
          <>
            <Loader2 className="h-4 w-4 animate-spin" />
            Считаем геометрию лица…
          </>
        ) : (
          <>
            <Upload className="h-4 w-4" />
            Анализировать лицо
          </>
        )}
      </button>
    </div>
  );
}

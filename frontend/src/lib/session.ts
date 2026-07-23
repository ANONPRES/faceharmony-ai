/**
 * Persist the latest analysis outside the tiny sessionStorage quota.
 * Gallery photos as data URLs routinely exceed ~5MB Web Storage limits.
 */

import type { AnalysisResult } from "./types";

const DB_NAME = "faceharmony";
const DB_VERSION = 1;
const STORE = "latest";
const RECORD_ID = "current";

const RESULT_KEY = "faceharmony.latestResult";
const IMAGE_KEY = "faceharmony.latestImage";

type LatestRecord = {
  id: string;
  imageDataUrl: string;
  result: AnalysisResult;
};

let memory: { imageDataUrl: string; result: AnalysisResult } | null = null;

function isQuotaError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { name?: string; code?: number };
  return (
    e.name === "QuotaExceededError" ||
    e.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
    e.code === 22
  );
}

function openDb(): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onerror = () => reject(req.error ?? new Error("indexedDB open failed"));
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE, { keyPath: "id" });
      }
    };
    req.onsuccess = () => resolve(req.result);
  });
}

async function idbPut(record: LatestRecord): Promise<void> {
  const db = await openDb();
  await new Promise<void>((resolve, reject) => {
    const tx = db.transaction(STORE, "readwrite");
    tx.oncomplete = () => resolve();
    tx.onerror = () => reject(tx.error ?? new Error("indexedDB write failed"));
    tx.objectStore(STORE).put(record);
  });
  db.close();
}

async function idbGet(): Promise<LatestRecord | null> {
  const db = await openDb();
  const record = await new Promise<LatestRecord | null>((resolve, reject) => {
    const tx = db.transaction(STORE, "readonly");
    const req = tx.objectStore(STORE).get(RECORD_ID);
    req.onsuccess = () => resolve((req.result as LatestRecord | undefined) ?? null);
    req.onerror = () => reject(req.error ?? new Error("indexedDB read failed"));
  });
  db.close();
  return record;
}

/** Drop heavy arrays that are not needed for history list cards. */
export function slimResultForHistory(result: AnalysisResult): AnalysisResult {
  return {
    ...result,
    landmarks: [],
    measurements: (result.measurements ?? []).map((m) => ({
      ...m,
      points: [],
      segments: [],
    })),
  };
}

/**
 * Store the latest analysis for the results page.
 * Prefer memory + IndexedDB; sessionStorage is a tiny optional fallback.
 */
export async function saveLatestAnalysis(
  imageDataUrl: string,
  result: AnalysisResult,
): Promise<void> {
  memory = { imageDataUrl, result };

  try {
    await idbPut({ id: RECORD_ID, imageDataUrl, result });
  } catch {
    // IndexedDB unavailable (private mode quirks) — fall through.
  }

  // Best-effort tiny session fallback for hard refresh without IDB.
  try {
    sessionStorage.removeItem(IMAGE_KEY);
    sessionStorage.removeItem(RESULT_KEY);
    // Result alone can already be large; skip image in sessionStorage.
    sessionStorage.setItem(RESULT_KEY, JSON.stringify(result));
  } catch (err) {
    if (!isQuotaError(err)) {
      // ignore non-quota write issues for optional fallback
    }
    try {
      sessionStorage.removeItem(RESULT_KEY);
      sessionStorage.removeItem(IMAGE_KEY);
    } catch {
      // ignore
    }
  }
}

/**
 * Load the latest analysis (memory → IndexedDB → sessionStorage).
 */
export async function loadLatestAnalysis(): Promise<{
  imageDataUrl: string | null;
  result: AnalysisResult | null;
}> {
  if (typeof window === "undefined") {
    return { imageDataUrl: null, result: null };
  }

  if (memory) {
    return { imageDataUrl: memory.imageDataUrl, result: memory.result };
  }

  try {
    const record = await idbGet();
    if (record?.result) {
      memory = { imageDataUrl: record.imageDataUrl, result: record.result };
      return { imageDataUrl: record.imageDataUrl, result: record.result };
    }
  } catch {
    // fall through
  }

  const imageDataUrl = sessionStorage.getItem(IMAGE_KEY);
  const raw = sessionStorage.getItem(RESULT_KEY);
  if (!raw) return { imageDataUrl, result: null };
  try {
    return { imageDataUrl, result: JSON.parse(raw) as AnalysisResult };
  } catch {
    return { imageDataUrl, result: null };
  }
}

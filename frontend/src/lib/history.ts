/**
 * LocalStorage-backed analysis history for comparison.
 * Stores compressed thumbs + slim results to stay under quota.
 */

import type { AnalysisResult, HistoryEntry } from "./types";
import { slimResultForHistory } from "./session";

const STORAGE_KEY = "faceharmony.history.v8";
const MAX_ENTRIES = 8;

function isQuotaError(err: unknown): boolean {
  if (!err || typeof err !== "object") return false;
  const e = err as { name?: string; code?: number };
  return (
    e.name === "QuotaExceededError" ||
    e.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
    e.code === 22
  );
}

/**
 * Read all saved analysis history entries (newest first).
 */
export function loadHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw =
      localStorage.getItem(STORAGE_KEY) ??
      localStorage.getItem("faceharmony.history.v7") ??
      localStorage.getItem("faceharmony.history.v6");
    if (!raw) return [];
    const parsed = JSON.parse(raw) as HistoryEntry[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/**
 * Persist history entries to localStorage, trimming until it fits quota.
 */
function saveHistory(entries: HistoryEntry[]): void {
  let next = entries.slice(0, MAX_ENTRIES);
  while (next.length > 0) {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
      localStorage.removeItem("faceharmony.history.v7");
      localStorage.removeItem("faceharmony.history.v6");
      return;
    } catch (err) {
      if (!isQuotaError(err)) return;
      if (next.length > 1) {
        next = next.slice(0, next.length - 1);
        continue;
      }
      const stripped = next.map((entry) => ({ ...entry, imageDataUrl: "" }));
      try {
        localStorage.setItem(STORAGE_KEY, JSON.stringify(stripped));
      } catch {
        localStorage.removeItem(STORAGE_KEY);
      }
      return;
    }
  }
  localStorage.removeItem(STORAGE_KEY);
}

/**
 * Append a new analysis to history and return the created entry.
 */
export function addHistoryEntry(
  imageDataUrl: string,
  result: AnalysisResult,
): HistoryEntry {
  const entry: HistoryEntry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    createdAt: new Date().toISOString(),
    imageDataUrl,
    result: slimResultForHistory(result),
  };
  const next = [entry, ...loadHistory()].slice(0, MAX_ENTRIES);
  saveHistory(next);
  return entry;
}

/**
 * Remove one history entry by id.
 */
export function removeHistoryEntry(id: string): void {
  saveHistory(loadHistory().filter((entry) => entry.id !== id));
}

/**
 * Clear the entire local analysis history.
 */
export function clearHistory(): void {
  localStorage.removeItem(STORAGE_KEY);
  localStorage.removeItem("faceharmony.history.v7");
  localStorage.removeItem("faceharmony.history.v6");
}

/**
 * Find a history entry by id.
 */
export function getHistoryEntry(id: string): HistoryEntry | undefined {
  return loadHistory().find((entry) => entry.id === id);
}

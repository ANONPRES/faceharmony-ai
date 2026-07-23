/**
 * LocalStorage-backed analysis history for comparison.
 */

import type { AnalysisResult, HistoryEntry } from "./types";

const STORAGE_KEY = "faceharmony.history.v6";
const MAX_ENTRIES = 20;

/**
 * Read all saved analysis history entries (newest first).
 */
export function loadHistory(): HistoryEntry[] {
  if (typeof window === "undefined") return [];
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw) as HistoryEntry[];
    return Array.isArray(parsed) ? parsed : [];
  } catch {
    return [];
  }
}

/**
 * Persist history entries to localStorage.
 *
 * @param entries - Full history list to store.
 */
function saveHistory(entries: HistoryEntry[]): void {
  localStorage.setItem(STORAGE_KEY, JSON.stringify(entries.slice(0, MAX_ENTRIES)));
}

/**
 * Append a new analysis to history and return the created entry.
 *
 * @param imageDataUrl - Preview image as a data URL.
 * @param result - Analysis payload from the API.
 */
export function addHistoryEntry(
  imageDataUrl: string,
  result: AnalysisResult,
): HistoryEntry {
  const entry: HistoryEntry = {
    id: `${Date.now()}-${Math.random().toString(36).slice(2, 8)}`,
    createdAt: new Date().toISOString(),
    imageDataUrl,
    result,
  };
  const next = [entry, ...loadHistory()].slice(0, MAX_ENTRIES);
  saveHistory(next);
  return entry;
}

/**
 * Remove one history entry by id.
 *
 * @param id - Entry identifier.
 */
export function removeHistoryEntry(id: string): void {
  saveHistory(loadHistory().filter((entry) => entry.id !== id));
}

/**
 * Clear the entire local analysis history.
 */
export function clearHistory(): void {
  localStorage.removeItem(STORAGE_KEY);
}

/**
 * Find a history entry by id.
 *
 * @param id - Entry identifier.
 */
export function getHistoryEntry(id: string): HistoryEntry | undefined {
  return loadHistory().find((entry) => entry.id === id);
}

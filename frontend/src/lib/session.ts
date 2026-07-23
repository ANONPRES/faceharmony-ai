/**
 * Session helpers for passing the latest analysis between pages.
 */

import type { AnalysisResult } from "./types";

const RESULT_KEY = "faceharmony.latestResult";
const IMAGE_KEY = "faceharmony.latestImage";

/**
 * Store the latest analysis and source image for the results page.
 */
export function saveLatestAnalysis(imageDataUrl: string, result: AnalysisResult): void {
  sessionStorage.setItem(RESULT_KEY, JSON.stringify(result));
  sessionStorage.setItem(IMAGE_KEY, imageDataUrl);
}

/**
 * Load the latest analysis from session storage.
 */
export function loadLatestAnalysis(): {
  imageDataUrl: string | null;
  result: AnalysisResult | null;
} {
  if (typeof window === "undefined") {
    return { imageDataUrl: null, result: null };
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

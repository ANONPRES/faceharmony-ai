"""Shared scoring helpers for facial metrics."""

from __future__ import annotations

import numpy as np


def distance(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Euclidean distance between two 2D points."""
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    """Midpoint between two points."""
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def soft_score(value: float, ideal: float, sigma: float) -> float:
    """Gaussian-like 0–100 score peaked at ideal."""
    if sigma <= 0:
        return 0.0
    z = (value - ideal) / sigma
    return float(np.clip(100.0 * np.exp(-0.5 * z * z), 0.0, 100.0))


def range_score(
    value: float,
    ideal_min: float,
    ideal_max: float,
    soft_margin: float | None = None,
) -> float:
    """
    0–100 score peaked at the midpoint of the ideal band.

    10/10 only near the center of [ideal_min, ideal_max]. Edges of the band
    score lower; outside falls off further. (Flat 100 across the whole band
    was too generous.)
    """
    lo, hi = float(ideal_min), float(ideal_max)
    if hi < lo:
        lo, hi = hi, lo
    center = 0.5 * (lo + hi)
    half = max(0.5 * (hi - lo), 1e-6)
    # At band edge (~1 half-width away) ≈ 7.5/10; outside continues to drop.
    sigma = soft_margin if soft_margin is not None else half * 1.15
    return soft_score(value, center, max(sigma, 1e-6))


def combine_scores(parts: list[tuple[float, float]]) -> float:
    """Weighted average of (score, weight) pairs."""
    total_w = sum(w for _, w in parts) or 1.0
    return float(sum(s * w for s, w in parts) / total_w)


def power_mean(parts: list[tuple[float, float]], p: float = 0.65) -> float:
    """Weighted power mean; p < 1 penalizes weak metrics."""
    if not parts:
        return 0.0
    total_w = sum(w for _, w in parts) or 1.0
    acc = 0.0
    for score, weight in parts:
        acc += weight * (max(score, 1.0) ** p)
    return float(acc / total_w) ** (1.0 / p)

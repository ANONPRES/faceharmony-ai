"use client";

import { motion, useMotionValue, useTransform, animate } from "framer-motion";
import { useEffect } from "react";

interface ScoreRingProps {
  score: number;
  size?: number;
  label?: string;
}

/**
 * Animated circular score indicator for the overall harmony value.
 */
export function ScoreRing({ score, size = 220, label = "Общая гармония" }: ScoreRingProps) {
  const progress = useMotionValue(0);
  // One decimal — do not Math.round to a fake 100 when backend is 99.0/99.5.
  const display = useTransform(progress, (v) => (Math.round(v * 10) / 10).toFixed(1));
  const radius = (size - 18) / 2;
  const circumference = 2 * Math.PI * radius;

  useEffect(() => {
    const controls = animate(progress, score, {
      duration: 1.4,
      ease: [0.22, 1, 0.36, 1],
    });
    return controls.stop;
  }, [score, progress]);

  const stroke = useTransform(progress, (v) => {
    const offset = circumference - (v / 100) * circumference;
    return offset;
  });

  return (
    <div className="relative mx-auto flex flex-col items-center" style={{ width: size, height: size }}>
      <svg width={size} height={size} className="-rotate-90">
        <circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="rgba(255,255,255,0.08)"
          strokeWidth="12"
          fill="none"
        />
        <motion.circle
          cx={size / 2}
          cy={size / 2}
          r={radius}
          stroke="url(#scoreGradient)"
          strokeWidth="12"
          strokeLinecap="round"
          fill="none"
          strokeDasharray={circumference}
          style={{ strokeDashoffset: stroke }}
        />
        <defs>
          <linearGradient id="scoreGradient" x1="0%" y1="0%" x2="100%" y2="100%">
            <stop offset="0%" stopColor="#8b5cf6" />
            <stop offset="50%" stopColor="#6366f1" />
            <stop offset="100%" stopColor="#38bdf8" />
          </linearGradient>
        </defs>
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center text-center">
        <motion.span className="font-[family-name:var(--font-display)] text-5xl text-white sm:text-6xl">
          {display}
        </motion.span>
        <span className="mt-1 text-sm text-white/50">/ 100</span>
        <span className="mt-2 text-xs uppercase tracking-[0.2em] text-violet-200/80">
          {label}
        </span>
      </div>
    </div>
  );
}

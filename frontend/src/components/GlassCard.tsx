"use client";

import { motion } from "framer-motion";
import type { ReactNode } from "react";

interface GlassCardProps {
  children: ReactNode;
  className?: string;
  delay?: number;
}

/**
 * Reusable glassmorphism card with a soft entrance animation.
 */
export function GlassCard({ children, className = "", delay = 0 }: GlassCardProps) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.45, delay, ease: [0.22, 1, 0.36, 1] }}
      className={`rounded-3xl border border-white/10 bg-white/5 p-5 shadow-[0_20px_60px_rgba(0,0,0,0.35)] backdrop-blur-xl sm:p-6 ${className}`}
    >
      {children}
    </motion.div>
  );
}

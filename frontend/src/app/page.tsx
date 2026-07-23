"use client";

import Link from "next/link";
import { motion } from "framer-motion";
import { ArrowRight, ScanFace, ShieldCheck, Sparkles, Ruler } from "lucide-react";

/**
 * Лендинг FaceHarmony AI.
 */
export default function HomePage() {
  return (
    <div className="relative overflow-hidden pb-16">
      <motion.div
        initial={{ opacity: 0, y: 24 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: [0.22, 1, 0.36, 1] }}
        className="relative z-10 mx-auto max-w-4xl pt-8 text-center sm:pt-16"
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ delay: 0.15, duration: 0.5 }}
          className="mx-auto mb-8 flex h-20 w-20 items-center justify-center rounded-[1.75rem] bg-gradient-to-br from-violet-500 to-sky-500 shadow-[0_0_60px_rgba(139,92,246,0.45)]"
        >
          <ScanFace className="h-10 w-10 text-white" />
        </motion.div>

        <h1 className="font-[family-name:var(--font-display)] text-5xl leading-[1.05] tracking-tight text-white sm:text-7xl">
          FaceHarmony
          <span className="block bg-gradient-to-r from-violet-300 via-indigo-200 to-sky-300 bg-clip-text text-transparent">
            AI
          </span>
        </h1>

        <p className="mx-auto mt-6 max-w-2xl text-base leading-relaxed text-white/60 sm:text-lg">
          Образовательный анализ геометрии лица по фото — пропорции, симметрия и
          баланс. Это не оценка красоты.
        </p>

        <div className="mt-10 flex flex-wrap items-center justify-center gap-4">
          <Link
            href="/upload"
            className="group inline-flex items-center gap-2 rounded-full bg-gradient-to-r from-violet-500 via-indigo-500 to-sky-500 px-7 py-3.5 text-sm font-semibold text-white shadow-lg shadow-violet-500/30 transition hover:brightness-110"
          >
            Анализировать лицо
            <ArrowRight className="h-4 w-4 transition group-hover:translate-x-0.5" />
          </Link>
          <Link
            href="/history"
            className="inline-flex items-center gap-2 rounded-full border border-white/15 bg-white/5 px-7 py-3.5 text-sm text-white/80 backdrop-blur transition hover:bg-white/10"
          >
            История
          </Link>
        </div>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.35, duration: 0.6 }}
        className="mt-20 grid gap-4 sm:grid-cols-3"
      >
        {[
          {
            icon: Ruler,
            title: "Измеримая геометрия",
            body: "Скулы, вырез глаз, нос, форма лица, челюсть — отдельно для анфаса и профиля.",
          },
          {
            icon: Sparkles,
            title: "Отчёт о гармонии",
            body: "Взвешенный overall, карточки метрик, оверлей landmark’ов и сравнение в истории.",
          },
          {
            icon: ShieldCheck,
            title: "Только образование",
            body: "Нейтральные формулировки. Без вердиктов о красоте или привлекательности.",
          },
        ].map((item, index) => (
          <motion.div
            key={item.title}
            initial={{ opacity: 0, y: 18 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.45 + index * 0.1 }}
            className="rounded-3xl border border-white/10 bg-white/[0.04] p-5 backdrop-blur-xl"
          >
            <item.icon className="mb-3 h-5 w-5 text-sky-300" />
            <h2 className="font-[family-name:var(--font-display)] text-lg text-white">
              {item.title}
            </h2>
            <p className="mt-2 text-sm leading-relaxed text-white/55">{item.body}</p>
          </motion.div>
        ))}
      </motion.div>

      <motion.div
        aria-hidden
        animate={{ y: [0, -12, 0] }}
        transition={{ duration: 7, repeat: Infinity, ease: "easeInOut" }}
        className="pointer-events-none absolute -left-10 top-24 h-40 w-40 rounded-full bg-violet-500/20 blur-3xl"
      />
      <motion.div
        aria-hidden
        animate={{ y: [0, 16, 0] }}
        transition={{ duration: 9, repeat: Infinity, ease: "easeInOut" }}
        className="pointer-events-none absolute -right-8 top-40 h-48 w-48 rounded-full bg-sky-500/20 blur-3xl"
      />
    </div>
  );
}

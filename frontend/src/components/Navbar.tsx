"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { History, ScanFace, Sparkles, Upload } from "lucide-react";

const links = [
  { href: "/", label: "Главная", icon: Sparkles },
  { href: "/upload", label: "Анализ", icon: Upload },
  { href: "/history", label: "История", icon: History },
];

/**
 * Верхняя навигация с активным маршрутом.
 */
export function Navbar() {
  const pathname = usePathname();

  return (
    <header className="sticky top-0 z-50 border-b border-white/10 bg-[#070714]/70 backdrop-blur-xl">
      <div className="mx-auto flex max-w-6xl items-center justify-between px-4 py-3 sm:px-6">
        <Link href="/" className="group flex items-center gap-2.5">
          <span className="flex h-9 w-9 items-center justify-center rounded-2xl bg-gradient-to-br from-violet-500 to-blue-500 shadow-lg shadow-violet-500/30">
            <ScanFace className="h-5 w-5 text-white" />
          </span>
          <span className="font-[family-name:var(--font-display)] text-lg tracking-tight text-white sm:text-xl">
            FaceHarmony
            <span className="bg-gradient-to-r from-violet-300 to-sky-300 bg-clip-text text-transparent">
              {" "}
              AI
            </span>
          </span>
        </Link>

        <nav className="flex items-center gap-1 rounded-full border border-white/10 bg-white/5 p-1">
          {links.map(({ href, label, icon: Icon }) => {
            const active = pathname === href;
            return (
              <Link
                key={href}
                href={href}
                className={`flex items-center gap-1.5 rounded-full px-3 py-1.5 text-sm transition ${
                  active
                    ? "bg-white/15 text-white shadow-inner"
                    : "text-white/60 hover:bg-white/5 hover:text-white"
                }`}
              >
                <Icon className="h-3.5 w-3.5" />
                <span className="hidden sm:inline">{label}</span>
              </Link>
            );
          })}
        </nav>
      </div>
    </header>
  );
}

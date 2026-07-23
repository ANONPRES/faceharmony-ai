import type { Metadata } from "next";
import { Outfit, Syne } from "next/font/google";
import { Navbar } from "@/components/Navbar";
import "./globals.css";

const body = Outfit({
  variable: "--font-body",
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700"],
});

const display = Syne({
  variable: "--font-display",
  subsets: ["latin"],
  weight: ["500", "600", "700", "800"],
});

export const metadata: Metadata = {
  title: "FaceHarmony AI — анализ геометрии лица",
  description:
    "Образовательный анализ пропорций, симметрии и гармонии лица по фото. Анфас и профиль. Не оценка красоты.",
};

/**
 * Root application layout with dark glass theme and site navigation.
 */
export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="ru">
      <body className={`${body.variable} ${display.variable} antialiased`}>
        <Navbar />
        <main className="mx-auto min-h-[calc(100vh-4rem)] max-w-6xl px-4 py-8 sm:px-6 sm:py-10">
          {children}
        </main>
      </body>
    </html>
  );
}

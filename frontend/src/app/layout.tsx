import type { Metadata, Viewport } from "next";
import { connection } from "next/server";
import { IBM_Plex_Mono, IBM_Plex_Sans, Newsreader } from "next/font/google";
import "./globals.css";

// next/font downloads these at build time and serves them from this site, so no page ever asks
// Google for a font: nothing to allow in the CSP, and no visitor data goes to a third party.
const plexSans = IBM_Plex_Sans({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  variable: "--font-plex-sans",
  display: "swap",
});
const plexMono = IBM_Plex_Mono({
  subsets: ["latin"],
  weight: ["400", "500"],
  variable: "--font-plex-mono",
  display: "swap",
});
const newsreader = Newsreader({
  subsets: ["latin"],
  weight: ["400", "500", "600"],
  style: ["normal", "italic"],
  variable: "--font-newsreader",
  display: "swap",
});

export const metadata: Metadata = {
  title: { default: "WikiGuard", template: "%s · WikiGuard" },
  description:
    "Vandalism triage for English Wikipedia: edits scored by a neural model, the riskiest first, and a person always decides.",
  applicationName: "WikiGuard",
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f8f6f1" },
    { media: "(prefers-color-scheme: dark)", color: "#16181d" },
  ],
};

export default async function RootLayout({ children }: LayoutProps<"/">) {
  // Render every page per request, so each one carries that request's CSP nonce (DEC-065).
  await connection();
  return (
    <html
      lang="en"
      className={`${plexSans.variable} ${plexMono.variable} ${newsreader.variable} h-full`}
    >
      <body className="min-h-full">{children}</body>
    </html>
  );
}

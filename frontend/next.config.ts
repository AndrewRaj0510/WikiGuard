import type { NextConfig } from "next";
import { STATIC_SECURITY_HEADERS } from "./src/lib/security/headers";

/**
 * The browser only ever calls `/api/...` on the site's own address (PLAN section 16, F-035), so the
 * session cookie is first-party and needs no CORS. In production the Cloudflare Worker sends
 * `/api/*` to Render and strips the prefix. Locally, and on a Vercel preview with no Worker, this
 * rewrite does the same job. `API_ORIGIN` is read on the server only and never reaches the bundle.
 */
const apiOrigin = process.env.API_ORIGIN?.replace(/\/+$/, "");

const nextConfig: NextConfig = {
  // Every page renders per request, so each gets a fresh CSP nonce (DEC-065, src/proxy.ts).
  // cacheComponents and partial prefetching are off: they exist to make pages static.
  reactCompiler: true,
  poweredByHeader: false,
  turbopack: {
    rules: {
      "*.css": {
        loaders: ["@tailwindcss/turbopack"],
        as: "*.css",
      },
    },
  },
  async headers() {
    return [{ source: "/:path*", headers: STATIC_SECURITY_HEADERS }];
  },
  async rewrites() {
    if (!apiOrigin) return [];
    return [{ source: "/api/:path*", destination: `${apiOrigin}/:path*` }];
  },
};

export default nextConfig;

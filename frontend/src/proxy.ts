import { NextResponse, type NextRequest } from "next/server";
import { contentSecurityPolicy, INDEXABLE_PATHS, newNonce } from "@/lib/security/headers";

/**
 * Runs before each page: a fresh nonce, the CSP that names it, and `noindex` for pages that can
 * show vandal text. Next.js reads the nonce from the request's CSP header while it renders and
 * puts it on its own scripts and styles. `/api`, static files, and prefetches are skipped.
 */
export function proxy(request: NextRequest) {
  const nonce = newNonce();
  const forwarded = request.headers.get("x-forwarded-proto");
  const https = (forwarded ?? request.nextUrl.protocol.replace(":", "")) === "https";
  const csp = contentSecurityPolicy({ nonce, dev: process.env.NODE_ENV === "development", https });

  const requestHeaders = new Headers(request.headers);
  requestHeaders.set("x-nonce", nonce);
  requestHeaders.set("content-security-policy", csp);

  const response = NextResponse.next({ request: { headers: requestHeaders } });
  response.headers.set("content-security-policy", csp);
  if (!INDEXABLE_PATHS.has(request.nextUrl.pathname)) {
    response.headers.set("x-robots-tag", "noindex, nofollow");
  }
  return response;
}

export const config = {
  matcher: [
    {
      // Only files that exist are skipped. A skipped path with no file behind it falls through to
      // the 404 page, which would then go out with no CSP: /favicon.ico and /robots.txt both did
      // until OWASP ZAP's baseline found it (Phase 6). The icon is icon.svg, and robots.txt comes
      // from app/robots.ts.
      source: "/((?!api/|_next/static|_next/image|icon.svg|robots.txt).*)",
      missing: [
        { type: "header", key: "next-router-prefetch" },
        { type: "header", key: "purpose", value: "prefetch" },
      ],
    },
  ],
};

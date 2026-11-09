/**
 * The site's security headers, in one place so tests can read them (PLAN section 8).
 *
 * The CSP uses a fresh nonce per request (DEC-065). Next.js puts inline scripts on every page for
 * its page data, so a strict `script-src` needs either a nonce or a hash of each one. Hashes would
 * keep pages static, but they can't cover what's streamed per request, and SRI alone blocked every
 * inline script in a test (F-114). So every page renders per request, which `proxy.ts` arranges.
 */

export interface CspOptions {
  nonce: string;
  /** `next dev` needs eval for React's debugging, and a WebSocket for hot reload. */
  dev: boolean;
  /** Only over HTTPS: on plain-http localhost it would send every asset to https and break. */
  https: boolean;
}

export function contentSecurityPolicy({ nonce, dev, https }: CspOptions): string {
  const directives: [string, string][] = [
    ["default-src", "'self'"],
    // 'strict-dynamic': scripts the nonced ones load are trusted too, and nothing else is.
    ["script-src", `'self' 'nonce-${nonce}' 'strict-dynamic'${dev ? " 'unsafe-eval'" : ""}`],
    ["style-src", `'self' 'nonce-${nonce}'`],
    ["img-src", "'self' data: blob:"],
    ["font-src", "'self'"],
    // The browser only talks to its own address: the API is at /api (DEC-016).
    ["connect-src", dev ? "'self' ws:" : "'self'"],
    ["object-src", "'none'"],
    ["base-uri", "'none'"],
    ["form-action", "'self'"],
    ["frame-ancestors", "'none'"],
  ];
  const policy = directives.map(([name, value]) => `${name} ${value}`).join("; ");
  return https ? `${policy}; upgrade-insecure-requests` : policy;
}

/** A nonce of 128 random bits. A UUID's version bits would leave 122. */
export function newNonce(): string {
  const bytes = new Uint8Array(16);
  crypto.getRandomValues(bytes);
  return btoa(String.fromCharCode(...bytes));
}

/** Pages anyone may find in a search engine. Everything else can show vandal text. */
export const INDEXABLE_PATHS = new Set(["/", "/privacy"]);

/** Sent with every response, including static files (next.config.ts). */
export const STATIC_SECURITY_HEADERS: { key: string; value: string }[] = [
  { key: "X-Content-Type-Options", value: "nosniff" },
  { key: "Referrer-Policy", value: "no-referrer" },
  // Older browsers that ignore frame-ancestors.
  { key: "X-Frame-Options", value: "DENY" },
  { key: "Cross-Origin-Opener-Policy", value: "same-origin" },
  // Two years. Browsers ignore it over plain http, so local runs are unaffected.
  { key: "Strict-Transport-Security", value: "max-age=63072000; includeSubDomains" },
  {
    key: "Permissions-Policy",
    value:
      "camera=(), microphone=(), geolocation=(), payment=(), usb=(), browsing-topics=(), interest-cohort=()",
  },
];

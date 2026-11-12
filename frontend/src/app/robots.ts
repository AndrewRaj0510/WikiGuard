import type { MetadataRoute } from "next";

/**
 * Served as /robots.txt. Without this file the path fell through to the 404 page, and the proxy
 * skips that path, so the page went out with no Content-Security-Policy. OWASP ZAP's baseline
 * found it in Phase 6.
 *
 * Console pages aren't listed as disallowed on purpose. They send `noindex` (src/proxy.ts), and
 * a crawler has to be allowed to fetch a page to see that. Blocking them here could leave a
 * linked page in search results with no description. Only /api is closed: crawling it would
 * spend the API's free quota for nothing.
 */
export default function robots(): MetadataRoute.Robots {
  return {
    rules: { userAgent: "*", allow: "/", disallow: "/api/" },
  };
}

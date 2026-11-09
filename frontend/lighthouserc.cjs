/**
 * Lighthouse CI budgets (PLAN section 10). It audits the pages a visitor without a session sees:
 * the landing page, the privacy notice, and the console's sign-in screen. Run after a build:
 *   npx lhci autorun
 * SEO is asserted on the public pages only: console pages send noindex on purpose.
 */
const PORT = 3220;
const url = (path) => `http://localhost:${PORT}${path}`;

module.exports = {
  ci: {
    collect: {
      startServerCommand: `npm run start -- -p ${PORT}`,
      startServerReadyPattern: "Ready",
      url: [url("/"), url("/privacy"), url("/queue")],
      numberOfRuns: 3,
      settings: { preset: "desktop" },
    },
    assert: {
      assertMatrix: [
        {
          matchingUrlPattern: ".*",
          assertions: {
            "categories:performance": ["error", { minScore: 0.9, aggregationMethod: "median-run" }],
            "categories:accessibility": ["error", { minScore: 1 }],
            "categories:best-practices": ["error", { minScore: 0.95 }],
            // The same JavaScript budget as scripts/check-bundle.mjs, in bytes, uncompressed
            // transfer as Lighthouse counts it.
            "resource-summary:script:size": ["error", { maxNumericValue: 260_000 }],
            "csp-xss": "off",
          },
        },
        {
          matchingUrlPattern: "^http://localhost:\\d+/(privacy)?$",
          assertions: { "categories:seo": ["error", { minScore: 1 }] },
        },
      ],
    },
    upload: { target: "filesystem", outputDir: ".lighthouseci" },
  },
};

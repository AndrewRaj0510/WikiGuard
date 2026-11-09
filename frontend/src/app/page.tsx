import Link from "next/link";
import { getImageProps } from "next/image";
import { ArrowRight, KeyRound, PlayCircle, UserRound } from "lucide-react";
import { Wordmark } from "@/components/logo";
import { RiskGlyph } from "@/components/risk";
import { SeatStatus } from "@/components/landing/seat-status";
import { SiteFooter, REPO_URL } from "@/components/site-footer";
import { buttonVariants } from "@/components/ui/button";
import { EXTERNAL_REL } from "@/lib/wikipedia";
import { cn } from "@/lib/utils";

/*
 * Public and never gated (PLAN section 16): a visitor who finds the app in use still sees all of
 * this. Measured numbers only (docs/metrics.md, test split, 2026-11-07).
 */
const FACTS = [
  { value: "5", label: "neural models compared, from an MLP to RoBERTa" },
  { value: "0.364", label: "test PR-AUC for the RoBERTa champion, against 0.187 for a rule baseline" },
  { value: "48 h", label: "until a community revert turns an edit into a label" },
];

const STEPS = [
  {
    band: "low" as const,
    title: "Every edit, as it happens",
    text: "Wikimedia's public stream feeds each English Wikipedia article edit through a fallback chain of models.",
  },
  {
    band: "elevated" as const,
    title: "The riskiest first",
    text: "Patrollers get a queue ranked by the model's calibrated chance of a revert, with a plain-language Explain button.",
  },
  {
    band: "flagged" as const,
    title: "A person decides",
    text: "WikiGuard never reverts anything. Community reverts become its labels, the basis for measuring live accuracy and for retraining.",
  },
];

// Built so far. Monitoring, retraining, and the cluster are listed as next, not as done.
const BUILT = ["PyTorch → ONNX int8", "Redpanda", "FastAPI", "Postgres", "MLflow + DVC", "Next.js"];
const NEXT = ["Evidently drift checks", "Automated retraining", "k3s + Argo CD"];

export default function LandingPage() {
  return (
    <div className="flex min-h-dvh flex-col">
      <div className="mx-auto w-full max-w-6xl flex-1 px-4">
        <header className="flex h-16 items-center justify-between">
          <Wordmark />
          <nav aria-label="Site" className="flex items-center gap-1">
            <a href={REPO_URL} rel={EXTERNAL_REL} className={buttonVariants({ variant: "ghost", size: "sm" })}>
              Source
            </a>
            <Link href="/queue" className={buttonVariants({ variant: "outline", size: "sm" })}>
              Open the console
            </Link>
          </nav>
        </header>

        <main>
          <section className="grid gap-12 pt-14 pb-14 sm:pt-20 lg:grid-cols-[minmax(0,1fr)_26rem] lg:items-end">
            <div className="max-w-3xl animate-rise">
              <p className="text-[11px] font-medium tracking-[0.16em] text-risk-flagged uppercase">
                Live vandalism triage for English Wikipedia
              </p>
              <h1 className="mt-4 font-display text-[44px] leading-[1.02] font-medium tracking-[-0.02em] sm:text-[64px]">
                Every edit scored.
                <br />
                <span className="italic">The worst</span> ones first.
              </h1>
              <p className="mt-6 max-w-xl text-[17px] leading-relaxed text-muted-foreground">
                WikiGuard reads every article edit from Wikipedia&rsquo;s public stream, scores it
                with a deep learning model, and puts the riskiest at the top of a review queue for
                human patrollers.
              </p>
            </div>
            <SignInOptions />
          </section>

          <section aria-label="Measured results" className="grid gap-px overflow-hidden rounded-lg border bg-border sm:grid-cols-3">
            {FACTS.map((fact) => (
              <div key={fact.value} className="bg-card px-6 py-5">
                <p className="font-mono text-[30px] leading-none font-medium tracking-tight tabular">
                  {fact.value}
                </p>
                <p className="mt-2 text-[13.5px] leading-snug text-muted-foreground">{fact.label}</p>
              </div>
            ))}
          </section>

          <Screenshots />

          <section aria-label="How it works" className="grid gap-8 py-16 sm:grid-cols-3">
            {STEPS.map((step, index) => (
              <div key={step.title} className="border-t pt-5">
                <p className="flex items-center gap-2 font-mono text-[12px] text-muted-foreground">
                  <RiskGlyph
                    band={step.band}
                    className={cn(
                      step.band === "flagged" && "text-risk-flagged",
                      step.band === "elevated" && "text-risk-elevated",
                      step.band === "low" && "text-risk-low",
                    )}
                  />
                  0{index + 1}
                </p>
                <h2 className="mt-3 font-display text-[22px] leading-tight font-medium">{step.title}</h2>
                <p className="mt-2 text-[14.5px] leading-relaxed text-muted-foreground">{step.text}</p>
              </div>
            ))}
          </section>

          <section aria-labelledby="demo-heading" className="grid gap-8 border-t py-14 lg:grid-cols-2">
            <div>
              <h2 id="demo-heading" className="font-display text-[26px] font-medium">
                See it in two minutes
              </h2>
              <p className="mt-2 max-w-md text-[14.5px] leading-relaxed text-muted-foreground">
                A short walkthrough video is on its way. Until then, the screenshots above show
                the review flow, and the console is open to try as a guest.
              </p>
            </div>
            <div className="flex aspect-video items-center justify-center rounded-lg border border-dashed bg-muted/50 text-muted-foreground">
              <p className="flex items-center gap-2 text-[13.5px]">
                <PlayCircle className="size-5" aria-hidden /> Demo video coming soon
              </p>
            </div>
          </section>

          <section aria-labelledby="stack-heading" className="border-t py-14">
            <h2 id="stack-heading" className="font-display text-[26px] font-medium">
              A production-style MLOps system, on free tiers
            </h2>
            <p className="mt-2 max-w-2xl text-[14.5px] leading-relaxed text-muted-foreground">
              Streaming ingestion, five models with a fallback chain, and an experiment registry
              are built. Drift monitoring, automated retraining, and the cluster come next. It all
              runs at no cost except the language model behind Explain.
            </p>
            <ul className="mt-5 flex flex-wrap gap-2" aria-label="Built">
              {BUILT.map((item) => (
                <li key={item} className="rounded-md border bg-card px-2.5 py-1 font-mono text-[12px]">
                  {item}
                </li>
              ))}
            </ul>
            <ul className="mt-2 flex flex-wrap gap-2" aria-label="Next">
              {NEXT.map((item) => (
                <li key={item} className="rounded-md border border-dashed px-2.5 py-1 font-mono text-[12px] text-muted-foreground">
                  next: {item}
                </li>
              ))}
            </ul>
          </section>
        </main>
      </div>
      <SiteFooter />
    </div>
  );
}

function SignInOptions() {
  return (
    <div className="space-y-3">
      <SeatStatus />
      <Link
        href="/queue"
        className="group flex items-start gap-4 rounded-lg border bg-card p-5 transition-colors hover:border-foreground/30"
      >
        <UserRound className="mt-0.5 size-5 shrink-0" aria-hidden />
        <span className="flex-1">
          <span className="flex items-center justify-between font-medium">
            Try it as a guest
            <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" aria-hidden />
          </span>
          <span className="mt-1 block text-[13.5px] leading-snug text-muted-foreground">
            One click, no password. Read the feed, the queue, and the models, and ask for five
            explanations. Up to 15 minutes, one person at a time.
          </span>
        </span>
      </Link>
      <Link
        href="/queue"
        className="flex items-start gap-4 rounded-lg border p-5 transition-colors hover:border-foreground/30"
      >
        <KeyRound className="mt-0.5 size-5 shrink-0 text-muted-foreground" aria-hidden />
        <span className="flex-1">
          <span className="font-medium">Owner sign-in</span>
          <span className="mt-1 block text-[13.5px] leading-snug text-muted-foreground">
            The owner’s account reviews edits and runs retraining, promotion, and rollback.
          </span>
        </span>
      </Link>
    </div>
  );
}

const SHOT_CLASS = "w-full rounded-lg border shadow-[0_24px_48px_-24px_oklch(0_0_0/0.35)]";

/**
 * next/image's own <img> carries style="color:transparent" (to hide alt text while it loads),
 * which the strict style-src CSP blocks (F-116). getImageProps gives the same sizing and srcset;
 * the style is dropped, and the images are pre-compressed WebP, so nothing is optimized at runtime.
 */
function shotProps(src: string, alt: string, sizes: string) {
  const { props } = getImageProps({ src, alt, width: 1360, height: 900, sizes, unoptimized: true });
  const { style, ...rest } = props;
  void style;
  return rest;
}

function Screenshots() {
  const queue = shotProps(
    "/screenshots/queue-light.webp",
    "The review queue: edits ranked by risk, each with a band, a score, a Logged out label where it applies, the size of the change, and the model that scored it.",
    "(min-width: 1024px) 60vw, 100vw",
  );
  const queueDark = shotProps("/screenshots/queue-dark.webp", "", "(min-width: 1024px) 60vw, 100vw");
  const edit = shotProps(
    "/screenshots/edit-light.webp",
    "An edit's page: the text it removed and added side by side, the owner's review buttons, and its score on a meter.",
    "(min-width: 1024px) 40vw, 100vw",
  );
  return (
    <section aria-labelledby="screens-heading" className="pt-16">
      <h2 id="screens-heading" className="sr-only">
        Screenshots
      </h2>
      <div className="grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <figure>
          <picture>
            <source media="(prefers-color-scheme: dark)" srcSet={queueDark.srcSet} />
            {/* eslint-disable-next-line jsx-a11y/alt-text -- alt comes from shotProps */}
            <img {...queue} loading="lazy" decoding="async" className={SHOT_CLASS} />
          </picture>
          <figcaption className="mt-2 text-[12.5px] text-muted-foreground">
            The review queue, riskiest first.
          </figcaption>
        </figure>
        <figure className="lg:pt-16">
          {/* eslint-disable-next-line @next/next/no-img-element, jsx-a11y/alt-text -- see shotProps */}
          <img {...edit} loading="lazy" decoding="async" className={SHOT_CLASS} />
          <figcaption className="mt-2 text-[12.5px] text-muted-foreground">
            One edit: what it changed, its score, and the review buttons.
          </figcaption>
        </figure>
      </div>
      <p className="mt-3 text-[12px] text-muted-foreground">
        Shown with recorded Wikipedia edits whose text was scrubbed for the test fixtures.
      </p>
    </section>
  );
}

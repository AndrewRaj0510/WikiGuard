import type { Metadata } from "next";
import Link from "next/link";
import { Wordmark } from "@/components/logo";
import { REPORT_URL, REPO_URL, SiteFooter } from "@/components/site-footer";
import { EXTERNAL_REL } from "@/lib/wikipedia";

export const metadata: Metadata = {
  title: "Privacy",
  description: "What WikiGuard stores about Wikipedia edits and about visitors, and for how long.",
};

/**
 * Written from the code, not from intentions: each line matches what the migrations, the API, and
 * the retention job do (docs/walkthrough.md). Change it when they change.
 */
export default function PrivacyPage() {
  return (
    <div className="flex min-h-dvh flex-col">
      <header className="mx-auto flex h-16 w-full max-w-3xl items-center px-4">
        <Link href="/" aria-label="WikiGuard home">
          <Wordmark />
        </Link>
      </header>
      <main className="mx-auto w-full max-w-3xl flex-1 px-4 pt-8 pb-16">
        <p className="text-[11px] font-medium tracking-[0.14em] text-risk-flagged uppercase">
          Privacy notice
        </p>
        <h1 className="mt-2 font-display text-[40px] leading-tight font-medium tracking-[-0.015em]">
          What WikiGuard keeps, and for how long
        </h1>
        <p className="mt-4 text-[16px] leading-relaxed text-muted-foreground">
          WikiGuard is a portfolio project that ranks English Wikipedia edits for human review. It
          has no accounts to sign up for, no adverts, and no analytics.
        </p>

        <Section title="About Wikipedia edits">
          <li>
            It reads public data from Wikimedia: each article edit’s title, edit summary, the text
            added and removed, its size, its time, and its revision number.
          </li>
          <li>
            <strong>No editor’s username or IP address is stored.</strong> Only a keyed hash of the
            username (HMAC-SHA-256 with a secret key), and whether the editor was logged out. The
            app never shows either.
          </li>
          <li>
            Edits are kept in the database for three days, and in a compressed archive for 30. Past
            that, only aggregate numbers remain, along with the dataset snapshots behind the
            current models.
          </li>
          <li>
            When Wikipedia hides a revision, usually because it held personal information,
            WikiGuard deletes its copy of the text and any explanation of it.
          </li>
        </Section>

        <Section title="About the Explain button">
          <li>
            Explain sends the stored text, title, and summary of that one edit to OpenAI, or to
            Google’s Gemini as a backup, to write a one-line summary. The answer is saved, so the
            same edit isn’t sent twice.
          </li>
          <li>Nothing about you is sent with it.</li>
        </Section>

        <Section title="About you, the visitor">
          <li>
            Signing in sets one first-party cookie, <code className="font-mono text-[14px]">wikiguard_session</code>,
            which scripts can’t read. It holds a random token; the server stores only a keyed hash
            of it. There are no other cookies.
          </li>
          <li>
            Your IP address is never stored as it is. A keyed hash of it is kept with your
            session, for the one-minute wait between guest sessions and to slow down password
            guessing. The API also uses the address in memory to rate-limit requests.
          </li>
          <li>Ended sessions are deleted after seven days.</li>
          <li>
            Actions such as signing in and saving a review are written to an audit log by account
            (guest or owner), not by person or address.
          </li>
          <li>
            Fonts and scripts are served from this site. No third-party scripts run in your
            browser.
          </li>
          <li>
            The site’s hosts (Vercel, Render, Cloudflare, and Supabase) handle your requests and may
            keep their own logs under their own policies.
          </li>
        </Section>

        <Section title="Wikipedia’s content">
          <li>
            Edit text and titles are Wikipedia content, available under{" "}
            <a
              href="https://creativecommons.org/licenses/by-sa/4.0/"
              rel={`license ${EXTERNAL_REL}`}
              className="underline underline-offset-2"
            >
              CC BY-SA 4.0
            </a>
            . Each edit’s page links to Wikipedia, which lists its authors.
          </li>
          <li>
            Some of it is vandalism and can be offensive. WikiGuard shows it unedited, as text
            only, so a person can judge it, and asks search engines not to index those pages.
          </li>
        </Section>

        <Section title="Questions and reports">
          <li>
            To report content, ask for something to be removed, or ask about this notice, open an
            issue on{" "}
            <a href={REPORT_URL} rel={EXTERNAL_REL} className="underline underline-offset-2">
              GitHub
            </a>
            . The{" "}
            <a href={REPO_URL} rel={EXTERNAL_REL} className="underline underline-offset-2">
              source code
            </a>{" "}
            shows exactly what’s stored.
          </li>
        </Section>
      </main>
      <SiteFooter />
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <section className="mt-10">
      <h2 className="font-display text-[24px] font-medium">{title}</h2>
      <ul className="mt-3 list-disc space-y-2 pl-5 text-[15.5px] leading-relaxed marker:text-muted-foreground">
        {children}
      </ul>
    </section>
  );
}

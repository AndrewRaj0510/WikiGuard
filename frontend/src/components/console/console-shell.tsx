"use client";

import { Suspense } from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { LogOut } from "lucide-react";
import { Wordmark } from "@/components/logo";
import { Button } from "@/components/ui/button";
import { GateScreen } from "@/components/console/gate-screen";
import { StatusBanner } from "@/components/console/status-banner";
import { ShortcutsHelp } from "@/components/shortcuts-help";
import { SiteFooter } from "@/components/site-footer";
import { formatCountdown } from "@/lib/format";
import { SessionProvider, useSession, useSessionStore } from "@/lib/session/provider";
import type { SeatedState } from "@/lib/session/store";
import { useNow } from "@/lib/use-now";
import { cn } from "@/lib/utils";

const NAV = [
  { href: "/queue", label: "Queue" },
  { href: "/feed", label: "Live feed" },
  { href: "/models", label: "Models" },
  { href: "/health", label: "Health" },
] as const;

export function ConsoleShell({ children }: { children: React.ReactNode }) {
  return (
    <SessionProvider>
      <div className="flex min-h-dvh flex-col">
        <a
          href="#main"
          className="sr-only z-50 rounded-md bg-card px-3 py-2 focus:not-sr-only focus:fixed focus:top-2 focus:left-2"
        >
          Skip to content
        </a>
        <TopBar />
        <main id="main" className="flex-1">
          <SeatedOnly>{children}</SeatedOnly>
        </main>
        <SiteFooter />
      </div>
    </SessionProvider>
  );
}

/** Pages mount only once this tab holds the seat, so none of them fetches before it can. */
function SeatedOnly({ children }: { children: React.ReactNode }) {
  const state = useSession();
  if (state.status !== "seated") return <GateScreen />;
  return (
    <>
      <StatusBanner />
      {children}
    </>
  );
}

function TopBar() {
  const state = useSession();
  return (
    <header className="sticky top-0 z-40 border-b bg-background/80 backdrop-blur-md">
      <div className="mx-auto flex h-14 max-w-6xl items-center gap-3 px-4 sm:gap-6">
        <Link href="/" aria-label="WikiGuard home" className="shrink-0 rounded-sm">
          <Wordmark />
        </Link>
        {/* usePathname suspends while the server renders a dynamic route, so the nav shows with
            nothing highlighted until the browser fills in the current tab. */}
        <div className="hidden md:block">
          <Suspense fallback={<Nav pathname={null} variant="bar" />}>
            <CurrentNav variant="bar" />
          </Suspense>
        </div>
        <div className="ml-auto flex items-center gap-1">
          {state.status === "seated" && (
            <>
              <ShortcutsHelp />
              <SessionChip seat={state} />
            </>
          )}
        </div>
      </div>
      {/* Narrow screens: the nav scrolls sideways under the logo instead of squeezing. */}
      <div className="overflow-x-auto border-t md:hidden">
        <Suspense fallback={<Nav pathname={null} variant="strip" />}>
          <CurrentNav variant="strip" />
        </Suspense>
      </div>
    </header>
  );
}

function CurrentNav({ variant }: { variant: "bar" | "strip" }) {
  return <Nav pathname={usePathname()} variant={variant} />;
}

function Nav({ pathname, variant }: { pathname: string | null; variant: "bar" | "strip" }) {
  return (
    <nav
      aria-label="Console"
      className={cn("flex items-center gap-0.5", variant === "strip" && "px-2 py-1")}
    >
      {NAV.map((item) => {
        const active =
          pathname !== null && (pathname === item.href || pathname.startsWith(`${item.href}/`));
        return (
          <Link
            key={item.href}
            href={item.href}
            aria-current={active ? "page" : undefined}
            className={cn(
              "relative rounded-md px-2.5 py-1.5 text-[13.5px] font-medium whitespace-nowrap text-muted-foreground transition-colors hover:text-foreground",
              active && "text-foreground",
              active &&
                variant === "bar" &&
                "after:absolute after:inset-x-2.5 after:-bottom-[11px] after:h-0.5 after:rounded-full after:bg-risk-flagged",
              active && variant === "strip" && "bg-accent",
            )}
          >
            {item.label}
          </Link>
        );
      })}
    </nav>
  );
}

function SessionChip({ seat }: { seat: SeatedState }) {
  const store = useSessionStore();
  const now = useNow();
  const secondsLeft =
    seat.guestEndsAt !== null && now > 0 ? Math.max(0, (seat.guestEndsAt - now) / 1000) : null;
  return (
    <div className="flex items-center gap-3">
      <p className="flex items-center gap-2 text-[12.5px] text-muted-foreground">
        <span className="hidden rounded-[5px] border px-1.5 py-px font-medium text-foreground sm:inline">
          {seat.role === "user" ? "Owner" : "Guest"}
        </span>
        {secondsLeft !== null && (
          <span
            className={cn("font-mono tabular", secondsLeft < 60 && "text-risk-flagged")}
            title="Guest sessions end after 15 minutes, or 5 without activity"
          >
            {formatCountdown(secondsLeft)} left
          </span>
        )}
      </p>
      <Button variant="ghost" size="sm" aria-label="Sign out" onClick={() => void store.signOut()}>
        <LogOut aria-hidden /> <span className="hidden sm:inline">Sign out</span>
      </Button>
    </div>
  );
}

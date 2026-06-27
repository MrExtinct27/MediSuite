"use client";

import Link from "next/link";
import { useState } from "react";
import { PanelLeftClose, PanelLeftOpen } from "lucide-react";

export function Sidebar() {
  const [open, setOpen] = useState(true);

  if (!open) {
    return (
      <aside className="hidden shrink-0 border-r border-[rgba(0,212,255,0.08)] md:block">
        <div className="flex h-[calc(100vh-80px)] flex-col py-6 pl-1 pr-2">
          <button
            type="button"
            onClick={() => setOpen(true)}
            aria-label="Open navigation"
            className="rounded-xl p-2 text-[rgba(228,240,255,0.45)] transition-all hover:bg-[rgba(0,212,255,0.06)] hover:text-[#00D4FF]"
          >
            <PanelLeftOpen className="size-4" />
          </button>
        </div>
      </aside>
    );
  }

  return (
    <aside className="hidden w-52 shrink-0 border-r border-[rgba(0,212,255,0.08)] md:block">
      <div className="flex h-[calc(100vh-80px)] flex-col gap-2 py-6 pl-1 pr-3">
        <div className="mb-2 flex items-center justify-between px-2">
          <p
            className="text-xs uppercase tracking-widest text-[rgba(228,240,255,0.4)]"
            style={{ fontFamily: "var(--font-dm-mono)" }}
          >
            Navigation
          </p>
          <button
            type="button"
            onClick={() => setOpen(false)}
            aria-label="Close navigation"
            className="rounded-md p-1 text-[rgba(228,240,255,0.35)] transition-all hover:bg-[rgba(0,212,255,0.06)] hover:text-[#00D4FF]"
          >
            <PanelLeftClose className="size-4" />
          </button>
        </div>
        <nav className="flex flex-col gap-0.5">
          <Link
            href="/dashboard"
            className="rounded-xl px-3 py-2.5 text-sm font-medium uppercase tracking-wider text-[rgba(228,240,255,0.55)] transition-all hover:bg-[rgba(0,212,255,0.06)] hover:text-[#00D4FF]"
            style={{ fontFamily: "var(--font-dm-mono)" }}
          >
            Dashboard
          </Link>
          <Link
            href="/claims/new"
            className="rounded-xl px-3 py-2.5 text-sm font-medium uppercase tracking-wider text-[rgba(228,240,255,0.55)] transition-all hover:bg-[rgba(0,212,255,0.06)] hover:text-[#00D4FF]"
            style={{ fontFamily: "var(--font-dm-mono)" }}
          >
            Submit Claim
          </Link>
        </nav>
      </div>
    </aside>
  );
}

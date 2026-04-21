import Link from "next/link";

export function Sidebar() {
  return (
    <aside className="hidden w-52 shrink-0 border-r border-[rgba(0,212,255,0.08)] md:block">
      <div className="flex h-[calc(100vh-80px)] flex-col gap-2 p-4 pt-6">
        <p
          className="mb-2 px-2 text-[9px] uppercase tracking-widest text-[rgba(228,240,255,0.3)]"
          style={{ fontFamily: "var(--font-dm-mono)" }}
        >
          Navigation
        </p>
        <nav className="flex flex-col gap-0.5">
          <Link
            href="/dashboard"
            className="rounded-xl px-3 py-2.5 text-xs font-medium uppercase tracking-widest text-[rgba(228,240,255,0.45)] transition-all hover:bg-[rgba(0,212,255,0.06)] hover:text-[#00D4FF]"
            style={{ fontFamily: "var(--font-dm-mono)" }}
          >
            Dashboard
          </Link>
          <Link
            href="/claims/new"
            className="rounded-xl px-3 py-2.5 text-xs font-medium uppercase tracking-widest text-[rgba(228,240,255,0.45)] transition-all hover:bg-[rgba(0,212,255,0.06)] hover:text-[#00D4FF]"
            style={{ fontFamily: "var(--font-dm-mono)" }}
          >
            Submit Claim
          </Link>
        </nav>

        <div className="mt-auto">
          <div className="rounded-2xl border border-[rgba(0,212,255,0.08)] bg-[rgba(0,212,255,0.02)] p-3">
            <p
              className="text-[9px] uppercase tracking-widest text-[rgba(228,240,255,0.3)]"
              style={{ fontFamily: "var(--font-dm-mono)" }}
            >
              MediSuite AI
            </p>
            <p
              className="mt-0.5 text-[9px] font-bold text-[#00D4FF]"
              style={{ fontFamily: "var(--font-dm-mono)" }}
            >
              v1.0 — Clinical Intelligence
            </p>
          </div>
        </div>
      </div>
    </aside>
  );
}

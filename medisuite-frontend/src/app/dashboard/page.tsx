"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState } from "react";
import { motion, useInView, useSpring, useTransform } from "framer-motion";
import { ArrowRight } from "lucide-react";

import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import type { ClaimSummary } from "@/lib/types";
import { api } from "@/lib/api";
import { Sidebar } from "@/components/layout/Sidebar";

/* ─── Mock data ─────────────────────────────────────────────────────── */
const MOCK_CLAIMS: ClaimSummary[] = [
  {
    claim_id: "f52fa36a-09d3-44ec-aec8-d092ab0a0a57",
    patient_name: "John Smith",
    service_date: "2025-01-10",
    validation_status: "passed",
    avg_confidence: 0.92,
  },
  {
    claim_id: "321093b7-9045-4ee0-8463-838c343dec0f",
    patient_name: "Maria Lopez",
    service_date: "2025-01-09",
    validation_status: "failed",
    avg_confidence: 0.91,
  },
  {
    claim_id: "c0ffee00-0000-0000-0000-000000000000",
    patient_name: "Daniel Kim",
    service_date: "2025-01-08",
    validation_status: "passed",
    avg_confidence: 0.9,
  },
];

interface Stats { totalClaims: number; passedValidation: number; avgConfidence: number; flaggedClaims: number; }

function computeStats(claims: ClaimSummary[]): Stats {
  let passedValidation = 0, flaggedClaims = 0, confidenceSum = 0;
  for (const c of claims) {
    if (c.validation_status === "passed") passedValidation++;
    else if (c.validation_status === "processing") { /* in-flight — neither passed nor flagged */ }
    else flaggedClaims++;
    confidenceSum += c.avg_confidence ?? 0;
  }
  return {
    totalClaims: claims.length,
    passedValidation,
    flaggedClaims,
    avgConfidence: claims.length ? confidenceSum / claims.length : 0,
  };
}

/* ─── Animated counter ──────────────────────────────────────────────── */
function AnimatedNumber({ to, decimals = 0, suffix = "" }: { to: number; decimals?: number; suffix?: string }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true });
  const spring = useSpring(0, { mass: 0.8, stiffness: 60, damping: 15 });
  const display = useTransform(spring, (v) => v.toFixed(decimals) + suffix);
  useEffect(() => { if (inView) spring.set(to); }, [inView, spring, to]);
  return <motion.span ref={ref}>{display}</motion.span>;
}

/* ─── Stat card ─────────────────────────────────────────────────────── */
function StatCard({ label, rawValue, decimals = 0, suffix = "", accentColor, index }: {
  label: string; rawValue: number; decimals?: number; suffix?: string; accentColor: string; index: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 + index * 0.08, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="glass-card relative overflow-hidden p-5"
      style={{ borderTop: `2px solid ${accentColor}` }}
    >
      <div
        className="mb-3 text-4xl font-bold leading-none"
        style={{ fontFamily: "var(--font-dm-mono)", color: accentColor }}
      >
        <AnimatedNumber to={rawValue} decimals={decimals} suffix={suffix} />
      </div>
      <div
        className="text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.45)]"
        style={{ fontFamily: "var(--font-dm-sans)" }}
      >
        {label}
      </div>
    </motion.div>
  );
}

/* ─── Confidence sparkline ──────────────────────────────────────────── */
// Round to 2 decimals so SSR and client serialize coordinates to identical
// strings (avoids float-precision hydration mismatches like 11.7165...874 vs ...595).
function r2(value: number): number {
  return Math.round(value * 100) / 100;
}

function SparklineCard({ children }: { children: React.ReactNode }) {
  return <div className="glass-card p-5">{children}</div>;
}

function SparklineHeader() {
  return (
    <p
      className="mb-4 text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.5)]"
      style={{ fontFamily: "var(--font-dm-mono)" }}
    >
      Confidence Trend
    </p>
  );
}

function ConfidenceSparkline({ claims }: { claims: ClaimSummary[] }) {
  // Belt-and-suspenders: render the chart only after client mount so there is
  // no SSR markup to mismatch against. The placeholder keeps layout stable.
  const [mounted, setMounted] = useState(false);
  useEffect(() => setMounted(true), []);

  // Real data: avg_confidence per claim, oldest → newest. The API returns
  // claims newest-first, so reverse to get chronological order.
  const data = [...claims]
    .reverse()
    .map((c) => (c.avg_confidence ?? 0) * 100);

  if (!mounted) {
    return <SparklineCard><SparklineHeader /><div style={{ height: 124 }} /></SparklineCard>;
  }

  // Fewer than 2 real claims → a single point isn't a meaningful trend.
  if (data.length < 2) {
    return (
      <SparklineCard>
        <SparklineHeader />
        <div className="flex h-[124px] items-center justify-center">
          <p
            className="text-xs text-[rgba(228,240,255,0.4)]"
            style={{ fontFamily: "var(--font-dm-mono)" }}
          >
            Not enough data yet
          </p>
        </div>
      </SparklineCard>
    );
  }

  const W = 280, H = 100;
  const rawMin = Math.min(...data);
  const rawMax = Math.max(...data);
  // Add a little headroom; guard against a flat line (all values equal).
  const min = rawMin === rawMax ? rawMin - 5 : rawMin - 2;
  const max = rawMin === rawMax ? rawMax + 5 : rawMax + 2;
  const pts = data.map((d, i) => {
    const x = r2((i / (data.length - 1)) * W);
    const y = r2(H - ((d - min) / (max - min)) * H * 0.85 - H * 0.05);
    return [x, y] as [number, number];
  });
  const pathD = `M ${pts.map(([x, y]) => `${x},${y}`).join(" L ")}`;
  const areaD = `${pathD} L ${W},${H} L 0,${H} Z`;

  const overallAvg = data.reduce((a, b) => a + b, 0) / data.length;

  return (
    <SparklineCard>
      <SparklineHeader />

      <svg viewBox={`0 0 ${W} ${H}`} className="w-full" style={{ height: 100 }}>
        <defs>
          <linearGradient id="sparkStroke" x1="0%" y1="0%" x2="100%" y2="0%">
            <stop offset="0%" stopColor="#00D4FF" />
            <stop offset="100%" stopColor="#00FF9C" />
          </linearGradient>
          <linearGradient id="sparkFill" x1="0%" y1="0%" x2="0%" y2="100%">
            <stop offset="0%" stopColor="#00D4FF" stopOpacity="0.15" />
            <stop offset="100%" stopColor="#00D4FF" stopOpacity="0" />
          </linearGradient>
        </defs>

        {/* Area fill */}
        <path d={areaD} fill="url(#sparkFill)" />

        {/* Stroke — animated draw */}
        <motion.path
          d={pathD}
          fill="none"
          stroke="url(#sparkStroke)"
          strokeWidth={2.5}
          strokeLinecap="round"
          strokeLinejoin="round"
          initial={{ pathLength: 0 }}
          animate={{ pathLength: 1 }}
          transition={{ duration: 2.5, ease: "easeInOut", delay: 0.8 }}
        />

        {/* Last data point dot */}
        <circle cx={r2(pts[pts.length - 1][0])} cy={r2(pts[pts.length - 1][1])} r={4} fill="#00FF9C" />
        <circle cx={r2(pts[pts.length - 1][0])} cy={r2(pts[pts.length - 1][1])} r={8} fill="#00FF9C" opacity={0.2} />
      </svg>

      <div className="mt-3 flex items-baseline justify-between">
        <span
          className="text-[9px] uppercase tracking-widest text-[rgba(228,240,255,0.4)]"
          style={{ fontFamily: "var(--font-dm-mono)" }}
        >
          Avg confidence · {data.length} claims
        </span>
        <span
          className="text-xl font-bold text-[#00FF9C]"
          style={{ fontFamily: "var(--font-dm-mono)" }}
        >
          {overallAvg.toFixed(1)}%
        </span>
      </div>
    </SparklineCard>
  );
}

/* ─── Page ──────────────────────────────────────────────────────────── */
export default function DashboardPage() {
  const [claims, setClaims] = useState<ClaimSummary[]>([]);

  useEffect(() => {
    let cancelled = false;
    let intervalId: ReturnType<typeof setInterval> | null = null;

    const stopPolling = () => {
      if (intervalId) {
        clearInterval(intervalId);
        intervalId = null;
      }
    };

    async function load() {
      try {
        const res = await api.get<ClaimSummary[]>("/claims");
        if (cancelled) return;
        setClaims(res.data);
        // Keep refreshing every 3s while any claim is still processing so its
        // badge flips from PROCESSING to PASSED/FLAGGED automatically; stop once none are.
        const anyProcessing = res.data.some((c) => c.validation_status === "processing");
        if (anyProcessing && !intervalId) {
          intervalId = setInterval(load, 3000);
        } else if (!anyProcessing) {
          stopPolling();
        }
      } catch {
        // On a transient error, keep any real claims we already have; only fall
        // back to mock data if we never loaded anything.
        if (!cancelled) setClaims((prev) => (prev.length ? prev : MOCK_CLAIMS));
      }
    }

    load();
    return () => {
      cancelled = true;
      stopPolling();
    };
  }, []);

  const stats = useMemo(() => computeStats(claims), [claims]);
  const displayClaims = claims.length ? claims : MOCK_CLAIMS;

  return (
    <div className="min-h-screen bg-background">
      <div className="mx-auto flex w-full max-w-6xl">
        <Sidebar />

        <main className="flex-1 space-y-6 p-6">
          {/* Header */}
          <motion.div
            initial={{ opacity: 0, y: -12 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.45 }}
            className="flex items-center justify-between gap-4"
          >
            <div>
              <p
                className="mb-1 text-[10px] uppercase tracking-widest text-[#00D4FF]"
                style={{ fontFamily: "var(--font-dm-mono)" }}
              >
                — Overview
              </p>
              <h1
                className="text-3xl font-bold uppercase text-[#E4F0FF]"
                style={{ fontFamily: "var(--font-syne)", letterSpacing: "-0.02em" }}
              >
                Dashboard
              </h1>
            </div>
            <Link
              href="/claims/new"
              className="flex items-center gap-2 rounded-2xl bg-[#00D4FF] px-5 py-2.5 text-[11px] font-bold uppercase tracking-wider text-black transition-all hover:bg-[#00D4FF]/85 hover:shadow-[0_0_20px_rgba(0,212,255,0.35)]"
              style={{ fontFamily: "var(--font-syne)" }}
            >
              New Claim <ArrowRight className="size-3.5" />
            </Link>
          </motion.div>

          {/* Stats row */}
          <section className="grid gap-3 md:grid-cols-4">
            <StatCard index={0} label="Total Claims"      rawValue={stats.totalClaims}                                      accentColor="#00D4FF" />
            <StatCard index={1} label="Passed Validation" rawValue={stats.passedValidation}                                  accentColor="#00FF9C" />
            <StatCard index={2} label="Avg Confidence"    rawValue={stats.avgConfidence * 100} decimals={1} suffix="%"      accentColor="#00FF9C" />
            <StatCard index={3} label="Flagged Claims"    rawValue={stats.flaggedClaims}                                     accentColor="#FF3B6B" />
          </section>

          {/* Bento — table + sparkline */}
          <section className="grid gap-4 md:grid-cols-[1fr_280px]">

            {/* Claims table */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.45, duration: 0.5 }}
              className="glass-card overflow-hidden"
            >
              <div className="flex items-center justify-between border-b border-[rgba(0,212,255,0.08)] px-5 py-3">
                <span
                  className="text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.5)]"
                  style={{ fontFamily: "var(--font-dm-mono)" }}
                >
                  Claims — {displayClaims.length} records
                </span>
              </div>

              <Table>
                <TableHeader>
                  <TableRow className="border-b border-[rgba(0,212,255,0.06)] hover:bg-transparent">
                    {["Claim ID", "Patient", "Date", "Status", ""].map((h) => (
                      <TableHead
                        key={h}
                        className="text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.35)]"
                        style={{ fontFamily: "var(--font-dm-mono)" }}
                      >
                        {h}
                      </TableHead>
                    ))}
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {displayClaims.map((claim, idx) => {
                    const isProcessing = claim.validation_status === "processing";
                    const passed = claim.validation_status === "passed";
                    const statusColor = isProcessing ? "#00D4FF" : passed ? "#00FF9C" : "#F59E0B";
                    const statusLabel = isProcessing ? "PROCESSING" : passed ? "PASSED" : "FLAGGED";

                    return (
                      <motion.tr
                        key={claim.claim_id}
                        initial={{ opacity: 0, x: -8 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: 0.5 + idx * 0.06, duration: 0.35 }}
                        className="group border-b border-[rgba(0,212,255,0.05)] transition-all hover:bg-[rgba(0,212,255,0.04)]"
                        style={{ borderLeft: "2px solid transparent" }}
                        onMouseEnter={(e) => { (e.currentTarget as HTMLElement).style.borderLeftColor = "#00D4FF"; }}
                        onMouseLeave={(e) => { (e.currentTarget as HTMLElement).style.borderLeftColor = "transparent"; }}
                      >
                        <TableCell
                          className="text-[11px] text-[#00D4FF]"
                          style={{ fontFamily: "var(--font-dm-mono)" }}
                        >
                          {claim.claim_id.slice(0, 8)}
                        </TableCell>
                        <TableCell
                          className="text-sm text-[#E4F0FF]"
                          style={{ fontFamily: "var(--font-dm-sans)" }}
                        >
                          {claim.patient_name?.trim() ? claim.patient_name : "N/A"}
                        </TableCell>
                        <TableCell
                          className="text-[11px] text-[rgba(228,240,255,0.45)]"
                          style={{ fontFamily: "var(--font-dm-mono)" }}
                        >
                          {claim.service_date ?? "—"}
                        </TableCell>
                        <TableCell>
                          <motion.span
                            className="inline-flex items-center gap-1.5 rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-widest"
                            style={{
                              fontFamily: "var(--font-dm-mono)",
                              color: statusColor,
                              background: `${statusColor}18`,
                              border: `1px solid ${statusColor}40`,
                            }}
                            animate={isProcessing ? { opacity: [0.55, 1, 0.55] } : { opacity: 1 }}
                            transition={
                              isProcessing
                                ? { duration: 1.6, repeat: Infinity, ease: "easeInOut" }
                                : { duration: 0 }
                            }
                          >
                            {isProcessing && (
                              <span
                                className="inline-block size-1.5 rounded-full"
                                style={{ background: statusColor, boxShadow: `0 0 6px ${statusColor}` }}
                              />
                            )}
                            {statusLabel}
                          </motion.span>
                        </TableCell>
                        <TableCell className="text-right">
                          <Link
                            href={`/claims/${claim.claim_id}`}
                            className="text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.35)] underline-offset-2 transition-colors hover:text-[#00D4FF] hover:underline"
                            style={{ fontFamily: "var(--font-dm-mono)" }}
                          >
                            View
                          </Link>
                        </TableCell>
                      </motion.tr>
                    );
                  })}
                </TableBody>
              </Table>
            </motion.div>

            {/* Sparkline panel */}
            <motion.div
              initial={{ opacity: 0, x: 16 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.55, duration: 0.5 }}
            >
              <ConfidenceSparkline claims={claims} />
            </motion.div>
          </section>
        </main>
      </div>
    </div>
  );
}

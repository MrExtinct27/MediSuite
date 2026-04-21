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
import type { Claim, DiagnosisCode } from "@/lib/types";
import { api } from "@/lib/api";
import { Sidebar } from "@/components/layout/Sidebar";

/* ─── Mock data ─────────────────────────────────────────────────────── */
const MOCK_CLAIMS: Claim[] = [
  {
    claim_id: "f52fa36a-09d3-44ec-aec8-d092ab0a0a57",
    form_type: "CMS-1500",
    generated_at: new Date().toISOString(),
    patient: { name: "John Smith", dob: "1972-03-15", insurance_id: "BCB123456" },
    service_date: "2025-01-10",
    provider_name: "Dr. Alice Carter",
    facility_name: "Cypress Medical Center",
    diagnosis_codes: [
      { code: "E11.21", description: "Type 2 diabetes mellitus with nephropathy", confidence: 0.95 },
      { code: "I10", description: "Essential (primary) hypertension", confidence: 0.9 },
    ],
    procedure_codes: [{ code: "99214", description: "Office outpatient visit", confidence: 0.9 }],
    validation_status: "passed", validation_errors: [],
    explainability: { icd10_reasoning: [], cpt4_reasoning: [], citations: [], icd10_reasoning_chain: "", cpt4_reasoning_chain: "" },
    processing_status: "claim_generated",
  },
  {
    claim_id: "321093b7-9045-4ee0-8463-838c343dec0f",
    form_type: "CMS-1500",
    generated_at: new Date().toISOString(),
    patient: { name: "Maria Lopez", dob: "1984-09-03", insurance_id: "AET987654" },
    service_date: "2025-01-09",
    provider_name: "Dr. Ryan Lee",
    facility_name: "OC Renal Clinic",
    diagnosis_codes: [
      { code: "N18.3", description: "Chronic kidney disease, stage 3", confidence: 0.93 },
      { code: "E11.21", description: "Type 2 diabetes mellitus with nephropathy", confidence: 0.96 },
    ],
    procedure_codes: [
      { code: "80053", description: "Comprehensive metabolic panel", confidence: 0.89 },
      { code: "83036", description: "Hemoglobin A1c", confidence: 0.91 },
    ],
    validation_status: "warning",
    validation_errors: [{ field: "icd10:E11.21", message: "Possible missing secondary code", severity: "warning" }],
    explainability: { icd10_reasoning: [], cpt4_reasoning: [], citations: [], icd10_reasoning_chain: "", cpt4_reasoning_chain: "" },
    processing_status: "claim_generated",
  },
  {
    claim_id: "c0ffee00-0000-0000-0000-000000000000",
    form_type: "CMS-1500",
    generated_at: new Date().toISOString(),
    patient: { name: "Daniel Kim", dob: "1990-11-22", insurance_id: "UHC445566" },
    service_date: "2025-01-08",
    provider_name: "Dr. Sarah Ahmed",
    facility_name: "Fullerton Family Practice",
    diagnosis_codes: [{ code: "J45.40", description: "Moderate persistent asthma", confidence: 0.92 }],
    procedure_codes: [{ code: "94640", description: "Nebulizer treatment", confidence: 0.88 }],
    validation_status: "passed", validation_errors: [],
    explainability: { icd10_reasoning: [], cpt4_reasoning: [], citations: [], icd10_reasoning_chain: "", cpt4_reasoning_chain: "" },
    processing_status: "claim_generated",
  },
];

interface Stats { totalClaims: number; passedValidation: number; avgConfidence: number; flaggedClaims: number; }

function computeStats(claims: Claim[]): Stats {
  let passedValidation = 0, flaggedClaims = 0, confidenceSum = 0, confidenceCount = 0;
  for (const c of claims) {
    if (c.validation_status === "passed") passedValidation++;
    if (c.validation_errors?.length) flaggedClaims++;
    const all: DiagnosisCode[] = [...(c.diagnosis_codes ?? []), ...(c.procedure_codes ?? [])];
    for (const d of all) { confidenceSum += d.confidence ?? 0; confidenceCount++; }
  }
  return { totalClaims: claims.length, passedValidation, flaggedClaims, avgConfidence: confidenceCount ? confidenceSum / confidenceCount : 0 };
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
function ConfidenceSparkline() {
  const data = [78, 82, 75, 89, 85, 94, 91, 88, 95, 97, 93, 96, 94.8];
  const W = 280, H = 100;
  const min = Math.min(...data) - 5;
  const max = Math.max(...data) + 2;
  const pts = data.map((d, i) => {
    const x = (i / (data.length - 1)) * W;
    const y = H - ((d - min) / (max - min)) * H * 0.85 - H * 0.05;
    return [x, y] as [number, number];
  });
  const pathD = `M ${pts.map(([x, y]) => `${x},${y}`).join(" L ")}`;
  const areaD = `${pathD} L ${W},${H} L 0,${H} Z`;

  return (
    <div className="glass-card p-5">
      <p
        className="mb-4 text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.5)]"
        style={{ fontFamily: "var(--font-dm-mono)" }}
      >
        Confidence Trend
      </p>

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

        {/* Vertical bar hints */}
        {pts.map(([x], i) => (
          <rect
            key={i}
            x={x - 3}
            y={H * (0.1 + Math.random() * 0.3)}
            width={6}
            height={H * (0.1 + Math.random() * 0.4)}
            fill="#00D4FF"
            opacity={0.08}
          />
        ))}

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
        <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r={4} fill="#00FF9C" />
        <circle cx={pts[pts.length - 1][0]} cy={pts[pts.length - 1][1]} r={8} fill="#00FF9C" opacity={0.2} />
      </svg>

      <p
        className="mt-3 text-right text-xl font-bold text-[#00FF9C]"
        style={{ fontFamily: "var(--font-dm-mono)" }}
      >
        94.8%
      </p>
    </div>
  );
}

/* ─── Page ──────────────────────────────────────────────────────────── */
export default function DashboardPage() {
  const [claims, setClaims] = useState<Claim[]>([]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      try {
        const res = await api.get<Claim[]>("/claims");
        if (!cancelled) setClaims(res.data);
      } catch {
        if (!cancelled) setClaims(MOCK_CLAIMS);
      }
    }
    load();
    return () => { cancelled = true; };
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
                    {["Claim ID", "Patient", "Date", "ICD-10", "Status", "Conf %", ""].map((h) => (
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
                    const codes = claim.diagnosis_codes ?? [];
                    const maxConf = codes.length ? Math.max(...codes.map((c) => c.confidence ?? 0)) : 0;
                    const passed = claim.validation_status === "passed";
                    const hasWarn = (claim.validation_errors?.length ?? 0) > 0;
                    const statusColor = passed ? "#00FF9C" : hasWarn ? "#F59E0B" : "rgba(228,240,255,0.3)";
                    const statusLabel = passed ? "PASSED" : hasWarn ? "WARNING" : "UNKNOWN";

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
                          {claim.patient?.name}
                        </TableCell>
                        <TableCell
                          className="text-[11px] text-[rgba(228,240,255,0.45)]"
                          style={{ fontFamily: "var(--font-dm-mono)" }}
                        >
                          {claim.service_date ?? "—"}
                        </TableCell>
                        <TableCell>
                          <div className="flex flex-wrap gap-1">
                            {codes.slice(0, 2).map((c) => (
                              <span key={c.code} className="code-badge">{c.code}</span>
                            ))}
                            {codes.length > 2 && (
                              <span
                                className="text-[10px] text-[rgba(228,240,255,0.35)]"
                                style={{ fontFamily: "var(--font-dm-mono)" }}
                              >
                                +{codes.length - 2}
                              </span>
                            )}
                          </div>
                        </TableCell>
                        <TableCell>
                          <span
                            className="rounded-full px-2 py-0.5 text-[9px] font-bold uppercase tracking-widest"
                            style={{
                              fontFamily: "var(--font-dm-mono)",
                              color: statusColor,
                              background: `${statusColor}18`,
                              border: `1px solid ${statusColor}40`,
                            }}
                          >
                            {statusLabel}
                          </span>
                        </TableCell>
                        <TableCell
                          className="text-[11px] font-bold"
                          style={{
                            fontFamily: "var(--font-dm-mono)",
                            color: maxConf >= 0.9 ? "#00FF9C" : maxConf >= 0.7 ? "#F59E0B" : "#FF3B6B",
                          }}
                        >
                          {maxConf ? `${Math.round(maxConf * 100)}%` : "—"}
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
              <ConfidenceSparkline />
            </motion.div>
          </section>
        </main>
      </div>
    </div>
  );
}

"use client";

import { useParams } from "next/navigation";
import { useEffect, useMemo, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { Download, FileText, Quote, TriangleAlert, ChevronDown } from "lucide-react";
import { jsPDF } from "jspdf";

import type { Claim, DiagnosisCode, Explainability } from "@/lib/types";
import { api } from "@/lib/api";
import { useClaimsStore } from "@/store/claimsStore";

/* ─── Confidence bar ─────────────────────────────────────────────────── */
function ConfidenceBar({ value }: { value: number }) {
  const color = value >= 0.9 ? "#00FF9C" : value >= 0.7 ? "#F59E0B" : "#FF3B6B";
  return (
    <div className="mt-2.5 h-1.5 w-full overflow-hidden rounded-full bg-[rgba(255,255,255,0.06)]">
      <motion.div
        className="h-full rounded-full"
        style={{ backgroundColor: color }}
        initial={{ width: 0 }}
        animate={{ width: `${Math.round(value * 100)}%` }}
        transition={{ duration: 0.8, ease: "easeOut", delay: 0.3 }}
      />
    </div>
  );
}

/* ─── Code card ──────────────────────────────────────────────────────── */
function CodeCard({ code, description, confidence, reasoning, index }: {
  code: string; description: string; confidence: number; reasoning?: string; index: number;
}) {
  const confColor = confidence >= 0.9 ? "#00FF9C" : confidence >= 0.7 ? "#F59E0B" : "#FF3B6B";
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.1 + index * 0.08, duration: 0.45 }}
      className="glass-card-sm p-4"
    >
      <div className="mb-2 flex items-start justify-between gap-3">
        <div>
          <div
            className="mb-1 text-xl font-bold tracking-wider text-[#00D4FF]"
            style={{ fontFamily: "var(--font-dm-mono)" }}
          >
            {code}
          </div>
          <div
            className="text-sm text-[rgba(228,240,255,0.8)]"
            style={{ fontFamily: "var(--font-dm-sans)", lineHeight: 1.5 }}
          >
            {description}
          </div>
        </div>
        <div
          className="shrink-0 text-right text-lg font-bold"
          style={{ fontFamily: "var(--font-dm-mono)", color: confColor }}
        >
          {confidence ? `${Math.round(confidence * 100)}%` : "—"}
        </div>
      </div>
      <ConfidenceBar value={confidence} />
      {reasoning && (
        <p
          className="mt-3 text-xs text-[rgba(228,240,255,0.4)]"
          style={{ fontFamily: "var(--font-dm-sans)", lineHeight: 1.7 }}
        >
          {reasoning}
        </p>
      )}
    </motion.div>
  );
}

/* ─── Accordion section ──────────────────────────────────────────────── */
function AccordionSection({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(false);
  return (
    <div
      className="glass-card-sm overflow-hidden"
      style={{ borderLeft: "2px solid rgba(0,212,255,0.3)" }}
    >
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-5 py-4 text-left"
      >
        <span
          className="text-sm font-bold uppercase tracking-wider text-[rgba(228,240,255,0.8)]"
          style={{ fontFamily: "var(--font-syne)", letterSpacing: "-0.01em" }}
        >
          {title}
        </span>
        <ChevronDown
          className="size-4 text-[#00D4FF] transition-transform duration-200"
          style={{ transform: open ? "rotate(180deg)" : "rotate(0deg)" }}
        />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div
            key="content"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25, ease: "easeInOut" }}
            className="overflow-hidden"
          >
            <div className="border-t border-[rgba(0,212,255,0.08)] px-6 py-10">
              {children}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

/* ─── Download button ────────────────────────────────────────────────── */
function DownloadJsonButton({ claim }: { claim: Claim | null }) {
  const handleDownload = () => {
    if (!claim) return;
    const blob = new Blob([JSON.stringify(claim, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `claim-${claim.claim_id}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  return (
    <button
      type="button"
      onClick={handleDownload}
      disabled={!claim}
      className="flex items-center gap-2 rounded-2xl border border-[#00D4FF] px-5 py-2.5 text-[11px] font-bold uppercase tracking-widest text-[#00D4FF] transition-all hover:bg-[rgba(0,212,255,0.08)] hover:shadow-[0_0_16px_rgba(0,212,255,0.25)] disabled:cursor-not-allowed disabled:opacity-40"
      style={{ fontFamily: "var(--font-syne)" }}
    >
      <Download className="size-3.5" />
      Download Claim JSON
    </button>
  );
}

/* ─── CMS-1500 PDF generation ────────────────────────────────────────── */
function generateCms1500Pdf(claim: Claim): void {
  const na = (v: unknown): string =>
    v === null || v === undefined || String(v).trim() === "" ? "N/A" : String(v);

  const doc = new jsPDF({ unit: "pt", format: "letter" });
  const pageWidth = doc.internal.pageSize.getWidth();
  const pageHeight = doc.internal.pageSize.getHeight();
  const marginX = 48;
  const bottomLimit = pageHeight - 60;
  let y = 56;

  const ensureSpace = (needed: number) => {
    if (y + needed > bottomLimit) {
      doc.addPage();
      y = 56;
    }
  };

  // Title
  doc.setFont("helvetica", "bold");
  doc.setFontSize(16);
  doc.setTextColor(20, 20, 20);
  doc.text("CMS-1500 HEALTH INSURANCE CLAIM FORM", pageWidth / 2, y, { align: "center" });
  y += 14;
  doc.setDrawColor(0, 150, 180);
  doc.setLineWidth(1);
  doc.line(marginX, y, pageWidth - marginX, y);
  y += 26;

  const sectionHeader = (title: string) => {
    ensureSpace(36);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(11);
    doc.setTextColor(0, 110, 140);
    doc.text(title, marginX, y);
    y += 6;
    doc.setDrawColor(200, 200, 200);
    doc.setLineWidth(0.5);
    doc.line(marginX, y, pageWidth - marginX, y);
    y += 16;
  };

  const field = (label: string, value: string) => {
    ensureSpace(18);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9);
    doc.setTextColor(90, 90, 90);
    doc.text(`${label}:`, marginX, y);
    doc.setFont("helvetica", "normal");
    doc.setTextColor(20, 20, 20);
    doc.text(value, marginX + 130, y);
    y += 18;
  };

  const patient = claim.patient ?? ({} as Claim["patient"]);

  // Patient Information
  sectionHeader("PATIENT INFORMATION");
  field("Name", na(patient?.name));
  field("DOB", na(patient?.dob));
  field("Sex", na(patient?.sex));
  field("Address", na(patient?.address));
  y += 8;

  // Insurance Information
  sectionHeader("INSURANCE INFORMATION");
  field("Provider", na(patient?.insurance_provider));
  field("Policy #", na(patient?.insurance_id));
  y += 8;

  // Clinical Information
  sectionHeader("CLINICAL INFORMATION");
  field("Place of Service", na((claim as { place_of_service?: string }).place_of_service));
  field("Service Date", na(claim.service_date));
  field("Provider", na(claim.provider_name));
  field("Facility", na(claim.facility_name));
  y += 8;

  // Two-column code table renderer
  const codeTable = (
    title: string,
    rows: { code: string; description: string }[]
  ) => {
    sectionHeader(title);

    const codeX = marginX;
    const descX = marginX + 110;
    const descWidth = pageWidth - descX - marginX;

    // Column headers
    ensureSpace(18);
    doc.setFont("helvetica", "bold");
    doc.setFontSize(9);
    doc.setTextColor(90, 90, 90);
    doc.text("Code", codeX, y);
    doc.text("Description", descX, y);
    y += 6;
    doc.setDrawColor(220, 220, 220);
    doc.setLineWidth(0.5);
    doc.line(marginX, y, pageWidth - marginX, y);
    y += 14;

    doc.setFont("helvetica", "normal");
    doc.setTextColor(20, 20, 20);

    if (!rows.length) {
      ensureSpace(16);
      doc.setTextColor(140, 140, 140);
      doc.text("N/A", codeX, y);
      y += 16;
      return;
    }

    rows.forEach((row) => {
      const descLines = doc.splitTextToSize(na(row.description), descWidth) as string[];
      const rowHeight = Math.max(14, descLines.length * 12);
      ensureSpace(rowHeight);
      doc.setTextColor(20, 20, 20);
      doc.setFont("helvetica", "bold");
      doc.text(na(row.code), codeX, y);
      doc.setFont("helvetica", "normal");
      doc.text(descLines, descX, y);
      y += rowHeight + 4;
    });
  };

  codeTable(
    "DIAGNOSIS CODES (ICD-10)",
    (claim.diagnosis_codes ?? []).map((c) => ({ code: c.code, description: c.description }))
  );
  y += 8;
  codeTable(
    "PROCEDURE CODES (CPT-4)",
    (claim.procedure_codes ?? []).map((c) => ({ code: c.code, description: c.description }))
  );

  // Footer
  ensureSpace(48);
  y = Math.max(y + 16, bottomLimit - 8);
  doc.setDrawColor(200, 200, 200);
  doc.setLineWidth(0.5);
  doc.line(marginX, y, pageWidth - marginX, y);
  y += 16;
  doc.setFont("helvetica", "italic");
  doc.setFontSize(8);
  doc.setTextColor(120, 120, 120);
  doc.text(
    "This is a computer-generated form created by MediSuite AI Medical Coding Assistant.",
    pageWidth / 2,
    y,
    { align: "center" }
  );
  y += 12;
  doc.text(`Generated on: ${new Date().toLocaleString()}`, pageWidth / 2, y, {
    align: "center",
  });

  doc.save(`CMS1500_${claim.claim_id}.pdf`);
}

/* ─── CMS-1500 PDF download button ───────────────────────────────────── */
function DownloadCms1500Button({ claim }: { claim: Claim | null }) {
  return (
    <button
      type="button"
      onClick={() => claim && generateCms1500Pdf(claim)}
      disabled={!claim}
      className="flex items-center gap-2 rounded-2xl border border-[#00D4FF] px-5 py-2.5 text-[11px] font-bold uppercase tracking-widest text-[#00D4FF] transition-all hover:bg-[rgba(0,212,255,0.08)] hover:shadow-[0_0_16px_rgba(0,212,255,0.25)] disabled:cursor-not-allowed disabled:opacity-40"
      style={{ fontFamily: "var(--font-syne)" }}
    >
      <FileText className="size-3.5" />
      Download CMS-1500 Form
    </button>
  );
}

/* ─── Page ───────────────────────────────────────────────────────────── */
export default function ClaimDetailPage() {
  const params = useParams<{ id: string }>();
  const id = params.id;

  const storeActiveClaim = useClaimsStore((s) => s.activeClaim);
  const [claim, setClaim] = useState<Claim | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setLoading(true);
      setError(null);
      try {
        const res = await api.get<Claim>(`/claims/${id}/result`);
        if (!cancelled) setClaim(res.data);
      } catch {
        if (!cancelled) {
          if (storeActiveClaim && storeActiveClaim.claim_id === id) {
            setClaim(storeActiveClaim as Claim);
          } else {
            setError("Unable to load claim from API or local state.");
          }
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    void load();
    return () => { cancelled = true; };
  }, [id, storeActiveClaim]);

  const explainability: Explainability | null = useMemo(
    () => (claim ? claim.explainability : null),
    [claim]
  );

  const validationPassed = claim?.validation_status === "passed";
  const hasWarnings = (claim?.validation_errors?.length ?? 0) > 0;
  const icd10Codes = claim?.diagnosis_codes ?? [];
  const cptCodes = (claim?.procedure_codes as DiagnosisCode[]) ?? [];

  return (
    <div className="min-h-screen bg-background">
      <main className="mx-auto w-full max-w-6xl px-6 py-10">

        {/* Header card */}
        <motion.div
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.45 }}
          className="glass-card mb-8 p-6"
          style={{ borderTop: "2px solid #00D4FF" }}
        >
          <div className="flex flex-col justify-between gap-4 md:flex-row md:items-start">
            <div>
              <p
                className="mb-1 text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.35)]"
                style={{ fontFamily: "var(--font-dm-mono)" }}
              >
                Claim ID
              </p>
              <p
                className="mb-3 text-sm text-[#00D4FF]"
                style={{ fontFamily: "var(--font-dm-mono)" }}
              >
                {claim?.claim_id ?? id}
              </p>
              <h1
                className="text-2xl font-bold text-[#E4F0FF]"
                style={{ fontFamily: "var(--font-syne)", letterSpacing: "-0.02em", textTransform: "uppercase" }}
              >
                {claim?.patient?.name ?? "Unknown Patient"}
              </h1>
              <div className="mt-3 grid gap-x-8 gap-y-1 sm:grid-cols-2">
                {[
                  { k: "DOB", v: claim?.patient?.dob },
                  { k: "Insurance", v: claim?.patient?.insurance_id },
                  { k: "Service Date", v: claim?.service_date },
                  { k: "Provider", v: claim?.provider_name },
                  { k: "Facility", v: claim?.facility_name },
                ].map(({ k, v }) => (
                  <div key={k} className="flex gap-2 text-xs">
                    <span
                      className="text-[rgba(228,240,255,0.35)]"
                      style={{ fontFamily: "var(--font-dm-mono)", textTransform: "uppercase", fontSize: 10 }}
                    >
                      {k}
                    </span>
                    <span
                      className="text-[rgba(228,240,255,0.65)]"
                      style={{ fontFamily: "var(--font-dm-mono)" }}
                    >
                      {v ?? "—"}
                    </span>
                  </div>
                ))}
              </div>
            </div>

            <div className="flex flex-col items-start gap-3 md:items-end">
              <span
                className="rounded-full px-4 py-1.5 text-[10px] font-bold uppercase tracking-widest"
                style={{
                  fontFamily: "var(--font-dm-mono)",
                  color: validationPassed ? "#00FF9C" : hasWarnings ? "#F59E0B" : "rgba(228,240,255,0.4)",
                  background: validationPassed ? "rgba(0,255,156,0.1)" : hasWarnings ? "rgba(245,158,11,0.1)" : "rgba(255,255,255,0.05)",
                  border: `1px solid ${validationPassed ? "rgba(0,255,156,0.35)" : hasWarnings ? "rgba(245,158,11,0.35)" : "rgba(255,255,255,0.1)"}`,
                }}
              >
                {validationPassed ? "Passed" : hasWarnings ? "Warning" : "Unknown"}
              </span>
              <div className="flex flex-col items-start gap-2 md:items-end">
                <DownloadJsonButton claim={claim} />
                <DownloadCms1500Button claim={claim} />
              </div>
            </div>
          </div>
        </motion.div>

        {loading && (
          <p className="text-sm text-[rgba(228,240,255,0.4)]" style={{ fontFamily: "var(--font-dm-mono)" }}>
            Loading claim…
          </p>
        )}

        {error && !loading && (
          <div
            className="glass-card-sm mb-6 flex items-center gap-2.5 px-4 py-3 text-[#FF3B6B]"
            style={{ borderLeft: "2px solid #FF3B6B" }}
          >
            <TriangleAlert className="size-4 shrink-0" />
            <span className="text-sm" style={{ fontFamily: "var(--font-dm-sans)" }}>{error}</span>
          </div>
        )}

        {!loading && claim && (
          <>
            {/* Codes panels */}
            <div className="mb-8 grid gap-6 md:grid-cols-2">
              <div>
                <p
                  className="mb-4 text-base font-bold uppercase tracking-wider text-[#00D4FF]"
                  style={{ fontFamily: "var(--font-dm-mono)" }}
                >
                  ICD-10 Diagnosis Codes
                </p>
                <div className="space-y-3">
                  {icd10Codes.length === 0
                    ? <p className="text-xs text-[rgba(228,240,255,0.4)]" style={{ fontFamily: "var(--font-dm-mono)" }}>No diagnosis codes.</p>
                    : icd10Codes.map((code, idx) => (
                      <CodeCard
                        key={`${code.code}-${idx}`}
                        index={idx}
                        code={code.code}
                        description={code.description}
                        confidence={code.confidence ?? 0}
                        reasoning={explainability?.icd10_reasoning?.[idx] ?? explainability?.icd10_reasoning?.[0]}
                      />
                    ))
                  }
                </div>
              </div>

              <div>
                <p
                  className="mb-4 text-base font-bold uppercase tracking-wider text-[#00D4FF]"
                  style={{ fontFamily: "var(--font-dm-mono)" }}
                >
                  CPT-4 Procedure Codes
                </p>
                <div className="space-y-3">
                  {cptCodes.length === 0
                    ? <p className="text-xs text-[rgba(228,240,255,0.4)]" style={{ fontFamily: "var(--font-dm-mono)" }}>No procedure codes.</p>
                    : cptCodes.map((code, idx) => (
                      <CodeCard
                        key={`${code.code}-${idx}`}
                        index={idx}
                        code={code.code}
                        description={code.description}
                        confidence={code.confidence ?? 0}
                        reasoning={explainability?.cpt4_reasoning?.[idx] ?? explainability?.cpt4_reasoning?.[0]}
                      />
                    ))
                  }
                </div>
              </div>
            </div>

            {/* Validation warnings */}
            {hasWarnings && claim.validation_errors && (
              <div className="mb-6 space-y-2">
                <p
                  className="mb-3 text-base font-bold uppercase tracking-wider text-[#F59E0B]"
                  style={{ fontFamily: "var(--font-dm-mono)" }}
                >
                  Validation Warnings
                </p>
                {claim.validation_errors.map((ve, idx) => (
                  <div
                    key={`${ve.field}-${idx}`}
                    className="glass-card-sm flex items-start gap-3 px-4 py-3"
                    style={{ borderLeft: "2px solid #F59E0B" }}
                  >
                    <TriangleAlert className="mt-0.5 size-4 shrink-0 text-[#F59E0B]" />
                    <div>
                      <p
                        className="mb-1 text-base font-bold uppercase tracking-wider text-[#F59E0B]"
                        style={{ fontFamily: "var(--font-dm-mono)" }}
                      >
                        {ve.field}
                      </p>
                      <p className="text-sm text-[#E4F0FF]" style={{ fontFamily: "var(--font-dm-sans)", lineHeight: 1.7 }}>
                        {ve.message}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Explainability accordion */}
            {explainability && (
              <div className="mb-10 space-y-3">
                <p
                  className="mb-4 text-base font-bold uppercase tracking-wider text-[#00D4FF]"
                  style={{ fontFamily: "var(--font-dm-mono)" }}
                >
                  Explainability
                </p>
                <AccordionSection title="ICD-10 Reasoning Chain">
                  <p className="text-sm text-[#E4F0FF]" style={{ fontFamily: "var(--font-dm-sans)", lineHeight: 1.7 }}>
                    {explainability.icd10_reasoning_chain || "No reasoning chain provided."}
                  </p>
                </AccordionSection>
                <AccordionSection title="CPT-4 Reasoning Chain">
                  <p className="text-sm text-[#E4F0FF]" style={{ fontFamily: "var(--font-dm-sans)", lineHeight: 1.7 }}>
                    {explainability.cpt4_reasoning_chain || "No reasoning chain provided."}
                  </p>
                </AccordionSection>
                <AccordionSection title="Clinical Citations">
                  {explainability.citations?.length ? (
                    <ul className="space-y-2">
                      {explainability.citations.map((c, idx) => (
                        <li key={idx} className="flex items-start gap-2.5">
                          <Quote className="mt-0.5 size-3 shrink-0 text-[#00D4FF]" />
                          <span className="text-sm text-[#E4F0FF]" style={{ fontFamily: "var(--font-dm-sans)", lineHeight: 1.7 }}>
                            {c}
                          </span>
                        </li>
                      ))}
                    </ul>
                  ) : (
                    <p className="text-xs text-[rgba(228,240,255,0.4)]" style={{ fontFamily: "var(--font-dm-mono)" }}>
                      No citations captured for this claim.
                    </p>
                  )}
                </AccordionSection>
              </div>
            )}
          </>
        )}
      </main>
    </div>
  );
}

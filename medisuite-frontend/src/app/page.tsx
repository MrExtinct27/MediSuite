"use client";

import Link from "next/link";
import type React from "react";
import { motion } from "framer-motion";
import { HeroSection } from "@/components/ui/background-paths";
import { useAuthStore } from "@/store/authStore";
import {
  ArrowRight,
  BrainCircuit,
  Database,
  FileText,
  Layers,
  ShieldCheck,
  Activity,
  Sparkles,
  Stethoscope,
  Bot,
  HeartPulse,
} from "lucide-react";

/* ─── Pipeline node ─────────────────────────────────────────────────── */
function PipelineNode({
  icon,
  title,
  description,
  index,
}: {
  icon: React.ReactNode;
  title: string;
  description: string;
  index: number;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 24 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-40px" }}
      transition={{ delay: index * 0.1, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -4, transition: { duration: 0.2 } }}
      className="group relative"
    >
      <div
        className="glass-card relative overflow-hidden p-5 transition-all duration-300 group-hover:border-[rgba(var(--ms-accent-rgb),0.3)] group-hover:shadow-[0_0_30px_rgba(var(--ms-accent-rgb),0.15)]"
        style={{ borderTop: "2px solid var(--ms-accent)" }}
      >
        {/* Radial glow behind icon on hover */}
        <div className="pointer-events-none absolute -left-8 -top-8 h-32 w-32 rounded-full bg-[var(--ms-accent)] opacity-0 blur-3xl transition-opacity duration-500 group-hover:opacity-[0.08]" />

        <div className="relative">
          <div
            className="mb-3 flex size-10 items-center justify-center rounded-xl border border-[rgba(var(--ms-accent-rgb),0.2)] bg-[rgba(var(--ms-accent-rgb),0.08)] text-[var(--ms-accent)]"
          >
            {icon}
          </div>
          <h3
            className="mb-1.5 text-sm font-bold uppercase tracking-wider text-[var(--ms-text)]"
            style={{ fontFamily: "var(--font-syne)", letterSpacing: "-0.01em" }}
          >
            {title}
          </h3>
          <p className="helper-text">
            {description}
          </p>
          {/* Status indicator */}
          <div className="mt-3 flex items-center gap-1.5">
            <span className="size-1.5 rounded-full bg-[rgba(var(--ms-text-rgb),0.2)]" />
            <span
              className="text-[9px] uppercase tracking-widest text-[rgba(var(--ms-text-rgb),0.35)]"
              style={{ fontFamily: "var(--font-dm-mono)" }}
            >
              Idle
            </span>
          </div>
        </div>
      </div>
    </motion.div>
  );
}

/* ─── Connector arrow ───────────────────────────────────────────────── */
function Connector() {
  return (
    <div className="relative hidden h-px w-10 shrink-0 bg-[rgba(var(--ms-accent-rgb),0.2)] md:block">
      <span className="ms-travel-dot absolute top-1/2 size-2 -translate-y-1/2 rounded-full bg-[var(--ms-accent)] shadow-[0_0_8px_var(--ms-accent)]" />
    </div>
  );
}

/* ─── Tech chip ─────────────────────────────────────────────────────── */
function TechChip({ name, icon, index }: { name: string; icon: React.ReactNode; index: number }) {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      whileInView={{ opacity: 1, scale: 1 }}
      viewport={{ once: true }}
      transition={{ delay: index * 0.04, duration: 0.35 }}
      whileHover={{ scale: 1.04, transition: { duration: 0.15 } }}
      className="glass-card flex items-center gap-2.5 px-4 py-2.5"
    >
      <span className="text-[var(--ms-accent)]">{icon}</span>
      <span
        className="text-xs uppercase tracking-wider text-[rgba(var(--ms-text-rgb),0.7)]"
        style={{ fontFamily: "var(--font-dm-mono)" }}
      >
        {name}
      </span>
    </motion.div>
  );
}

/* ─── Page ──────────────────────────────────────────────────────────── */
export default function Home() {
  const isAuthenticated = useAuthStore((s) => s.isAuthenticated);
  const authHydrated = useAuthStore((s) => s.hydrated);
  const authed = authHydrated && isAuthenticated;

  return (
    <div className="min-h-screen bg-background">

      {/* ── Hero ─────────────────────────────────────────────────── */}
      <HeroSection />

      {/* ── Agent pipeline ─────────────────────────────────────────── */}
      <section className="mx-auto max-w-6xl px-6 py-20">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          transition={{ duration: 0.5 }}
          className="mb-10 flex items-end justify-between"
        >
          <div>
            <p
              className="mb-2 text-[10px] uppercase tracking-widest text-[var(--ms-accent)]"
              style={{ fontFamily: "var(--font-dm-mono)" }}
            >
              — Architecture
            </p>
            <h2
              className="text-3xl font-bold text-[var(--ms-text)]"
              style={{ fontFamily: "var(--font-syne)", letterSpacing: "-0.02em", textTransform: "uppercase" }}
            >
              Agent Pipeline
            </h2>
          </div>
          <span
            className="glass-card-sm px-3 py-1.5 text-[10px] uppercase tracking-widest text-[var(--ms-accent)]"
            style={{ fontFamily: "var(--font-dm-mono)" }}
          >
            LangGraph
          </span>
        </motion.div>

        {/* Desktop linear */}
        <div className="hidden md:flex md:items-stretch md:gap-0">
          <PipelineNode index={0} icon={<FileText className="size-4" />} title="Document Agent" description="Extracts clinical entities from PDF/DOCX/TXT using the selected LLM. Returns structured data with confidence + citations." />
          <Connector />
          <PipelineNode index={1} icon={<BrainCircuit className="size-4" />} title="Coding Agent" description="2-stage RAG: ChromaDB semantic search (HuggingFace) → LLM reranking for ICD-10/CPT-4." />
          <Connector />
          <PipelineNode index={2} icon={<ShieldCheck className="size-4" />} title="Validation Agent" description="Rules + clinical logic checks. Routes back to coding on failure with automatic retry." />
          <Connector />
          <PipelineNode index={3} icon={<Layers className="size-4" />} title="Claim Agent" description="Generates CMS-1500 claim JSON with full explainability and persists to SQLite." />
        </div>

        {/* Mobile stacked */}
        <div className="grid gap-3 md:hidden">
          <PipelineNode index={0} icon={<FileText className="size-4" />} title="Document Agent" description="Extracts clinical entities from PDF/DOCX/TXT using the selected LLM." />
          <PipelineNode index={1} icon={<BrainCircuit className="size-4" />} title="Coding Agent" description="2-stage RAG: ChromaDB + LLM reranking." />
          <PipelineNode index={2} icon={<ShieldCheck className="size-4" />} title="Validation Agent" description="Clinical logic checks with automatic retry." />
          <PipelineNode index={3} icon={<Layers className="size-4" />} title="Claim Agent" description="Generates CMS-1500 with full explainability." />
        </div>
      </section>

      {/* ── Tech stack ─────────────────────────────────────────────── */}
      <section className="mx-auto max-w-6xl px-6 pb-20">
        <motion.div
          initial={{ opacity: 0, y: 16 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mb-8"
        >
          <p
            className="mb-2 text-[10px] uppercase tracking-widest text-[var(--ms-accent)]"
            style={{ fontFamily: "var(--font-dm-mono)" }}
          >
            — Stack
          </p>
          <h2
            className="text-3xl font-bold text-[var(--ms-text)]"
            style={{ fontFamily: "var(--font-syne)", letterSpacing: "-0.02em", textTransform: "uppercase" }}
          >
            Built for Observability
          </h2>
        </motion.div>

        <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
          {[
            { name: "LangGraph", icon: <Bot className="size-4" /> },
            { name: "GPT-4o / GPT-5 / GPT-5.5", icon: <Sparkles className="size-4" /> },
            { name: "ChromaDB", icon: <Database className="size-4" /> },
            { name: "HuggingFace", icon: <Stethoscope className="size-4" /> },
            { name: "FastAPI", icon: <Activity className="size-4" /> },
            { name: "LangSmith", icon: <HeartPulse className="size-4" /> },
            { name: "SQLite", icon: <Database className="size-4" /> },
            { name: "Next.js", icon: <Layers className="size-4" /> },
          ].map((t, i) => <TechChip key={t.name} index={i} name={t.name} icon={t.icon} />)}
        </div>
      </section>

      {/* ── CTA ────────────────────────────────────────────────────── */}
      <section className="mx-auto max-w-6xl px-6 pb-24">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="glass-card relative overflow-hidden p-12"
          style={{ borderTop: "2px solid var(--ms-accent)" }}
        >
          <div className="pointer-events-none absolute -right-24 -top-24 h-72 w-72 rounded-full bg-[var(--ms-accent)] opacity-[0.05] blur-3xl" />
          <div className="pointer-events-none absolute -bottom-24 -left-24 h-72 w-72 rounded-full bg-[var(--ms-success)] opacity-[0.04] blur-3xl" />

          <div className="relative flex flex-col items-start justify-between gap-8 md:flex-row md:items-center">
            <div>
              <h3
                className="mb-3 text-3xl font-bold text-[var(--ms-text)]"
                style={{ fontFamily: "var(--font-syne)", letterSpacing: "-0.02em", textTransform: "uppercase" }}
              >
                Ready to process your first claim?
              </h3>
              <p className="helper-text max-w-md">
                Upload a clinical note and receive ICD‑10/CPT‑4 codes with
                validation and full explainability in seconds.
              </p>
            </div>
            {authed ? (
              <Link
                href="/claims/new"
                className="shrink-0 flex items-center gap-2 rounded-2xl bg-[var(--ms-accent)] px-8 py-3.5 font-bold text-[var(--ms-on-accent)] transition-all duration-200 hover:bg-[var(--ms-accent)]/85 hover:shadow-[0_0_32px_rgba(var(--ms-accent-rgb),0.4)]"
                style={{ fontFamily: "var(--font-syne)", letterSpacing: "-0.01em", textTransform: "uppercase", fontSize: "0.85rem" }}
              >
                Get Started <ArrowRight className="size-4" />
              </Link>
            ) : (
              <div className="flex shrink-0 items-center gap-3">
                <Link
                  href="/login"
                  className="flex items-center rounded-2xl border border-[rgba(var(--ms-accent-rgb),0.3)] bg-[rgba(var(--ms-accent-rgb),0.04)] px-7 py-3.5 font-bold text-[var(--ms-text)] transition-all duration-200 hover:bg-[rgba(var(--ms-accent-rgb),0.08)]"
                  style={{ fontFamily: "var(--font-syne)", letterSpacing: "-0.01em", textTransform: "uppercase", fontSize: "0.85rem" }}
                >
                  Login
                </Link>
                <Link
                  href="/register"
                  className="flex items-center gap-2 rounded-2xl bg-[var(--ms-accent)] px-7 py-3.5 font-bold text-[var(--ms-on-accent)] transition-all duration-200 hover:bg-[var(--ms-accent)]/85 hover:shadow-[0_0_32px_rgba(var(--ms-accent-rgb),0.4)]"
                  style={{ fontFamily: "var(--font-syne)", letterSpacing: "-0.01em", textTransform: "uppercase", fontSize: "0.85rem" }}
                >
                  Sign Up <ArrowRight className="size-4" />
                </Link>
              </div>
            )}
          </div>
        </motion.div>
      </section>

    </div>
  );
}

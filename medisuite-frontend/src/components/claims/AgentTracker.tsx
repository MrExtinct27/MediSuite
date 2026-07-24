"use client";

import { AnimatePresence, motion } from "framer-motion";
import {
  BrainCircuit,
  CircleCheck,
  FileText,
  Layers,
  Loader2,
  ShieldCheck,
  XCircle,
} from "lucide-react";
import type { AgentState, AgentStatus } from "@/lib/types";

export interface AgentTrackerProps {
  agentState: AgentState;
  currentAgent: string | null;
}

type AgentKey = keyof AgentState;

const AGENTS: Array<{ key: AgentKey; name: string; description: string; icon: React.ReactNode }> = [
  {
    key: "document_agent",
    name: "Document Agent",
    description: "Extracting clinical entities, diagnoses, and procedures from document",
    icon: <FileText className="size-4" />,
  },
  {
    key: "coding_agent",
    name: "Coding Agent",
    description: "Running 2-stage RAG: ChromaDB retrieval + LLM reranking",
    icon: <BrainCircuit className="size-4" />,
  },
  {
    key: "validation_agent",
    name: "Validation Agent",
    description: "3-level validation: CMS rules, LLM logic, anomaly detection",
    icon: <ShieldCheck className="size-4" />,
  },
  {
    key: "claim_agent",
    name: "Claim Agent",
    description: "Generating CMS-1500 claim with explainability trails",
    icon: <Layers className="size-4" />,
  },
];

function statusFromProps(agentKey: AgentKey, agentState: AgentState, currentAgent: string | null): AgentStatus {
  if (currentAgent && currentAgent === agentKey) return "running";
  return agentState[agentKey];
}

/* ─── Status dot indicator ───────────────────────────────────────────── */
function AgentStatusDot({ status }: { status: AgentStatus }) {
  if (status === "running") {
    return (
      <div className="relative flex size-2.5 items-center justify-center">
        <span
          className="absolute inline-flex h-full w-full rounded-full bg-[var(--ms-accent)] opacity-60"
          style={{ animation: "neon-pulse 1.5s ease-in-out infinite" }}
        />
        <span className="relative inline-flex size-2 rounded-full bg-[var(--ms-accent)]" />
      </div>
    );
  }
  if (status === "complete") {
    return (
      <span
        className="inline-flex size-2 rounded-full bg-[var(--ms-success)]"
        style={{ boxShadow: "0 0 6px var(--ms-success)" }}
      />
    );
  }
  if (status === "error") {
    return (
      <span
        className="inline-flex size-2 rounded-full bg-[var(--ms-error)]"
        style={{ boxShadow: "0 0 6px var(--ms-error)" }}
      />
    );
  }
  return <span className="inline-flex size-2 rounded-full bg-[rgba(var(--ms-text-rgb),0.15)]" />;
}

/* ─── Status badge ───────────────────────────────────────────────────── */
function StatusBadge({ status }: { status: AgentStatus }) {
  return (
    <AnimatePresence mode="wait" initial={false}>
      {status === "running" && (
        <motion.span
          key="running"
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.85 }}
          transition={{ duration: 0.15 }}
          className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[9px] font-bold uppercase tracking-widest"
          style={{
            fontFamily: "var(--font-dm-mono)",
            color: "var(--ms-accent)",
            background: "rgba(var(--ms-accent-rgb),0.12)",
            border: "1px solid rgba(var(--ms-accent-rgb),0.35)",
          }}
        >
          <Loader2 className="size-3 animate-spin" />
          Processing
        </motion.span>
      )}
      {status === "complete" && (
        <motion.span
          key="complete"
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.85 }}
          transition={{ duration: 0.15 }}
          className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[9px] font-bold uppercase tracking-widest"
          style={{
            fontFamily: "var(--font-dm-mono)",
            color: "var(--ms-success)",
            background: "rgba(var(--ms-success-rgb),0.1)",
            border: "1px solid rgba(var(--ms-success-rgb),0.3)",
          }}
        >
          <CircleCheck className="size-3" />
          Complete
        </motion.span>
      )}
      {status === "error" && (
        <motion.span
          key="error"
          initial={{ opacity: 0, scale: 0.85 }}
          animate={{ opacity: 1, scale: 1 }}
          exit={{ opacity: 0, scale: 0.85 }}
          transition={{ duration: 0.15 }}
          className="inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-[9px] font-bold uppercase tracking-widest"
          style={{
            fontFamily: "var(--font-dm-mono)",
            color: "var(--ms-error)",
            background: "rgba(var(--ms-error-rgb),0.1)",
            border: "1px solid rgba(var(--ms-error-rgb),0.3)",
          }}
        >
          <XCircle className="size-3" />
          Failed
        </motion.span>
      )}
    </AnimatePresence>
  );
}

/* ─── Icon ───────────────────────────────────────────────────────────── */
function StatusIcon({ status, baseIcon }: { status: AgentStatus; baseIcon: React.ReactNode }) {
  if (status === "complete") return <CircleCheck className="size-5 text-[var(--ms-success)]" />;
  if (status === "error")    return <XCircle className="size-5 text-[var(--ms-error)]" />;
  if (status === "running") {
    return (
      <div className="relative">
        <span className="absolute -inset-2 rounded-full bg-[var(--ms-accent)] opacity-20 blur-md" />
        <motion.div
          className="relative text-[var(--ms-accent)]"
          animate={{ opacity: [0.6, 1, 0.6] }}
          transition={{ duration: 1.2, repeat: Infinity, ease: "easeInOut" }}
        >
          {baseIcon}
        </motion.div>
      </div>
    );
  }
  return <div className="text-[rgba(var(--ms-text-rgb),0.3)]">{baseIcon}</div>;
}

/* ─── Main tracker ───────────────────────────────────────────────────── */
export function AgentTracker({ agentState, currentAgent }: AgentTrackerProps) {
  const statuses = AGENTS.map((a) => statusFromProps(a.key, agentState, currentAgent));

  return (
    <div className="space-y-4">
      {AGENTS.map((agent, idx) => {
        const status = statuses[idx];
        const isRunning = status === "running";
        const isComplete = status === "complete";
        const isError = status === "error";

        const nameColor = isComplete ? "var(--ms-success)" : isError ? "var(--ms-error)" : isRunning ? "var(--ms-text)" : "rgba(var(--ms-text-rgb),0.45)";

        const card = (
          <div
            className="relative overflow-hidden rounded-2xl p-4 transition-all duration-300"
            style={{
              background: isRunning
                ? "rgba(var(--ms-accent-rgb),0.04)"
                : isComplete
                  ? "rgba(var(--ms-success-rgb),0.03)"
                  : "rgba(var(--ms-accent-rgb),0.02)",
              border: isRunning
                ? "1px solid rgba(var(--ms-accent-rgb),0.35)"
                : isComplete
                  ? "1px solid rgba(var(--ms-success-rgb),0.2)"
                  : isError
                    ? "1px solid rgba(var(--ms-error-rgb),0.2)"
                    : "1px solid rgba(var(--ms-accent-rgb),0.08)",
              boxShadow: isRunning
                ? "0 0 30px rgba(var(--ms-accent-rgb),0.4), 0 0 60px rgba(var(--ms-accent-rgb),0.1)"
                : isComplete
                  ? "0 0 12px rgba(var(--ms-success-rgb),0.15)"
                  : "none",
            }}
          >
            {/* Radial glow (running only) */}
            {isRunning && (
              <div className="pointer-events-none absolute -left-8 -top-8 h-24 w-24 rounded-full bg-[var(--ms-accent)] opacity-[0.06] blur-2xl" />
            )}

            <div className="relative flex items-start gap-3.5">
              <div
                className="mt-0.5 flex size-9 shrink-0 items-center justify-center rounded-xl"
                style={{
                  border: isRunning
                    ? "1px solid rgba(var(--ms-accent-rgb),0.4)"
                    : "1px solid rgba(var(--ms-accent-rgb),0.1)",
                  background: isRunning
                    ? "rgba(var(--ms-accent-rgb),0.1)"
                    : "rgba(var(--ms-accent-rgb),0.04)",
                }}
              >
                <StatusIcon status={status} baseIcon={agent.icon} />
              </div>

              <div className="min-w-0 flex-1">
                <div className="mb-1 flex items-center justify-between gap-2">
                  <div className="flex items-center gap-2">
                    <AgentStatusDot status={status} />
                    <span
                      className="text-sm font-bold uppercase"
                      style={{
                        fontFamily: "var(--font-syne)",
                        letterSpacing: "-0.01em",
                        color: nameColor,
                      }}
                    >
                      {agent.name}
                    </span>
                  </div>
                  <div className="shrink-0">
                    <StatusBadge status={status} />
                  </div>
                </div>

                <p
                  className="text-xs leading-relaxed"
                  style={{
                    fontFamily: "var(--font-dm-sans)",
                    lineHeight: 1.6,
                    color: isRunning ? "rgba(var(--ms-text-rgb),0.65)" : "rgba(var(--ms-text-rgb),0.35)",
                  }}
                >
                  {agent.description}
                </p>
              </div>
            </div>
          </div>
        );

        return (
          <div key={agent.key}>
            {isRunning ? (
              <motion.div
                initial={{ scale: 1 }}
                animate={{ scale: [1, 1.005, 1] }}
                transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
              >
                {card}
              </motion.div>
            ) : (
              card
            )}
          </div>
        );
      })}
    </div>
  );
}

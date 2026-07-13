"use client";

import { useRouter } from "next/navigation";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import type React from "react";
import { motion } from "framer-motion";
import { CloudUpload, CheckCircle, FileText, ChevronDown } from "lucide-react";
import { toast } from "sonner";

import { AgentTracker } from "@/components/claims/AgentTracker";
import { api } from "@/lib/api";
import type { AgentState, Claim, ClaimStatusResponse, ProcessingStage } from "@/lib/types";
import { useClaimsStore } from "@/store/claimsStore";

const initialAgentState: AgentState = {
  document_agent: "idle",
  coding_agent: "idle",
  validation_agent: "idle",
  claim_agent: "idle",
};

/* ─── Pipeline stage → tracker state ─────────────────────────────────── */
const AGENT_ORDER: Array<keyof AgentState> = [
  "document_agent",
  "coding_agent",
  "validation_agent",
  "claim_agent",
];

const STAGE_TO_AGENT: Record<Exclude<ProcessingStage, "complete">, keyof AgentState> = {
  document: "document_agent",
  coding: "coding_agent",
  validation: "validation_agent",
  claim: "claim_agent",
};

/**
 * Derive the tracker state from the backend processing_stage: agents before the
 * active one are complete, the active one is running, later ones stay dimmed.
 */
function agentStateFromStage(stage: ProcessingStage): {
  state: AgentState;
  current: keyof AgentState | null;
} {
  if (stage === "complete") {
    return {
      state: {
        document_agent: "complete",
        coding_agent: "complete",
        validation_agent: "complete",
        claim_agent: "complete",
      },
      current: null,
    };
  }
  const activeKey = STAGE_TO_AGENT[stage];
  const activeIdx = AGENT_ORDER.indexOf(activeKey);
  const state = { ...initialAgentState };
  AGENT_ORDER.forEach((key, idx) => {
    state[key] = idx < activeIdx ? "complete" : idx === activeIdx ? "running" : "idle";
  });
  return { state, current: activeKey };
}

/** Pull a human-readable message out of an axios/HTTP error, falling back gracefully. */
function extractErrorMessage(err: unknown, fallback: string): string {
  if (typeof err === "object" && err !== null) {
    const anyErr = err as { response?: { data?: { detail?: unknown } }; message?: unknown };
    const detail = anyErr.response?.data?.detail;
    if (typeof detail === "string" && detail.trim()) return detail;
    if (typeof anyErr.message === "string" && anyErr.message.trim()) return anyErr.message;
  }
  return fallback;
}

/* ─── LLM model options (sent as the `llm_model` form field) ─────────── */
const LLM_MODELS: { value: string; tier: string; name: string; note: string }[] = [
  { value: "gpt-4o", tier: "Easy", name: "GPT-4o", note: "Fastest, ~12s" },
  { value: "gpt-5", tier: "Medium", name: "GPT-5", note: "Deeper reasoning, slower" },
  { value: "gpt-5.5", tier: "Hard", name: "GPT-5.5", note: "Highest accuracy, slower" },
];

/* ─── Cyberpunk Input ───────────────────────────────────────────────── */
function CyberInput({
  id,
  label,
  value,
  onChange,
  type = "text",
  placeholder,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  type?: string;
  placeholder?: string;
}) {
  return (
    <div className="space-y-1.5">
      <label
        htmlFor={id}
        className="block text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.5)]"
        style={{ fontFamily: "var(--font-dm-mono)" }}
      >
        {label}
      </label>
      <input
        id={id}
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-full rounded-xl border border-[rgba(0,212,255,0.12)] bg-[rgba(0,212,255,0.04)] px-4 py-3 text-sm text-[#E4F0FF] placeholder-[rgba(228,240,255,0.25)] outline-none transition-all duration-200 focus:border-[#00D4FF] focus:shadow-[0_0_0_1px_rgba(0,212,255,0.3),0_0_12px_rgba(0,212,255,0.1)]"
        style={{ fontFamily: "var(--font-dm-mono)" }}
      />
    </div>
  );
}

/* ─── Cyberpunk Select ──────────────────────────────────────────────── */
function CyberSelect({
  id,
  label,
  value,
  onChange,
  options,
  placeholder,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (v: string) => void;
  options: { value: string; label: string }[];
  placeholder?: string;
}) {
  return (
    <div className="space-y-1.5">
      <label
        htmlFor={id}
        className="block text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.5)]"
        style={{ fontFamily: "var(--font-dm-mono)" }}
      >
        {label}
      </label>
      <div className="relative">
        <select
          id={id}
          value={value}
          onChange={(e) => onChange(e.target.value)}
          className="w-full appearance-none rounded-xl border border-[rgba(0,212,255,0.12)] bg-[rgba(0,212,255,0.04)] px-4 py-3 pr-10 text-sm text-[#E4F0FF] outline-none transition-all duration-200 focus:border-[#00D4FF] focus:shadow-[0_0_0_1px_rgba(0,212,255,0.3),0_0_12px_rgba(0,212,255,0.1)]"
          style={{ fontFamily: "var(--font-dm-mono)" }}
        >
          <option value="" className="bg-[#0a0f1e] text-[rgba(228,240,255,0.4)]">
            {placeholder ?? "Select…"}
          </option>
          {options.map((opt) => (
            <option key={opt.value} value={opt.value} className="bg-[#0a0f1e] text-[#E4F0FF]">
              {opt.label}
            </option>
          ))}
        </select>
        <ChevronDown className="pointer-events-none absolute right-3 top-1/2 size-4 -translate-y-1/2 text-[rgba(0,212,255,0.6)]" />
      </div>
    </div>
  );
}

export default function NewClaimPage() {
  const router = useRouter();

  const [file, setFile] = useState<File | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const [patientName, setPatientName] = useState("");
  const [dob, setDob] = useState("");
  const [insuranceId, setInsuranceId] = useState("");
  const [sex, setSex] = useState("");
  const [address, setAddress] = useState("");
  const [insuranceProvider, setInsuranceProvider] = useState("");
  const [llmModel, setLlmModel] = useState("gpt-4o");
  const [submitting, setSubmitting] = useState(false);
  const [agentState, setAgentState] = useState<AgentState>(initialAgentState);
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);

  // Live-progress polling refs (not state — they must not trigger re-renders).
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const finishedRef = useRef(false);
  const lastStageRef = useRef<ProcessingStage>("document");

  const setActiveClaim = useClaimsStore((s) => s.setActiveClaim);

  // Stop polling if the user navigates away mid-process.
  useEffect(
    () => () => {
      if (pollRef.current) clearInterval(pollRef.current);
    },
    []
  );

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files?.[0]) setFile(e.target.files[0]);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(false);
    if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => setIsDragging(false);

  // Only the uploaded document is required. All patient fields are optional
  // fallbacks, so they must never block submission.
  const canSubmit = useMemo(
    () => !!file && !submitting,
    [file, submitting]
  );

  const handleSubmit = useCallback(
    async (e: React.FormEvent) => {
      e.preventDefault();
      if (!file) {
        toast.error("Please upload a clinical document.");
        return;
      }

      // Generate the id client-side so we can poll for progress while the
      // (blocking) POST runs the multi-agent pipeline server-side.
      const claimId =
        typeof crypto !== "undefined" && "randomUUID" in crypto
          ? crypto.randomUUID()
          : `${Date.now()}-${Math.random().toString(16).slice(2)}`;

      finishedRef.current = false;
      lastStageRef.current = "document";
      setSubmitting(true);
      setAgentState({ ...initialAgentState, document_agent: "running" });
      setCurrentAgent("document_agent");

      const stopPolling = () => {
        if (pollRef.current) {
          clearInterval(pollRef.current);
          pollRef.current = null;
        }
      };

      const applyStage = (stage: ProcessingStage) => {
        lastStageRef.current = stage;
        const { state, current } = agentStateFromStage(stage);
        setAgentState(state);
        setCurrentAgent(current);
      };

      const finishComplete = () => {
        if (finishedRef.current) return;
        finishedRef.current = true;
        stopPolling();
        setAgentState(agentStateFromStage("complete").state);
        setCurrentAgent(null);
        setActiveClaim({
          claim_id: claimId,
          form_type: "CMS-1500",
          generated_at: new Date().toISOString(),
          patient: {
            name: patientName,
            dob,
            insurance_id: insuranceId,
            sex: sex || null,
            address: address || null,
            insurance_provider: insuranceProvider || null,
          },
          service_date: "",
          provider_name: "",
          facility_name: "",
          diagnosis_codes: [],
          procedure_codes: [],
          validation_status: "",
          validation_errors: [],
          explainability: { icd10_reasoning: [], cpt4_reasoning: [], citations: [], icd10_reasoning_chain: "", cpt4_reasoning_chain: "" },
          processing_status: "complete",
        } as Claim);
        toast.success("Claim processed successfully.");
        router.push(`/claims/${claimId}`);
      };

      const finishError = (err: unknown) => {
        if (finishedRef.current) return;
        finishedRef.current = true;
        stopPolling();
        // Mark the agent that was running as failed; keep prior ones complete.
        const { state, current } = agentStateFromStage(lastStageRef.current);
        if (current) state[current] = "error";
        setAgentState(state);
        setCurrentAgent(null);
        setSubmitting(false);
        toast.error(extractErrorMessage(err, "Failed to process claim. Please try again."));
      };

      // Poll live progress every second. Runs in parallel with the POST below.
      let pollFailures = 0;
      pollRef.current = setInterval(async () => {
        if (finishedRef.current) return;
        try {
          const { data } = await api.get<ClaimStatusResponse>(`/claims/${claimId}/status`);
          pollFailures = 0;
          if (finishedRef.current) return;
          applyStage(data.processing_stage);
          if (data.complete) finishComplete();
        } catch (err) {
          const status = (err as { response?: { status?: number } })?.response?.status;
          // 404 = the claim row isn't registered yet (first ticks); keep waiting.
          if (status === 404) return;
          pollFailures += 1;
          if (pollFailures >= 6) {
            finishError(new Error("Lost connection to the server while processing the claim."));
          }
        }
      }, 1000);

      // Fire the processing request. It blocks until the pipeline finishes, but
      // polling drives the live tracker in the meantime.
      try {
        const formData = new FormData();
        formData.append("claim_id", claimId);
        formData.append("file", file);
        formData.append("patient_name", patientName);
        formData.append("patient_dob", dob);
        formData.append("patient_insurance_id", insuranceId);
        formData.append("patient_sex", sex);
        formData.append("patient_address", address);
        formData.append("insurance_provider", insuranceProvider);
        formData.append("llm_model", llmModel);

        await api.post("/claims/process", formData, {
          headers: { "Content-Type": "multipart/form-data" },
        });
        // Pipeline finished server-side — finalize even if the last poll tick
        // hasn't observed "complete" yet.
        finishComplete();
      } catch (err) {
        finishError(err);
      }
    },
    [file, patientName, dob, insuranceId, sex, address, insuranceProvider, llmModel, setActiveClaim, router]
  );

  return (
    <div className="min-h-screen bg-background">
      <main className="mx-auto w-full max-w-6xl px-6 py-10">

        {/* Page heading — full width, matches Dashboard heading style */}
        <motion.div
          initial={{ opacity: 0, y: -12 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <p
            className="mb-1 text-[10px] uppercase tracking-widest text-[#00D4FF]"
            style={{ fontFamily: "var(--font-dm-mono)" }}
          >
            — New claim
          </p>
          <h1
            className="text-3xl font-bold uppercase text-[#E4F0FF]"
            style={{ fontFamily: "var(--font-syne)", letterSpacing: "-0.02em" }}
          >
            Submit Claim
          </h1>
          <p
            className="mt-2 text-sm text-[rgba(228,240,255,0.5)]"
            style={{ fontFamily: "var(--font-dm-sans)", lineHeight: 1.7 }}
          >
            Upload a clinical note and patient details to run the multi-agent pipeline.
          </p>
        </motion.div>

        <div className="mt-8 flex flex-col gap-8 md:flex-row md:items-start">

        {/* Left column — form */}
        <section className="w-full md:w-3/5">
          <form onSubmit={handleSubmit} className="space-y-6">

            {/* Upload zone */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.15, duration: 0.5 }}
              className="glass-card p-5"
            >
              <p
                className="mb-4 text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.45)]"
                style={{ fontFamily: "var(--font-dm-mono)" }}
              >
                01 / Upload Document
              </p>

              <div
                onClick={() => fileInputRef.current?.click()}
                onDrop={handleDrop}
                onDragOver={handleDragOver}
                onDragLeave={handleDragLeave}
                className="relative cursor-pointer rounded-2xl border-2 border-dashed p-10 text-center transition-all duration-200"
                style={{
                  borderColor: isDragging ? "#00D4FF" : file ? "#00FF9C" : "rgba(0,212,255,0.3)",
                  background: isDragging
                    ? "rgba(0,212,255,0.07)"
                    : file
                      ? "rgba(0,255,156,0.04)"
                      : "rgba(0,212,255,0.02)",
                }}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".pdf,.docx,.txt,.doc"
                  onChange={handleFileChange}
                  className="hidden"
                />

                {file ? (
                  <div className="flex flex-col items-center gap-3">
                    <CheckCircle className="size-8 text-[#00FF9C]" />
                    <span
                      className="text-sm text-[#00FF9C]"
                      style={{ fontFamily: "var(--font-dm-mono)" }}
                    >
                      {file.name}
                    </span>
                    <span
                      className="text-[10px] text-[rgba(228,240,255,0.4)]"
                      style={{ fontFamily: "var(--font-dm-mono)" }}
                    >
                      {(file.size / 1024).toFixed(1)} KB — Click to change
                    </span>
                  </div>
                ) : (
                  <div className="flex flex-col items-center gap-3">
                    <CloudUpload className="size-10 text-[#00D4FF] opacity-70" />
                    <p
                      className="text-sm text-[rgba(228,240,255,0.7)]"
                      style={{ fontFamily: "var(--font-dm-sans)" }}
                    >
                      Drag & drop clinical document
                    </p>
                    <p
                      className="text-[11px] text-[rgba(228,240,255,0.35)]"
                      style={{ fontFamily: "var(--font-dm-mono)" }}
                    >
                      PDF, DOCX, TXT accepted
                    </p>
                  </div>
                )}
              </div>
            </motion.div>

            {/* Patient details */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.25, duration: 0.5 }}
              className="glass-card p-5 space-y-4"
            >
              <div className="space-y-1.5">
                <p
                  className="text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.45)]"
                  style={{ fontFamily: "var(--font-dm-mono)" }}
                >
                  02 / Patient Details
                </p>
                <p
                  className="text-sm text-[rgba(228,240,255,0.5)]"
                  style={{ fontFamily: "var(--font-dm-sans)", lineHeight: 1.7 }}
                >
                  Optional. The clinical note is used first; these fields fill in only what the note does not contain.
                </p>
              </div>

              <CyberInput
                id="patient_name"
                label="Patient Name"
                value={patientName}
                onChange={setPatientName}
                placeholder="Jane Doe"
              />
              <CyberInput
                id="dob"
                label="Date of Birth"
                value={dob}
                onChange={setDob}
                type="date"
              />
              <CyberInput
                id="insurance_id"
                label="Insurance ID"
                value={insuranceId}
                onChange={setInsuranceId}
                placeholder="BCB123456"
              />
              <CyberSelect
                id="patient_sex"
                label="Patient Sex"
                value={sex}
                onChange={setSex}
                placeholder="Select sex…"
                options={[
                  { value: "Male", label: "Male" },
                  { value: "Female", label: "Female" },
                  { value: "Other", label: "Other" },
                ]}
              />
              <CyberInput
                id="patient_address"
                label="Patient Address"
                value={address}
                onChange={setAddress}
                placeholder="123 Main St, City, State ZIP"
              />
              <CyberInput
                id="insurance_provider"
                label="Insurance Provider"
                value={insuranceProvider}
                onChange={setInsuranceProvider}
                placeholder="e.g. Metlife, Aetna"
              />
            </motion.div>

            {/* Model selection */}
            <motion.div
              initial={{ opacity: 0, y: 16 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.32, duration: 0.5 }}
              className="glass-card p-5 space-y-4"
            >
              <div className="space-y-1.5">
                <p
                  className="text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.45)]"
                  style={{ fontFamily: "var(--font-dm-mono)" }}
                >
                  03 / Model Selection
                </p>
                <p
                  className="text-sm text-[rgba(228,240,255,0.5)]"
                  style={{ fontFamily: "var(--font-dm-sans)", lineHeight: 1.7 }}
                >
                  Choose based on clinical note complexity. Higher tiers use more reasoning but take longer to process.
                </p>
              </div>

              <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
                {LLM_MODELS.map((m) => {
                  const selected = llmModel === m.value;
                  return (
                    <button
                      key={m.value}
                      type="button"
                      onClick={() => setLlmModel(m.value)}
                      aria-pressed={selected}
                      className="rounded-xl border px-4 py-3 text-left outline-none transition-all duration-200"
                      style={{
                        fontFamily: "var(--font-dm-mono)",
                        borderColor: selected ? "#00D4FF" : "rgba(0,212,255,0.12)",
                        background: selected ? "rgba(0,212,255,0.08)" : "rgba(0,212,255,0.04)",
                        boxShadow: selected
                          ? "0 0 0 1px rgba(0,212,255,0.3), 0 0 12px rgba(0,212,255,0.1)"
                          : "none",
                      }}
                    >
                      <span
                        className="block text-[10px] uppercase tracking-widest"
                        style={{ color: selected ? "#00D4FF" : "rgba(228,240,255,0.5)" }}
                      >
                        {m.tier}
                      </span>
                      <span className="mt-1 block text-sm text-[#E4F0FF]">{m.name}</span>
                      <span className="mt-1 block text-[10px] text-[rgba(228,240,255,0.4)]">
                        {m.note}
                      </span>
                    </button>
                  );
                })}
              </div>
            </motion.div>

            {/* Submit */}
            <motion.button
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.4, duration: 0.4 }}
              type="submit"
              disabled={!canSubmit}
              className="w-full rounded-2xl py-4 font-bold uppercase tracking-wider text-black transition-all duration-200 disabled:cursor-not-allowed disabled:opacity-40"
              style={{
                fontFamily: "var(--font-syne)",
                letterSpacing: "-0.01em",
                background: canSubmit ? "#00D4FF" : "rgba(0,212,255,0.3)",
                boxShadow: canSubmit ? "0 0 24px rgba(0,212,255,0.3)" : "none",
              }}
            >
              {submitting ? (
                <span className="inline-flex items-center justify-center gap-2">
                  <span className="size-4 animate-spin rounded-full border-2 border-black/30 border-t-black" />
                  Processing Claim...
                </span>
              ) : (
                "Process Claim →"
              )}
            </motion.button>
          </form>
        </section>

        {/* Right column — Pipeline tracker */}
        <section className="w-full md:w-2/5">
          <motion.div
            initial={{ opacity: 0, x: 24 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            className="glass-card sticky top-24 p-5"
          >
            <div className="mb-5 flex items-center gap-2.5 border-b border-[rgba(0,212,255,0.08)] pb-4">
              <FileText className="size-4 text-[#00D4FF]" />
              <p
                className="text-[10px] uppercase tracking-widest text-[rgba(228,240,255,0.45)]"
                style={{ fontFamily: "var(--font-dm-mono)" }}
              >
                Pipeline Status
              </p>
            </div>
            <AgentTracker agentState={agentState} currentAgent={currentAgent} />
          </motion.div>
        </section>

        </div>
      </main>
    </div>
  );
}

export interface DiagnosisCode {
  code: string;
  description: string;
  confidence: number;
}

export interface ProcedureCode {
  code: string;
  description: string;
  confidence: number;
}

export interface ValidationError {
  field: string;
  message: string;
  severity: string;
}

export interface Explainability {
  icd10_reasoning: string[];
  cpt4_reasoning: string[];
  citations: string[];
  icd10_reasoning_chain: string;
  cpt4_reasoning_chain: string;
}

export interface Claim {
  claim_id: string;
  form_type: string;
  generated_at: string;
  patient: {
    name: string;
    dob: string;
    insurance_id: string;
    sex?: string | null;
    address?: string | null;
    insurance_provider?: string | null;
  };
  service_date: string;
  provider_name: string;
  facility_name: string;
  diagnosis_codes: DiagnosisCode[];
  procedure_codes: ProcedureCode[];
  validation_status: string;
  validation_errors: ValidationError[];
  explainability: Explainability;
  processing_status: string;
}

export interface ClaimSummary {
  claim_id: string;
  patient_name: string | null;
  service_date: string | null;
  validation_status: string; // "passed" | "failed"
  avg_confidence: number;    // 0.0 – 1.0
}

export type AgentStatus = "idle" | "running" | "complete" | "error";

export interface AgentState {
  document_agent: AgentStatus;
  coding_agent: AgentStatus;
  validation_agent: AgentStatus;
  claim_agent: AgentStatus;
}

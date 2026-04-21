import { create } from "zustand";

import type { AgentState, Claim } from "@/lib/types";

interface ClaimsStore {
  claims: Claim[];
  activeClaim?: Claim;
  agentState: AgentState;
  setClaims: (claims: Claim[]) => void;
  setActiveClaim: (claim?: Claim) => void;
  setAgentState: (partial: Partial<AgentState>) => void;
}

const idleAgents: AgentState = {
  document_agent: "idle",
  coding_agent: "idle",
  validation_agent: "idle",
  claim_agent: "idle",
};

export const useClaimsStore = create<ClaimsStore>((set) => ({
  claims: [],
  activeClaim: undefined,
  agentState: idleAgents,
  setClaims: (claims) => set({ claims }),
  setActiveClaim: (claim) => set({ activeClaim: claim }),
  setAgentState: (partial) =>
    set((s) => ({ agentState: { ...s.agentState, ...partial } })),
}));

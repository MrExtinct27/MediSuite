"""
LangGraph StateGraph definition for the MediSuite AI Agent pipeline.

Flow:
  document_agent → coding_agent → validation_agent
                                        ↓
                              route_after_validation
                             /                      \\
                     claim_agent              coding_agent (retry, max 2)
                          ↓                          ↓
                        END              validation_agent → … → END
"""

from __future__ import annotations

from langgraph.graph import END, StateGraph

from agents.claim_agent import claim_agent
from agents.coding_agent import coding_agent
from agents.document_agent import document_agent
from agents.validation_agent import validation_agent
from orchestrator.router import route_after_validation
from orchestrator.state import ClaimState


def _increment_revalidation(state: ClaimState) -> ClaimState:
    """
    Thin wrapper inserted on the retry edge: increments revalidation_count
    before coding_agent is called again, so the router can enforce the limit.
    """
    count = (state.get("revalidation_count") or 0) + 1
    return {**state, "revalidation_count": count}


def build_graph() -> StateGraph:
    """Build and return the compiled LangGraph StateGraph."""
    workflow = StateGraph(ClaimState)

    # ---- nodes -------------------------------------------------------
    workflow.add_node("document_agent",   document_agent)
    workflow.add_node("coding_agent",     coding_agent)
    workflow.add_node("validation_agent", validation_agent)
    workflow.add_node("claim_agent",      claim_agent)
    # Retry shim: bumps revalidation_count before re-entering coding_agent
    workflow.add_node("increment_retry",  _increment_revalidation)

    # ---- entry point -------------------------------------------------
    workflow.set_entry_point("document_agent")

    # ---- linear edges ------------------------------------------------
    workflow.add_edge("document_agent",   "coding_agent")
    workflow.add_edge("coding_agent",     "validation_agent")
    workflow.add_edge("increment_retry",  "coding_agent")
    workflow.add_edge("claim_agent",      END)

    # ---- conditional edge after validation ---------------------------
    workflow.add_conditional_edges(
        "validation_agent",
        route_after_validation,
        {
            "claim_agent":   "claim_agent",
            "coding_agent":  "increment_retry",  # goes through counter shim first
            "__end__":       END,
        },
    )

    return workflow


# Compiled graph — import this in the API layer
graph = build_graph().compile()

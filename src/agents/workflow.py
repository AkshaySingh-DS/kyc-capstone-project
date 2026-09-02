from langgraph.graph import StateGraph, START, END

from src.agents.state import KYCState

from src.agents.document_agent import (
    DocumentVerificationAgent
)

from src.agents.identity_agent import (
    IdentityVerificationAgent
)

from src.agents.sanctions_agent import (
    SanctionsScreeningAgent
)

from src.agents.policy_agent import (
    PolicyAgent
)

from src.agents.decision_agent import (
    DecisionAgent
)

import mlflow

from config.mlflow_config import init_mlflow

# ------------------------------------------------------------
# Initialize MLflow
# ------------------------------------------------------------
init_mlflow()

# ============================================================
# CREATE AGENTS
# ============================================================

document_agent = DocumentVerificationAgent()
identity_agent = IdentityVerificationAgent()
sanctions_agent = SanctionsScreeningAgent()
policy_agent = PolicyAgent()
decision_agent = DecisionAgent()


# ============================================================
# DOCUMENT AGENT NODE
# ============================================================
@mlflow.trace(name="Document Agent")
def document_node(state: KYCState):
    result = document_agent.run(state)

    document_result = result.get("document_result", {})

    # If documents are incomplete, create an early
    # MORE_DOCUMENTS result so the final output is not empty.
    if document_result.get("status") == "INCOMPLETE":

        decision_result = {
            "applicant_id": state["applicant_id"],
            "decision": "MORE_DOCUMENTS",
            "confidence": "HIGH",
            "reasons": [
                "Required KYC documents are missing.",
                (
                    "The application cannot proceed to identity "
                    "verification until all required documents are available."
                ),
            ],
            "missing_documents": document_result.get(
                "missing_documents", []
            ),
            "documents_found": document_result.get(
                "documents_found", []
            ),
            "next_action": "END",
        }

        return {
            "document_result": document_result,
            "decision_result": decision_result,
            "next_action": "MORE_DOCUMENTS",
            "messages": result.get("messages", []),
        }

    return {
        "document_result": document_result,
        "next_action": result.get("next_action"),
        "messages": result.get("messages", []),
    }


# ============================================================
# ROUTE AFTER DOCUMENT AGENT
# ============================================================

def route_after_document(state: KYCState):
    document_result = state.get("document_result", {})

    if document_result.get("status") == "INCOMPLETE":
        return "end"

    return "identity"


# ============================================================
# IDENTITY AGENT NODE
# ============================================================
@mlflow.trace(name="identity Agent")
def identity_node(state: KYCState):

    result = identity_agent.run(
        applicant_id=state["applicant_id"],
        expected_profile=state["expected_profile"],
        document_result=state["document_result"],
        photo_paths=state.get("photo_paths", {}),
    )

    return {
        "identity_result": result.get(
            "identity_result", {}
        ),
        "next_action": result.get("next_action"),
        "messages": [
            result.get("message", "")
        ],
    }


# ============================================================
# SANCTIONS AGENT NODE
# ============================================================
@mlflow.trace(name="sanctions Agent")
def sanctions_node(state: KYCState):

    result = sanctions_agent.run(
        applicant_id=state["applicant_id"],
        identity_result=state.get(
            "identity_result", {}
        ),
    )

    return {
        "sanctions_result": result.get(
            "sanctions_result", {}
        ),
        "next_action": result.get("next_action"),
        "messages": [
            result.get("message", "")
        ],
    }


# ============================================================
# SANCTIONS REVIEW NODE
#
# This node is NOT another agent.
# It simply marks that the one-time feedback loop
# has been requested.
# ============================================================

def sanctions_review_node(state: KYCState):

    applicant_id = state["applicant_id"]

    return {
        "sanctions_review_attempted": True,
        "messages": [
            (
                f"Workflow: requesting additional sanctions "
                f"evidence for {applicant_id}."
            )
        ],
    }


# ============================================================
# POLICY AGENT NODE
# ============================================================
@mlflow.trace(name="policy Agent")
def policy_node(state: KYCState):

    result = policy_agent.run(
        applicant_id=state["applicant_id"],
        document_result=state.get(
            "document_result", {}
        ),
        identity_result=state.get(
            "identity_result", {}
        ),
        sanctions_result=state.get(
            "sanctions_result", {}
        ),
    )

    return {
        "policy_result": result.get(
            "policy_result", {}
        ),
        "next_action": result.get("next_action"),
        "messages": [
            result.get("message", "")
        ],
    }


# ============================================================
# DECISION AGENT NODE
# ============================================================
@mlflow.trace(name="decision Agent")
def decision_node(state: KYCState):

    result = decision_agent.run(
        applicant_id=state["applicant_id"],
        document_result=state.get(
            "document_result", {}
        ),
        identity_result=state.get(
            "identity_result", {}
        ),
        sanctions_result=state.get(
            "sanctions_result", {}
        ),
        policy_result=state.get(
            "policy_result", {}
        ),
    )

    return {
        "decision_result": result.get(
            "decision_result", {}
        ),
        "next_action": result.get("next_action"),
        "messages": [
            result.get("message", "")
        ],
    }


# ============================================================
# ROUTE AFTER DECISION AGENT
#
# REVIEW + LOW confidence can trigger ONE
# sanctions re-check.
#
# After the re-check, the workflow MUST END.
# This prevents an infinite LangGraph loop.
# ============================================================

def route_after_decision(state: KYCState):

    decision_result = state.get(
        "decision_result", {}
    )

    decision = decision_result.get(
        "decision"
    )

    confidence = decision_result.get(
        "confidence"
    )

    review_attempted = state.get(
        "sanctions_review_attempted",
        False
    )

    # Final decisions
    if decision in (
        "APPROVE",
        "REJECT",
        "MORE_DOCUMENTS",
    ):
        return "end"

    # One-time low-confidence feedback loop
    if (
        decision == "REVIEW"
        and confidence == "LOW"
        and not review_attempted
    ):
        return "sanctions_review"

    # If review has already been attempted,
    # stop the workflow.
    return "end"


# ============================================================
# BUILD LANGGRAPH WORKFLOW
# ============================================================

def build_workflow():

    workflow = StateGraph(KYCState)

    # --------------------------------------------------------
    # Add nodes
    # --------------------------------------------------------

    workflow.add_node(
        "document_agent",
        document_node
    )

    workflow.add_node(
        "identity_agent",
        identity_node
    )

    workflow.add_node(
        "sanctions_agent",
        sanctions_node
    )

    workflow.add_node(
        "sanctions_review",
        sanctions_review_node
    )

    workflow.add_node(
        "policy_agent",
        policy_node
    )

    workflow.add_node(
        "decision_agent",
        decision_node
    )

    # --------------------------------------------------------
    # Start → Document Agent
    # --------------------------------------------------------

    workflow.add_edge(
        START,
        "document_agent"
    )

    # --------------------------------------------------------
    # Document Agent routing
    # --------------------------------------------------------

    workflow.add_conditional_edges(
        "document_agent",
        route_after_document,
        {
            "identity": "identity_agent",
            "end": END,
        },
    )

    # --------------------------------------------------------
    # Normal workflow
    # --------------------------------------------------------

    workflow.add_edge(
        "identity_agent",
        "sanctions_agent"
    )

    workflow.add_edge(
        "sanctions_agent",
        "policy_agent"
    )

    workflow.add_edge(
        "policy_agent",
        "decision_agent"
    )

    # --------------------------------------------------------
    # Decision Agent routing
    #
    # REVIEW + LOW → one-time sanctions review
    # --------------------------------------------------------

    workflow.add_conditional_edges(
        "decision_agent",
        route_after_decision,
        {
            "end": END,
            "sanctions_review": "sanctions_review",
        },
    )

    # --------------------------------------------------------
    # Feedback loop
    #
    # sanctions_review → sanctions_agent
    # --------------------------------------------------------

    workflow.add_edge(
        "sanctions_review",
        "sanctions_agent"
    )

    # --------------------------------------------------------
    # Compile
    # --------------------------------------------------------

    return workflow.compile()


# ============================================================
# CREATE WORKFLOW
# ============================================================

kyc_workflow = build_workflow()

# ============================================================
# CREATE WORKFLOW GRAPH FOR VISUALIZATION
# ============================================================

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DIAGRAM_DIR = PROJECT_ROOT / "docs" / "diagrams"
DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
png_data = kyc_workflow.get_graph().draw_mermaid_png()
output_file = DIAGRAM_DIR / "kyc_workflow.png"

with open(output_file, "wb") as f:
    f.write(png_data)
print("Saved: kyc_workflow.png")


@mlflow.trace(name="KYC Workflow")
def run_kyc(initial_state):
    return kyc_workflow.invoke(initial_state)

# ============================================================
# MAIN PROGRAM
# ============================================================

if __name__ == "__main__":

    import json

    print("\n========================================")
    print("KYC MULTI-AGENT WORKFLOW")
    print("========================================\n")

    # --------------------------------------------------------
    # Ask for applicant ID
    # --------------------------------------------------------

    applicant_id = input(
        "Enter applicant ID (e.g. APP-001): "
    ).strip().upper()

    if not applicant_id:
        print("Applicant ID cannot be empty.")
        raise SystemExit(1)

    # --------------------------------------------------------
    # Load applicant profiles
    # --------------------------------------------------------

    applicants_file = (
        "synthetic_documents/applicants.json"
    )

    try:

        with open(
            applicants_file,
            "r",
            encoding="utf-8"
        ) as file:

            applicants = json.load(file)

    except FileNotFoundError:

        print(
            f"Applicant file not found: "
            f"{applicants_file}"
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Find applicant
    # --------------------------------------------------------

    expected_profile = next(
        (
            applicant
            for applicant in applicants
            if applicant.get("applicant_id")
            == applicant_id
        ),
        None,
    )

    if expected_profile is None:

        print(
            f"Applicant '{applicant_id}' "
            f"was not found."
        )

        raise SystemExit(1)

    # --------------------------------------------------------
    # Initial LangGraph state
    # --------------------------------------------------------

    initial_state = {

        "applicant_id": applicant_id,

        "expected_profile": expected_profile,

        "document_paths": {},

        "photo_paths": {},

        "document_result": {},

        "identity_result": {},

        "sanctions_result": {},

        "policy_result": {},

        "decision_result": {},

        "next_action": None,

        "messages": [],

        "error": None,

        # Important:
        # Allows only ONE sanctions feedback loop.
        "sanctions_review_attempted": False,
    }

    # --------------------------------------------------------
    # Execute workflow
    # --------------------------------------------------------

    try:

        final_state = run_kyc(initial_state)

    except Exception as e:

        print("\nWorkflow execution failed:")
        print(str(e))

        raise SystemExit(1)

    # --------------------------------------------------------
    # Display final result
    # --------------------------------------------------------

    print("\n========================================")
    print("FINAL KYC RESULT")
    print("========================================\n")

    decision_result = final_state.get(
        "decision_result",
        {}
    )

    print(
        json.dumps(
            decision_result,
            indent=2,
            ensure_ascii=False
        )
    )

    # --------------------------------------------------------
    # Display workflow information
    # --------------------------------------------------------

    print("\n========================================")
    print("WORKFLOW COMPLETED")
    print("========================================")

    print(
        f"\nSanctions review attempted: "
        f"{final_state.get('sanctions_review_attempted', False)}"
    )

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


# =========================================================
# Create Agents
# =========================================================

document_agent = DocumentVerificationAgent()
identity_agent = IdentityVerificationAgent()
sanctions_agent = SanctionsScreeningAgent()
policy_agent = PolicyAgent()
decision_agent = DecisionAgent()


# =========================================================
# DOCUMENT AGENT
# =========================================================

def document_node(state: KYCState):

    result = document_agent.run(state)

    return {
        "document_result": result.get(
            "document_result",
            {}
        ),
        "next_action": result.get(
            "next_action"
        ),
        "messages": result.get(
            "messages",
            []
        ),
    }


# =========================================================
# DOCUMENT ROUTER
# =========================================================

def route_after_document(state: KYCState):

    document_result = state.get(
        "document_result",
        {}
    )

    if document_result.get("status") == "INCOMPLETE":

        return "end"

    return "identity"


# =========================================================
# IDENTITY AGENT
# =========================================================

def identity_node(state: KYCState):

    result = identity_agent.run(
        applicant_id=state["applicant_id"],
        expected_profile=state["expected_profile"],
        document_result=state["document_result"],
        photo_paths=state.get(
            "photo_paths",
            {}
        ),
    )

    return {
        "identity_result": result.get(
            "identity_result",
            {}
        ),
        "next_action": result.get(
            "next_action"
        ),
        "messages": [
            result.get(
                "message",
                ""
            )
        ],
    }


# =========================================================
# SANCTIONS AGENT
# =========================================================

def sanctions_node(state: KYCState):

    result = sanctions_agent.run(
        applicant_id=state["applicant_id"],
        identity_result=state.get(
            "identity_result",
            {}
        ),
    )

    return {
        "sanctions_result": result.get(
            "sanctions_result",
            {}
        ),
        "next_action": result.get(
            "next_action"
        ),
        "messages": [
            result.get(
                "message",
                ""
            )
        ],
    }


# =========================================================
# POLICY AGENT
# =========================================================

def policy_node(state: KYCState):

    result = policy_agent.run(
        applicant_id=state["applicant_id"],
        document_result=state.get(
            "document_result",
            {}
        ),
        identity_result=state.get(
            "identity_result",
            {}
        ),
        sanctions_result=state.get(
            "sanctions_result",
            {}
        ),
    )

    return {
        "policy_result": result.get(
            "policy_result",
            {}
        ),
        "next_action": result.get(
            "next_action"
        ),
        "messages": [
            result.get(
                "message",
                ""
            )
        ],
    }


# =========================================================
# DECISION AGENT
# =========================================================

def decision_node(state: KYCState):

    result = decision_agent.run(
        applicant_id=state["applicant_id"],
        document_result=state.get(
            "document_result",
            {}
        ),
        identity_result=state.get(
            "identity_result",
            {}
        ),
        sanctions_result=state.get(
            "sanctions_result",
            {}
        ),
        policy_result=state.get(
            "policy_result",
            {}
        ),
    )

    return {
        "decision_result": result.get(
            "decision_result",
            {}
        ),
        "next_action": result.get(
            "next_action"
        ),
        "messages": [
            result.get(
                "message",
                ""
            )
        ],
    }


# =========================================================
# DECISION ROUTER
# =========================================================

def route_after_decision(state: KYCState):

    decision_result = state.get(
        "decision_result",
        {}
    )

    decision = decision_result.get(
        "decision"
    )

    confidence = decision_result.get(
        "confidence"
    )

    # -----------------------------------------------------
    # Final decisions
    # -----------------------------------------------------

    if decision in (
        "APPROVE",
        "REJECT"
    ):

        return "end"

    # -----------------------------------------------------
    # REVIEW + LOW confidence
    #
    # Ask Sanctions Agent for additional evidence.
    # -----------------------------------------------------

    if (
        decision == "REVIEW"
        and confidence == "LOW"
    ):

        return "sanctions_review"

    # -----------------------------------------------------
    # REVIEW but confidence is not LOW.
    #
    # For this simple college demo we finish with REVIEW.
    # -----------------------------------------------------

    return "end"


# =========================================================
# BUILD LANGGRAPH
# =========================================================

def build_workflow():

    workflow = StateGraph(KYCState)

    # -----------------------------------------------------
    # Add nodes
    # -----------------------------------------------------

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
        "policy_agent",
        policy_node
    )

    workflow.add_node(
        "decision_agent",
        decision_node
    )

    # -----------------------------------------------------
    # START → DOCUMENT
    # -----------------------------------------------------

    workflow.add_edge(
        START,
        "document_agent"
    )

    # -----------------------------------------------------
    # DOCUMENT → IDENTITY or END
    # -----------------------------------------------------

    workflow.add_conditional_edges(
        "document_agent",
        route_after_document,
        {
            "identity": "identity_agent",
            "end": END,
        }
    )

    # -----------------------------------------------------
    # IDENTITY → SANCTIONS
    # -----------------------------------------------------

    workflow.add_edge(
        "identity_agent",
        "sanctions_agent"
    )

    # -----------------------------------------------------
    # SANCTIONS → POLICY
    # -----------------------------------------------------

    workflow.add_edge(
        "sanctions_agent",
        "policy_agent"
    )

    # -----------------------------------------------------
    # POLICY → DECISION
    # -----------------------------------------------------

    workflow.add_edge(
        "policy_agent",
        "decision_agent"
    )

    # -----------------------------------------------------
    # DECISION ROUTING
    # -----------------------------------------------------

    workflow.add_conditional_edges(
        "decision_agent",
        route_after_decision,
        {
            "end": END,

            # Low-confidence REVIEW
            # goes back to Sanctions Agent.
            "sanctions_review": "sanctions_agent",
        }
    )

    return workflow.compile()


# =========================================================
# Create compiled workflow
# =========================================================

kyc_workflow = build_workflow()


# =========================================================
# Simple Test
# =========================================================

if __name__ == "__main__":

    import json

    print("\n========================================")
    print("KYC MULTI-AGENT WORKFLOW")
    print("========================================\n")

    applicant_id = input(
        "Enter applicant ID (e.g. APP-001): "
    ).strip().upper()

    if not applicant_id:

        print(
            "Applicant ID cannot be empty."
        )

        raise SystemExit(1)

    # -----------------------------------------------------
    # Load applicant profile
    # -----------------------------------------------------

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

    # -----------------------------------------------------
    # Find applicant
    # -----------------------------------------------------

    expected_profile = next(
        (
            applicant
            for applicant in applicants
            if applicant.get(
                "applicant_id"
            ) == applicant_id
        ),
        None
    )

    if expected_profile is None:

        print(
            f"Applicant '{applicant_id}' "
            "was not found."
        )

        raise SystemExit(1)

    # -----------------------------------------------------
    # Initial workflow state
    # -----------------------------------------------------

    initial_state = {

        "applicant_id":
            applicant_id,

        "expected_profile":
            expected_profile,

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
    }

    # -----------------------------------------------------
    # Run workflow
    # -----------------------------------------------------

    try:

        final_state = kyc_workflow.invoke(
            initial_state
        )

    except Exception as e:

        print(
            "\nWorkflow execution failed:"
        )

        print(str(e))

        raise SystemExit(1)

    # -----------------------------------------------------
    # Display final result
    # -----------------------------------------------------

    print(
        "\n========================================"
    )

    print(
        "FINAL KYC RESULT"
    )

    print(
        "========================================\n"
    )

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

    print(
        "\n========================================"
    )

    print(
        "WORKFLOW COMPLETED"
    )

    print(
        "========================================"
    )
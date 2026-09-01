from typing import TypedDict, Optional, Dict, Any, List


class KYCState(TypedDict, total=False):
    """
    Shared state used by the LangGraph KYC workflow.

    The state is passed from one agent to the next.

    Flow:

        Document Agent
              ↓
        Identity Agent
              ↓
        Sanctions Agent
              ↓
        Policy Agent
              ↓
        Decision Agent
    """

    # =====================================================
    # APPLICATION INPUT
    # =====================================================

    applicant_id: str

    # Expected applicant information from applicants.json
    expected_profile: Dict[str, Any]

    # =====================================================
    # DOCUMENT INPUTS
    # =====================================================

    # Optional custom document paths.
    #
    # Example:
    #
    # {
    #     "aadhar": "synthetic_documents/APP-001/aadhar.png",
    #     "pan": "synthetic_documents/APP-001/pan.png",
    #     "address_proof": "synthetic_documents/APP-001/address_proof.png"
    # }
    document_paths: Dict[str, str]

    # Applicant photograph paths.
    #
    # Example:
    #
    # {
    #     "id_photo": "synthetic_photos/APP-001/id_photo.png",
    #     "selfie": "synthetic_photos/APP-001/selfie.png"
    # }
    photo_paths: Dict[str, str]

    # =====================================================
    # DOCUMENT AGENT OUTPUT
    # =====================================================

    document_result: Dict[str, Any]

    # =====================================================
    # IDENTITY AGENT OUTPUT
    # =====================================================

    identity_result: Dict[str, Any]

    # =====================================================
    # SANCTIONS AGENT OUTPUT
    # =====================================================

    sanctions_result: Dict[str, Any]

    # =====================================================
    # POLICY AGENT OUTPUT
    # =====================================================

    policy_result: Dict[str, Any]

    # =====================================================
    # DECISION AGENT OUTPUT
    # =====================================================

    decision_result: Dict[str, Any]

    # =====================================================
    # WORKFLOW CONTROL
    # =====================================================

    # Determines which step the LangGraph workflow
    # should execute next.
    #
    # Examples:
    #
    # MORE_DOCUMENTS
    # IDENTITY
    # SANCTIONS
    # POLICY
    # DECISION
    # END
    next_action: str

    # =====================================================
    # WORKFLOW MESSAGES
    # =====================================================

    # Simple messages from agents for logging/demo output.
    messages: List[str]

    # Optional error information.
    error: Optional[str]
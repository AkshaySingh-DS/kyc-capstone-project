from typing import Any, Dict, List, Optional, TypedDict


class KYCState(TypedDict, total=False):
    """
    Shared state for the Phase 7 KYC multi-agent workflow.

    Each agent reads the information it needs from this state
    and adds its own results.

    The state intentionally contains no business logic.
    """

    # =====================================================
    # APPLICATION
    # =====================================================

    applicant_id: str

    # Path to the applicant's synthetic documents/photos.
    document_paths: Dict[str, str]
    photo_paths: Dict[str, str]

    # =====================================================
    # DOCUMENT VERIFICATION AGENT
    # =====================================================

    document_result: Dict[str, Any]

    # Examples:
    #
    # {
    #     "status": "COMPLETE",
    #     "documents_found": [...],
    #     "missing_documents": [...]
    # }

    # =====================================================
    # IDENTITY VERIFICATION AGENT
    # =====================================================

    identity_result: Dict[str, Any]

    # Examples:
    #
    # {
    #     "assessment": "PASS",
    #     "name_match": True,
    #     "dob_match": True
    # }

    # =====================================================
    # SANCTIONS SCREENING AGENT
    # =====================================================

    sanctions_result: Dict[str, Any]

    # Examples:
    #
    # {
    #     "status": "CLEAR",
    #     "candidates": []
    # }

    # =====================================================
    # POLICY / COMPLIANCE AGENT
    # =====================================================

    policy_result: Dict[str, Any]

    # Examples:
    #
    # {
    #     "answer": "...",
    #     "sources": [...]
    # }

    # =====================================================
    # MULTIMODAL VERIFICATION
    # =====================================================

    multimodal_result: Dict[str, Any]

    # Contains:
    #
    # - document visual analysis
    # - face verification
    #
    # Example:
    #
    # {
    #     "document_analysis": [...],
    #     "face_verification": {...}
    # }

    # =====================================================
    # FINAL DECISION AGENT
    # =====================================================

    decision: Optional[str]

    confidence: Optional[float]

    decision_reason: Optional[str]

    # Examples:
    #
    # decision:
    #     "APPROVE"
    #     "REVIEW"
    #     "MORE_DOCUMENTS"
    #     "ESCALATE"

    # =====================================================
    # WORKFLOW CONTROL
    # =====================================================

    next_action: Optional[str]

    # Used for conditional routing.
    #
    # Examples:
    #
    # "IDENTITY"
    # "SANCTIONS"
    # "POLICY"
    # "DECISION"
    # "MORE_DOCUMENTS"
    # "ESCALATE"

    # =====================================================
    # AGENT MESSAGES / AUDIT TRAIL
    # =====================================================

    messages: List[str]

    # Human-readable trace of what each agent did.
    #
    # Example:
    #
    # [
    #     "Document Agent: documents complete",
    #     "Identity Agent: identity verified",
    #     "Sanctions Agent: no sanctions candidates found"
    # ]

    # =====================================================
    # ERRORS
    # =====================================================

    errors: List[str]

    # Non-fatal errors encountered during processing.
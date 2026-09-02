from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials

# =========================================================
# Configuration
# =========================================================

MODEL_ID = (
    "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
)

WATSONX_URL = (
    "https://us-south.ml.cloud.ibm.com"
)

PROJECT_ID = "skills-network"


class DecisionAgent:
    """
    Final KYC Decision Agent.

    Responsibilities:

    1. Collect evidence from:
       - Document Agent
       - Identity Agent
       - Sanctions Agent
       - Policy Agent

    2. Apply deterministic KYC decision rules.

    3. Build a structured, audit-ready decision result.

    4. Use Llama 4 only to generate a concise explanation.

    5. Return the final KYC disposition.

    Possible decisions:

        APPROVE
        REVIEW
        REJECT
        MORE_DOCUMENTS

    Important:
        The LLM does NOT control the decision.
        Deterministic rules control the decision.
    """

    def __init__(self):

        credentials = Credentials(
            url=WATSONX_URL
        )

        self.model = ModelInference(
            model_id=MODEL_ID,
            credentials=credentials,
            project_id=PROJECT_ID,
            params={
                "temperature": 0,
                "max_tokens": 500
            }
        )

    # =====================================================
    # DETERMINE DECISION
    # =====================================================

    @staticmethod
    def determine_decision(
        document_result: dict,
        identity_result: dict,
        sanctions_result: dict,
    ) -> tuple:
        """
        Determine the final KYC disposition using
        deterministic rules.

        Returns:

            decision
            reasons
        """

        reasons = []

        # -------------------------------------------------
        # Rule 1 — Missing documents
        # -------------------------------------------------

        document_status = document_result.get(
            "status"
        )

        if document_status == "INCOMPLETE":

            missing_documents = document_result.get(
                "missing_documents",
                []
            )

            reasons.append(
                "Required KYC documents are missing: "
                + ", ".join(missing_documents)
            )

            return (
                "MORE_DOCUMENTS",
                reasons
            )

        # -------------------------------------------------
        # Rule 2 — Document analysis concerns
        # -------------------------------------------------

        visual_analysis = document_result.get(
            "visual_analysis",
            []
        )

        for analysis in visual_analysis:

            assessment = analysis.get(
                "assessment"
            )

            tampering_indicators = analysis.get(
                "tampering_indicators",
                []
            )

            if tampering_indicators:

                reasons.append(
                    "Possible document tampering or "
                    "authenticity concerns were detected."
                )

                return (
                    "REVIEW",
                    reasons
                )

            if assessment == "ANALYSIS_UNCERTAIN":

                reasons.append(
                    "Document visual analysis could not "
                    "be completed with sufficient confidence."
                )

                return (
                    "REVIEW",
                    reasons
                )

            if assessment == "REVIEW":

                reasons.append(
                    "Document verification requires "
                    "additional review."
                )

                return (
                    "REVIEW",
                    reasons
                )

        # -------------------------------------------------
        # Rule 3 — Identity verification
        # -------------------------------------------------

        identity_status = identity_result.get(
            "status"
        )

        identity_verification = (
            identity_result.get(
                "identity_verification",
                {}
            )
        )

        face_verification = (
            identity_result.get(
                "face_verification",
                {}
            )
        )

        identity_match_status = (
            identity_verification.get(
                "status"
            )
        )

        face_status = face_verification.get(
            "assessment"
        )

        if identity_match_status == "MISMATCH":

            reasons.append(
                "Customer identity information does "
                "not match the submitted documents."
            )

            return (
                "REVIEW",
                reasons
            )

        if identity_status == "REVIEW":

            reasons.append(
                "Identity verification requires "
                "manual review."
            )

            return (
                "REVIEW",
                reasons
            )

        if identity_status == "ANALYSIS_UNCERTAIN":

            reasons.append(
                "Identity verification could not "
                "be completed with sufficient confidence."
            )

            return (
                "REVIEW",
                reasons
            )

        # -------------------------------------------------
        # Rule 4 — Face verification
        # -------------------------------------------------

        if face_status == "REVIEW":

            reasons.append(
                "Face verification identified a "
                "possible mismatch between the ID "
                "photo and selfie."
            )

            return (
                "REVIEW",
                reasons
            )

        if face_status == "ANALYSIS_UNCERTAIN":

            reasons.append(
                "Face verification could not be "
                "completed with sufficient confidence."
            )

            return (
                "REVIEW",
                reasons
            )

        # -------------------------------------------------
        # Rule 5 — Sanctions screening
        # -------------------------------------------------

        sanctions_status = sanctions_result.get(
            "screening_status"
        )

        if sanctions_status == "POTENTIAL_MATCH":

            reasons.append(
                "Sanctions screening identified a "
                "potential OFAC match."
            )

            return (
                "REVIEW",
                reasons
            )

        if sanctions_status == "REVIEW":

            reasons.append(
                "Sanctions screening identified a "
                "candidate requiring further review."
            )

            return (
                "REVIEW",
                reasons
            )

        if sanctions_status == "ERROR":

            reasons.append(
                "Sanctions screening could not be "
                "completed successfully."
            )

            return (
                "REVIEW",
                reasons
            )

        # -------------------------------------------------
        # Rule 6 — Everything passed
        # -------------------------------------------------

        reasons.append(
            "Required documents are available, identity "
            "verification passed, face verification did "
            "not identify a concern, and sanctions "
            "screening is clear."
        )

        return (
            "APPROVE",
            reasons
        )

    # =====================================================
    # BUILD DOCUMENT ASSESSMENT
    # =====================================================

    @staticmethod
    def build_document_assessment(
        document_result: dict
    ) -> dict:
        """
        Convert Document Agent output into a concise,
        audit-friendly assessment.
        """

        status = document_result.get(
            "status",
            "UNKNOWN"
        )

        documents_found = document_result.get(
            "documents_found",
            []
        )

        missing_documents = document_result.get(
            "missing_documents",
            []
        )

        evidence = []

        if documents_found:

            evidence.append(
                "Documents found: "
                + ", ".join(documents_found)
            )

        if missing_documents:

            evidence.append(
                "Missing documents: "
                + ", ".join(missing_documents)
            )

        visual_analysis = document_result.get(
            "visual_analysis",
            []
        )

        for analysis in visual_analysis:

            tampering_indicators = analysis.get(
                "tampering_indicators",
                []
            )

            if tampering_indicators:

                evidence.extend(
                    tampering_indicators
                )

        return {
            "status": status,
            "documents_found": documents_found,
            "missing_documents": missing_documents,
            "evidence": evidence,
        }

    # =====================================================
    # BUILD IDENTITY ASSESSMENT
    # =====================================================

    @staticmethod
    def build_identity_assessment(
        identity_result: dict
    ) -> dict:
        """
        Convert Identity Agent output into a concise,
        audit-friendly assessment.
        """

        identity_status = identity_result.get(
            "status",
            "UNKNOWN"
        )

        identity_verification = (
            identity_result.get(
                "identity_verification",
                {}
            )
        )

        face_verification = (
            identity_result.get(
                "face_verification",
                {}
            )
        )

        evidence = []

        # -------------------------------------------------
        # Identity evidence
        # -------------------------------------------------

        name_similarity = identity_verification.get(
            "name_similarity"
        )

        dob_match = identity_verification.get(
            "dob_match"
        )

        address_similarity = identity_verification.get(
            "address_similarity"
        )

        if name_similarity is not None:

            evidence.append(
                f"Name similarity: {name_similarity}"
            )

        if dob_match is not None:

            evidence.append(
                f"DOB match: {dob_match}"
            )

        if address_similarity is not None:

            evidence.append(
                f"Address similarity: "
                f"{address_similarity}"
            )

        identity_match_status = (
            identity_verification.get(
                "status"
            )
        )

        if identity_match_status:

            evidence.append(
                f"Identity verification: "
                f"{identity_match_status}"
            )

        # -------------------------------------------------
        # Face evidence
        # -------------------------------------------------

        face_status = face_verification.get(
            "assessment"
        )

        if face_status:

            evidence.append(
                f"Face verification: "
                f"{face_status}"
            )

        face_similarity = face_verification.get(
            "face_similarity"
        )

        if face_similarity is not None:

            evidence.append(
                f"Face similarity: "
                f"{face_similarity}"
            )

        observations = face_verification.get(
            "observations",
            []
        )

        evidence.extend(observations)

        return {
            "status": identity_status,
            "identity_match": identity_match_status,
            "face_verification": face_status,
            "evidence": evidence,
        }

    # =====================================================
    # BUILD SANCTIONS ASSESSMENT
    # =====================================================

    @staticmethod
    def build_sanctions_assessment(
        sanctions_result: dict
    ) -> dict:
        """
        Convert Sanctions Agent output into a concise,
        audit-friendly assessment.
        """

        status = sanctions_result.get(
            "screening_status",
            "UNKNOWN"
        )

        candidates = sanctions_result.get(
            "candidates",
            []
        )

        evidence = []

        # -------------------------------------------------
        # Candidate evidence
        # -------------------------------------------------

        for candidate in candidates:

            name = candidate.get(
                "name"
            )

            name_similarity = candidate.get(
                "name_similarity"
            )

            dob_match = candidate.get(
                "dob_match"
            )

            country_match = candidate.get(
                "country_match"
            )

            candidate_status = candidate.get(
                "status"
            )

            if name:

                evidence.append(
                    f"Candidate: {name}"
                )

            if name_similarity is not None:

                evidence.append(
                    f"Name similarity: "
                    f"{name_similarity}"
                )

            if dob_match is not None:

                evidence.append(
                    f"DOB match: {dob_match}"
                )

            if country_match is not None:

                evidence.append(
                    f"Country match: "
                    f"{country_match}"
                )

            if candidate_status:

                evidence.append(
                    f"Candidate assessment: "
                    f"{candidate_status}"
                )

        # -------------------------------------------------
        # Clear screening
        # -------------------------------------------------

        if status == "CLEAR" and not evidence:

            evidence.append(
                "No sanctions candidate identified."
            )

        return {
            "status": status,
            "candidate_count": len(candidates),
            "evidence": evidence,
        }

    # =====================================================
    # BUILD POLICY ASSESSMENT
    # =====================================================

    @staticmethod
    def build_policy_assessment(
        policy_result: dict
    ) -> dict:
        """
        Convert Policy Agent output into a concise,
        audit-friendly assessment.
        """

        status = policy_result.get(
            "status",
            "UNKNOWN"
        )

        citations = []

        sources = policy_result.get(
            "sources",
            []
        )

        for source in sources:

            citations.append(
                {
                    "section": source.get(
                        "section"
                    ),
                    "title": source.get(
                        "title"
                    ),
                    "pages": source.get(
                        "pages"
                    ),
                }
            )

        answer = policy_result.get(
            "answer"
        )

        return {
            "status": status,
            "citations": citations,
            "policy_basis": answer,
        }

    # =====================================================
    # BUILD LLM PROMPT
    # =====================================================

    @staticmethod
    def build_prompt(
        applicant_id: str,
        decision: str,
        confidence: str,
        reasons: list,
        document_assessment: dict,
        identity_assessment: dict,
        sanctions_assessment: dict,
        policy_assessment: dict,
    ) -> str:
        """
        Build a concise prompt for Llama 4.

        The LLM only explains the deterministic result.
        It must not change the decision.
        """

        return f"""
You are a KYC Decision Assistant.

You are reviewing customer onboarding case:
{applicant_id}

IMPORTANT:
The final decision has already been determined by
deterministic KYC rules.

You MUST NOT change the decision.

You MUST NOT invent facts.

You MUST NOT invent RBI requirements.

You MUST NOT claim that a person is sanctioned unless
the supplied sanctions evidence explicitly says so.

FINAL DECISION:
{decision}

CONFIDENCE:
{confidence}

DETERMINISTIC REASONS:
{reasons}

DOCUMENT ASSESSMENT:
{document_assessment}

IDENTITY ASSESSMENT:
{identity_assessment}

SANCTIONS ASSESSMENT:
{sanctions_assessment}

POLICY ASSESSMENT:
{policy_assessment}

Instructions:

1. Explain exactly the supplied decision.
2. Use only the supplied evidence.
3. Mention the strongest evidence supporting the decision.
4. Mention sanctions evidence when relevant.
5. Mention RBI policy evidence when available.
6. Do not invent policy requirements.
7. Do not change the applicant ID.
8. Do not change the decision.
9. Keep the explanation concise.
10. End with a clear recommendation.

Format:

Decision:
{decision}

Confidence:
{confidence}

Reasoning:
<short explanation>

Policy Basis:
<short explanation using only supplied policy evidence>

Recommendation:
<next action>
"""

    # =====================================================
    # DETERMINE CONFIDENCE
    # =====================================================

    @staticmethod
    def determine_confidence(
        decision: str
    ) -> str:
        """
        Determine simple confidence level.

        For this college demo:

            APPROVE        → HIGH
            REJECT         → HIGH
            MORE_DOCUMENTS → HIGH
            REVIEW         → LOW

        REVIEW is LOW because additional human/specialist
        verification may be required.
        """

        if decision == "REVIEW":

            return "LOW"

        return "HIGH"

    # =====================================================
    # DETERMINE NEXT ACTION
    # =====================================================

    @staticmethod
    def determine_next_action(
        decision: str
    ) -> str:
        """
        Determine the next workflow action.
        """

        if decision == "APPROVE":

            return "END"

        if decision == "REJECT":

            return "END"

        if decision == "MORE_DOCUMENTS":

            return "MORE_DOCUMENTS"

        if decision == "REVIEW":

            return "MANUAL_REVIEW"

        return "END"

    # =====================================================
    # MAIN AGENT
    # =====================================================
    def run(
        self,
        applicant_id: str,
        document_result: dict,
        identity_result: dict,
        sanctions_result: dict,
        policy_result: dict,
    ) -> dict:
        """
        Execute the final KYC decision.

        Returns a structured, audit-ready decision result.
        """

        # -------------------------------------------------
        # 1. Determine decision
        # -------------------------------------------------

        decision, reasons = (
            self.determine_decision(
                document_result=document_result,
                identity_result=identity_result,
                sanctions_result=sanctions_result,
            )
        )

        # -------------------------------------------------
        # 2. Determine confidence
        # -------------------------------------------------

        confidence = self.determine_confidence(
            decision
        )

        # -------------------------------------------------
        # 3. Build structured assessments
        # -------------------------------------------------

        document_assessment = (
            self.build_document_assessment(
                document_result
            )
        )

        identity_assessment = (
            self.build_identity_assessment(
                identity_result
            )
        )

        sanctions_assessment = (
            self.build_sanctions_assessment(
                sanctions_result
            )
        )

        policy_assessment = (
            self.build_policy_assessment(
                policy_result
            )
        )

        # -------------------------------------------------
        # 4. Build LLM prompt
        # -------------------------------------------------

        prompt = self.build_prompt(
            applicant_id=applicant_id,
            decision=decision,
            confidence=confidence,
            reasons=reasons,
            document_assessment=document_assessment,
            identity_assessment=identity_assessment,
            sanctions_assessment=sanctions_assessment,
            policy_assessment=policy_assessment,
        )

        # -------------------------------------------------
        # 5. Ask Llama 4 for explanation
        # -------------------------------------------------

        try:

            response = self.model.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            explanation = (
                response[
                    "choices"
                ][0][
                    "message"
                ][
                    "content"
                ].strip()
            )

        except Exception as e:

            explanation = (
                "LLM explanation could not be generated: "
                f"{str(e)}"
            )

        # -------------------------------------------------
        # 6. Determine next action
        # -------------------------------------------------

        next_action = (
            self.determine_next_action(
                decision
            )
        )

        # -------------------------------------------------
        # 7. Build final audit-ready result
        # -------------------------------------------------

        result = {

            "applicant_id":
                applicant_id,

            "decision":
                decision,

            "confidence":
                confidence,

            "identity_assessment":
                identity_assessment,

            "sanctions_assessment":
                sanctions_assessment,

            "document_assessment":
                document_assessment,

            "policy_assessment":
                policy_assessment,

            "missing_documents":
                document_result.get(
                    "missing_documents",
                    []
                ),

            "reasons":
                reasons,

            "explanation":
                explanation,

            "policy_basis":
                policy_result.get(
                    "answer"
                ),

            "next_action":
                next_action,
        }

        return {

            "decision_result":
                result,

            "next_action":
                next_action,

            "message":
                (
                    f"Decision Agent: final KYC decision "
                    f"for {applicant_id} is {decision}."
                ),
        }


# =========================================================
# SIMPLE MANUAL TEST
# =========================================================

if __name__ == "__main__":

    import json

    print(
        "\n========================================"
    )

    print(
        "DECISION AGENT TEST"
    )

    print(
        "========================================\n"
    )

    # -----------------------------------------------------
    # Select applicant
    # -----------------------------------------------------

    applicant_id = input(
        "Enter applicant ID (e.g. APP-001): "
    ).strip().upper()

    if not applicant_id:

        print(
            "Applicant ID cannot be empty."
        )

        raise SystemExit(1)

    # -----------------------------------------------------
    # Load applicant
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

    applicant = next(
        (
            item
            for item in applicants
            if item.get("applicant_id")
            == applicant_id
        ),
        None
    )

    if applicant is None:

        print(
            f"Applicant '{applicant_id}' "
            "was not found."
        )

        raise SystemExit(1)

    scenario = applicant.get(
        "scenario",
        "perfect"
    )

    # -----------------------------------------------------
    # Representative test evidence
    #
    # Later LangGraph provides actual results.
    # -----------------------------------------------------

    document_result = {
        "status": "COMPLETE",
        "documents_found": [
            "aadhar",
            "pan",
            "address_proof"
        ],
        "missing_documents": [],
        "visual_analysis": [],
    }

    identity_result = {
        "status": "PASS",
        "identity_verification": {
            "status": "MATCH",
            "name_similarity": 1.0,
            "dob_match": True,
            "address_similarity": 1.0,
        },
        "face_verification": {
            "assessment": "PASS",
            "face_similarity": 1.0,
            "observations": [],
        },
    }

    sanctions_result = {
        "screening_status": "CLEAR",
        "candidates": [],
    }

    policy_result = {
        "status": "EVIDENCE_FOUND",
        "answer": (
            "RBI KYC requirements require customer "
            "identification and verification using "
            "reliable and independent sources."
        ),
        "sources": [
            {
                "section": "RBI KYC",
                "title": "RBI KYC Master Direction",
                "pages": 12,
            }
        ],
    }

    # -----------------------------------------------------
    # Simulate scenarios
    # -----------------------------------------------------

    if scenario == "missing_document":

        document_result["status"] = "INCOMPLETE"

        document_result[
            "missing_documents"
        ] = ["pan"]

    elif scenario == "document_tampering":

        document_result[
            "visual_analysis"
        ] = [
            {
                "assessment": "REVIEW",
                "tampering_indicators": [
                    "Possible document alteration"
                ],
            }
        ]

    elif scenario in (
        "photo_mismatch",
        "photo missmatch",
    ):

        identity_result["status"] = "REVIEW"

        identity_result[
            "face_verification"
        ] = {
            "assessment": "REVIEW",
            "face_similarity": 0.0,
            "observations": [
                "Facial features appear different."
            ]
        }

    elif scenario == "ofac_name_only":

        sanctions_result[
            "screening_status"
        ] = "REVIEW"

        sanctions_result[
            "candidates"
        ] = [
            {
                "name": "Example OFAC Candidate",
                "name_similarity": 0.95,
                "dob_match": False,
                "country_match": False,
                "status": "REVIEW",
            }
        ]

    elif scenario == "ofac_potential_match":

        sanctions_result[
            "screening_status"
        ] = "POTENTIAL_MATCH"

        sanctions_result[
            "candidates"
        ] = [
            {
                "name": "Example Sanctioned Person",
                "name_similarity": 0.98,
                "dob_match": True,
                "country_match": True,
                "status": "POTENTIAL_MATCH",
            }
        ]

    elif scenario == "dob_mismatch":

        identity_result["status"] = "REVIEW"

        identity_result[
            "identity_verification"
        ] = {
            "status": "MISMATCH",
            "name_similarity": 1.0,
            "dob_match": False,
            "address_similarity": 1.0,
        }

    # -----------------------------------------------------
    # Run Decision Agent
    # -----------------------------------------------------

    agent = DecisionAgent()

    result = agent.run(
        applicant_id=applicant_id,
        document_result=document_result,
        identity_result=identity_result,
        sanctions_result=sanctions_result,
        policy_result=policy_result,
    )

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print(
        "\n===== DECISION AGENT RESULT =====\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

    decision_result = result.get(
        "decision_result",
        {}
    )

    print(
        "\n===== FINAL DECISION =====\n"
    )

    print(
        decision_result.get(
            "decision"
        )
    )

    print(
        "\n===== CONFIDENCE =====\n"
    )

    print(
        decision_result.get(
            "confidence"
        )
    )

    print(
        "\n===== REASONS =====\n"
    )

    for reason in decision_result.get(
        "reasons",
        []
    ):

        print(
            f"- {reason}"
        )

    print(
        "\n===== LLM EXPLANATION =====\n"
    )

    print(
        decision_result.get(
            "explanation"
        )
    )

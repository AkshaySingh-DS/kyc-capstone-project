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

    2. Apply simple deterministic decision rules.

    3. Use Llama 4 to generate a concise explanation.

    4. Return the final KYC disposition.

    Possible decisions:

        APPROVE
        REVIEW
        REJECT
        MORE_DOCUMENTS

    The LLM explains the decision.
    The deterministic rules control the decision.
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
        simple deterministic rules.

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
    # BUILD LLM PROMPT
    # =====================================================

    @staticmethod
    def build_prompt(
        applicant_id: str,
        decision: str,
        reasons: list,
        document_result: dict,
        identity_result: dict,
        sanctions_result: dict,
        policy_result: dict,
    ) -> str:
        """
        Build a concise prompt for Llama 4.

        The LLM explains the decision using only
        the supplied evidence.
        """

        return f"""
You are a KYC Decision Assistant.

You are reviewing a customer onboarding case.

Your job is to explain the final KYC decision using
ONLY the evidence provided below.

Do NOT invent facts.

Do NOT change the proposed decision.

The deterministic KYC rules have already produced:

FINAL DECISION:
{decision}

REASONS:
{reasons}

APPLICANT ID:
{applicant_id}

DOCUMENT VERIFICATION:
{document_result}

IDENTITY VERIFICATION:
{identity_result}

SANCTIONS SCREENING:
{sanctions_result}

RBI POLICY EVIDENCE:
{policy_result}

Instructions:

1. Explain the decision clearly.
2. Mention the most important evidence.
3. Mention relevant RBI policy evidence when available.
4. Do not invent RBI requirements.
5. Do not claim that a person is sanctioned unless the
   evidence explicitly says so.
6. Keep the explanation concise.
7. End with a one-line recommendation.

Format:

Decision:
<decision>

Confidence:
<HIGH / MEDIUM / LOW>

Reasoning:
<short explanation>

Policy Basis:
<short explanation based only on supplied policy evidence>

Recommendation:
<next action>
"""

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

        Returns a result that can later be stored
        in LangGraph state.
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
        # 2. Build LLM prompt
        # -------------------------------------------------

        prompt = self.build_prompt(
            applicant_id=applicant_id,
            decision=decision,
            reasons=reasons,
            document_result=document_result,
            identity_result=identity_result,
            sanctions_result=sanctions_result,
            policy_result=policy_result,
        )

        # -------------------------------------------------
        # 3. Ask Llama 4 for explanation
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
        # 4. Determine confidence
        # -------------------------------------------------

        if decision in (
            "MORE_DOCUMENTS",
            "REJECT"
        ):

            confidence = "HIGH"

        elif decision == "REVIEW":

            confidence = "HIGH"

        else:

            confidence = "HIGH"

        # -------------------------------------------------
        # 5. Determine next action
        # -------------------------------------------------

        if decision == "APPROVE":

            next_action = "END"

        elif decision == "MORE_DOCUMENTS":

            next_action = "MORE_DOCUMENTS"

        elif decision == "REJECT":

            next_action = "END"

        else:

            next_action = "MANUAL_REVIEW"

        # -------------------------------------------------
        # 6. Build final result
        # -------------------------------------------------

        result = {
            "applicant_id": applicant_id,
            "decision": decision,
            "confidence": confidence,
            "reasons": reasons,
            "explanation": explanation,
            "policy_basis": policy_result.get(
                "answer"
            ),
            "next_action": next_action,
        }

        return {
            "decision_result": result,
            "next_action": next_action,
            "message": (
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
    # Later LangGraph will provide actual results.
    # -----------------------------------------------------

    document_result = {
        "status": "COMPLETE",
        "missing_documents": [],
        "visual_analysis": [],
    }

    identity_result = {
        "status": "PASS",
        "identity_verification": {
            "status": "MATCH"
        },
        "face_verification": {
            "assessment": "PASS"
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
        "sources": [],
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
            "observations": [
                "Facial features appear different."
            ]
        }

    elif scenario == "ofac_name_only":

        sanctions_result[
            "screening_status"
        ] = "REVIEW"

    elif scenario == "ofac_potential_match":

        sanctions_result[
            "screening_status"
        ] = "POTENTIAL_MATCH"

    elif scenario == "dob_mismatch":

        identity_result["status"] = "REVIEW"

        identity_result[
            "identity_verification"
        ] = {
            "status": "MISMATCH"
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
        "\n===== REASON =====\n"
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

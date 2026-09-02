from typing import Optional

from src.policy_rag.policy_answer import RBIPolicyAnswer

class PolicyAgent:
    """
    Policy / Compliance Agent.

    Responsibilities:

    1. Review evidence produced by previous agents.
    2. Identify the relevant KYC compliance question.
    3. Query the RBI Policy RAG system.
    4. Return RBI-grounded policy evidence.

    The Policy Agent does NOT make the final KYC decision.

    It provides policy interpretation to the
    downstream Decision Agent.
    """

    def __init__(self):

        # Reuse the existing RBI RAG + Watsonx implementation.
        self.policy_answer = RBIPolicyAnswer()

    # =====================================================
    # BUILD POLICY QUESTION
    # =====================================================

    @staticmethod
    def build_question(
        document_result: dict,
        identity_result: dict,
        sanctions_result: dict,
    ) -> str:
        """
        Build a simple policy question based on the
        evidence produced by previous agents.

        The question is intentionally simple because
        RBIPolicyAnswer already performs semantic retrieval.
        """

        # -------------------------------------------------
        # 1. Document issue
        # -------------------------------------------------

        document_status = document_result.get(
            "status"
        )

        if document_status == "INCOMPLETE":

            missing_documents = document_result.get(
                "missing_documents",
                []
            )

            return (
                "What does RBI KYC guidance require "
                "when required KYC documents are missing "
                "during customer onboarding? "
                f"The missing documents are: "
                f"{', '.join(missing_documents)}."
            )

        # -------------------------------------------------
        # 2. Document tampering / uncertainty
        # -------------------------------------------------

        visual_analysis = document_result.get(
            "visual_analysis",
            []
        )

        for analysis in visual_analysis:

            assessment = analysis.get(
                "assessment"
            )

            if assessment in (
                "REVIEW",
                "ANALYSIS_UNCERTAIN"
            ):

                return (
                    "What does RBI KYC guidance require "
                    "when submitted customer documents "
                    "cannot be confidently verified or "
                    "may require additional review?"
                )

            tampering_indicators = analysis.get(
                "tampering_indicators",
                []
            )

            if tampering_indicators:

                return (
                    "What does RBI KYC guidance require "
                    "when customer KYC documents show "
                    "possible tampering or authenticity concerns?"
                )

        # -------------------------------------------------
        # 3. Identity issue
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

        if identity_status in (
            "REVIEW",
            "ANALYSIS_UNCERTAIN"
        ):

            return (
                "What does RBI KYC guidance require "
                "when customer identity verification "
                "produces a mismatch or requires manual review?"
            )

        verification_status = (
            identity_verification.get(
                "status"
            )
        )

        if verification_status in (
            "REVIEW",
            "MISMATCH"
        ):

            return (
                "What does RBI KYC guidance require "
                "when customer identity information "
                "does not fully match the submitted KYC documents?"
            )

        # -------------------------------------------------
        # 4. Sanctions issue
        # -------------------------------------------------

        sanctions_status = sanctions_result.get(
            "screening_status"
        )

        if sanctions_status == "POTENTIAL_MATCH":

            return (
                "What does RBI KYC guidance require "
                "when sanctions screening identifies "
                "a potential match that requires further review?"
            )

        if sanctions_status == "REVIEW":

            return (
                "What does RBI KYC guidance require "
                "when sanctions screening produces "
                "an unresolved potential match during "
                "customer onboarding?"
            )

        # -------------------------------------------------
        # 5. No major issue
        # -------------------------------------------------

        return (
            "What are the RBI KYC requirements for "
            "customer identification and verification "
            "during customer onboarding?"
        )

    # =====================================================
    # MAIN AGENT
    # =====================================================
    
    def run(
        self,
        document_result: dict,
        identity_result: dict,
        sanctions_result: dict,
        applicant_id: Optional[str] = None,
    ) -> dict:
        """
        Run the Policy Agent.

        Parameters
        ----------
        document_result:
            Output from Document Verification Agent.

        identity_result:
            Output from Identity Verification Agent.

        sanctions_result:
            Output from Sanctions Agent.

        applicant_id:
            Optional applicant identifier.

        Returns
        -------
        dict
            Policy evidence for the Decision Agent.
        """

        # -------------------------------------------------
        # 1. Build policy question
        # -------------------------------------------------

        question = self.build_question(
            document_result=document_result,
            identity_result=identity_result,
            sanctions_result=sanctions_result,
        )

        # -------------------------------------------------
        # 2. Query RBI Policy RAG
        # -------------------------------------------------

        try:

            policy_result = (
                self.policy_answer.answer(
                    question=question,
                    top_k=5,
                )
            )

        except Exception as e:

            return {
                "policy_result": {
                    "applicant_id": applicant_id,
                    "question": question,
                    "answer": (
                        "Policy analysis failed."
                    ),
                    "sources": [],
                    "error": str(e),
                },
                "next_action": "DECISION",
                "message": (
                    "Policy Agent: RBI policy analysis "
                    "could not be completed."
                ),
            }

        # -------------------------------------------------
        # 3. Determine policy evidence status
        # -------------------------------------------------

        answer = policy_result.get(
            "answer",
            ""
        )

        sources = policy_result.get(
            "sources",
            []
        )

        if not sources:

            policy_status = (
                "INSUFFICIENT_EVIDENCE"
            )

        else:

            policy_status = "EVIDENCE_FOUND"

        # -------------------------------------------------
        # 4. Build result
        # -------------------------------------------------

        result = {
            "applicant_id": applicant_id,
            "status": policy_status,
            "question": question,
            "answer": answer,
            "sources": sources,
        }

        # -------------------------------------------------
        # Policy Agent provides evidence to the
        # Decision Agent.
        # -------------------------------------------------

        return {
            "policy_result": result,
            "next_action": "DECISION",
            "message": (
                f"Policy Agent: RBI policy analysis "
                f"completed"
                + (
                    f" for {applicant_id}."
                    if applicant_id
                    else "."
                )
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
        "POLICY AGENT TEST"
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
            f"\nApplicant file not found: "
            f"{applicants_file}"
        )

        raise SystemExit(1)

    # -----------------------------------------------------
    # Find applicant
    # -----------------------------------------------------

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
            f"\nApplicant '{applicant_id}' "
            "was not found."
        )

        raise SystemExit(1)

    # -----------------------------------------------------
    # For this manual test we create representative
    # evidence from the applicant scenario.
    #
    # Later LangGraph will provide the real results
    # from the previous agents.
    # -----------------------------------------------------

    scenario = applicant.get(
        "scenario",
        "perfect"
    )

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
    }

    sanctions_result = {
        "screening_status": "CLEAR",
        "candidates": [],
    }

    # -----------------------------------------------------
    # Simulate important scenarios
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

    elif scenario == "ofac_name_only":

        sanctions_result[
            "screening_status"
        ] = "REVIEW"

    elif scenario == "ofac_potential_match":

        sanctions_result[
            "screening_status"
        ] = "POTENTIAL_MATCH"

    # -----------------------------------------------------
    # Run Policy Agent
    # -----------------------------------------------------

    agent = PolicyAgent()

    result = agent.run(
        document_result=document_result,
        identity_result=identity_result,
        sanctions_result=sanctions_result,
        applicant_id=applicant_id,
    )

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print(
        "\n===== POLICY AGENT RESULT =====\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

    # -----------------------------------------------------
    # Human-readable output
    # -----------------------------------------------------

    policy_result = result.get(
        "policy_result",
        {}
    )

    print(
        "\n===== POLICY QUESTION =====\n"
    )

    print(
        policy_result.get(
            "question"
        )
    )

    print(
        "\n===== RBI POLICY ANSWER =====\n"
    )

    print(
        policy_result.get(
            "answer"
        )
    )

    print(
        "\n===== RBI SOURCES =====\n"
    )

    sources = policy_result.get(
        "sources",
        []
    )

    if not sources:

        print(
            "No RBI policy sources found."
        )

    else:

        for source in sources:

            print(
                f"Section: "
                f"{source.get('section')} | "
                f"Pages: "
                f"{source.get('pages')}"
            )


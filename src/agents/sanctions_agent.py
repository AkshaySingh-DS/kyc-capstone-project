from pathlib import Path
import json

from src.sanctions.sanctions_screening import (
    SanctionsScreener
)


class SanctionsScreeningAgent:
    """
    Phase 8 - Sanctions Screening Agent.

    Responsibilities:
    1. Screen applicant against the OFAC SDN list.
    2. Identify possible sanctions candidates.
    3. Provide detailed evidence to the Decision Agent.

    The agent does NOT make the final KYC decision.

    Existing sanctions logic is reused from:
        src/sanctions/sanctions_screening.py
    """

    # ---------------------------------------------------------
    # OFAC SDN file
    # ---------------------------------------------------------

    OFAC_FILE = (
        Path("data")
        / "sanctions"
        / "ofac_sdn.xml"
    )

    def __init__(self):

        if not self.OFAC_FILE.exists():

            raise FileNotFoundError(
                f"OFAC file not found: "
                f"{self.OFAC_FILE}"
            )

        self.screener = SanctionsScreener(
            str(self.OFAC_FILE)
        )

    # =========================================================
    # MAIN SCREENING
    # =========================================================

    def run(
        self,
        applicant_id: str,
        identity_result: dict
    ) -> dict:
        """
        Run sanctions screening.

        Parameters
        ----------
        applicant_id:
            Applicant identifier.

        identity_result:
            Result produced by the Identity Agent.

        Example:

            {
                "applicant_id": "APP-001",
                "actual_identity": {
                    "name": "Rahul Sharma",
                    "dob": "1995-03-12",
                    "address": "123 Example Street"
                }
            }

        Returns
        -------
        dict
            Sanctions Agent result.
        """

        # -----------------------------------------------------
        # 1. Get actual identity
        # -----------------------------------------------------

        actual_identity = identity_result.get(
            "actual_identity",
            {}
        )

        name = actual_identity.get(
            "name"
        )

        dob = actual_identity.get(
            "dob"
        )

        address = actual_identity.get(
            "address"
        )

        country = actual_identity.get(
            "country"
        )

        # -----------------------------------------------------
        # 2. Validate name
        # -----------------------------------------------------

        if not name:

            result = {
                "applicant_id": applicant_id,
                "screening_status": "ERROR",
                "candidates": [],
                "message": (
                    "Applicant name is missing."
                )
            }

            return {
                "sanctions_result": result,
                "next_action": "POLICY",
                "message": (
                    "Sanctions Agent: "
                    "screening could not be completed."
                )
            }

        # -----------------------------------------------------
        # 3. Prepare screening input
        # -----------------------------------------------------

        applicant = {
            "applicant_id": applicant_id,
            "name": name,
            "dob": dob,
            "address": address,
            "country": country
        }

        # -----------------------------------------------------
        # 4. Run existing sanctions screener
        # -----------------------------------------------------

        try:

            screening_result = self.screener.screen(
                applicant
            )

        except Exception as e:

            result = {
                "applicant_id": applicant_id,
                "screening_status": "ERROR",
                "candidates": [],
                "message": (
                    f"Sanctions screening failed: "
                    f"{str(e)}"
                )
            }

            return {
                "sanctions_result": result,
                "next_action": "POLICY",
                "message": (
                    "Sanctions Agent: "
                    "screening encountered an error."
                )
            }

        # -----------------------------------------------------
        # 5. Get screening status
        # -----------------------------------------------------

        screening_status = screening_result.get(
            "screening_status",
            "ERROR"
        )

        candidates = screening_result.get(
            "candidates",
            []
        )

        # -----------------------------------------------------
        # 6. Build simple sanctions result
        # -----------------------------------------------------

        result = {
            "applicant_id": applicant_id,
            "screening_status": screening_status,
            "candidates": candidates
        }

        # -----------------------------------------------------
        # 7. Agent message
        # -----------------------------------------------------

        if screening_status == "CLEAR":

            message = (
                f"Sanctions Agent: {applicant_id} "
                "cleared OFAC screening."
            )

        elif screening_status == "REVIEW":

            message = (
                f"Sanctions Agent: {applicant_id} "
                "has a possible OFAC match "
                "requiring review."
            )

        elif screening_status == "POTENTIAL_MATCH":

            message = (
                f"Sanctions Agent: {applicant_id} "
                "has a potential OFAC match."
            )

        else:

            message = (
                f"Sanctions Agent: screening failed "
                f"for {applicant_id}."
            )

        # -----------------------------------------------------
        # 8. Return result
        # -----------------------------------------------------

        return {
            "sanctions_result": result,
            "next_action": "POLICY",
            "message": message
        }

    # =========================================================
    # ADDITIONAL EVIDENCE
    # =========================================================

    def get_additional_evidence(
        self,
        sanctions_result: dict
    ) -> dict:
        """
        Provide detailed sanctions evidence.

        This method will be used later by the
        Decision Agent when sanctions screening
        results in REVIEW and confidence is low.

        It does not perform new screening.
        It simply exposes the evidence already
        generated by the Sanctions Screener.
        """

        candidates = sanctions_result.get(
            "candidates",
            []
        )

        evidence = []

        for candidate in candidates:

            evidence.append(
                {
                    "ofac_uid": candidate.get(
                        "ofac_uid"
                    ),

                    "ofac_name": candidate.get(
                        "ofac_name"
                    ),

                    "matched_name": candidate.get(
                        "matched_name"
                    ),

                    "name_similarity": candidate.get(
                        "name_similarity"
                    ),

                    "dob_match": candidate.get(
                        "dob_match"
                    ),

                    "country_match": candidate.get(
                        "country_match"
                    ),

                    "candidate_score": candidate.get(
                        "candidate_score"
                    ),

                    "assessment": candidate.get(
                        "assessment"
                    ),

                    "sdn_type": candidate.get(
                        "sdn_type"
                    ),

                    "programs": candidate.get(
                        "programs",
                        []
                    )
                }
            )

        return {
            "applicant_id": sanctions_result.get(
                "applicant_id"
            ),

            "screening_status": sanctions_result.get(
                "screening_status"
            ),

            "evidence": evidence
        }


# =========================================================
# MANUAL TEST
# =========================================================

if __name__ == "__main__":

    print(
        "\n========================================"
    )

    print(
        "PHASE 8 — SANCTIONS AGENT TEST"
    )

    print(
        "========================================\n"
    )

    # -----------------------------------------------------
    # Load applicants
    # -----------------------------------------------------

    applicants_file = (
        Path("synthetic_documents")
        / "applicants.json"
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
    # Create agent
    # -----------------------------------------------------

    try:

        agent = SanctionsScreeningAgent()

    except Exception as e:

        print(
            "\nFailed to create Sanctions Agent:"
        )

        print(str(e))

        raise SystemExit(1)

    # -----------------------------------------------------
    # Standalone test
    #
    # Normally these values will come from
    # Identity Agent.
    # -----------------------------------------------------

    identity_result = {

        "applicant_id": applicant_id,

        "actual_identity": {

            "applicant_id": applicant_id,

            "name": applicant.get(
                "name"
            ),

            "dob": applicant.get(
                "dob"
            ),

            "address": applicant.get(
                "address"
            ),

            "country": applicant.get(
                "country"
            )
        }
    }

    # -----------------------------------------------------
    # Run Sanctions Agent
    # -----------------------------------------------------

    result = agent.run(
        applicant_id=applicant_id,
        identity_result=identity_result
    )

    # -----------------------------------------------------
    # Display main result
    # -----------------------------------------------------

    print(
        "\n===== SANCTIONS AGENT RESULT =====\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

    # -----------------------------------------------------
    # Display additional evidence
    #
    # This simulates what the Decision Agent
    # will request later.
    # -----------------------------------------------------

    sanctions_result = result.get(
        "sanctions_result",
        {}
    )

    if sanctions_result.get(
        "screening_status"
    ) in (
        "REVIEW",
        "POTENTIAL_MATCH"
    ):

        print(
            "\n===== ADDITIONAL SANCTIONS EVIDENCE =====\n"
        )

        evidence = agent.get_additional_evidence(
            sanctions_result
        )

        print(
            json.dumps(
                evidence,
                indent=2,
                ensure_ascii=False
            )
        )

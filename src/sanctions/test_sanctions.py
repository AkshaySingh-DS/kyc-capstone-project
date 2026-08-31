import json
from pathlib import Path

from src.sanctions.sanctions_screening import (
    SanctionsScreener
)


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

APPLICANTS_FILE = (
    BASE_DIR
    / "synthetic_documents"
    / "applicants.json"
)

OFAC_FILE = (
    BASE_DIR
    / "data"
    / "sanctions"
    / "ofac_sdn.xml"
)


# ---------------------------------------------------------
# Load Applicants
# ---------------------------------------------------------

def load_applicants():
    """
    Load synthetic applicant master data.
    """

    with open(
        APPLICANTS_FILE,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


# ---------------------------------------------------------
# Build Screening Input
# ---------------------------------------------------------

def build_screening_input(applicant):
    """
    Convert applicant master data into the
    format expected by SanctionsScreener.
    """

    return {
        "applicant_id": applicant.get(
            "applicant_id"
        ),

        "name": applicant.get(
            "name"
        ),

        "dob": applicant.get(
            "dob"
        ),

        # Country is not currently available
        # in applicants.json.
        "country": applicant.get(
            "country"
        )
    }


# ---------------------------------------------------------
# Print Result
# ---------------------------------------------------------

def print_result(result):
    """
    Display a readable sanctions screening result.
    """

    print("\n===== SANCTIONS RESULT =====")

    print(
        f"Applicant ID: "
        f"{result['applicant_id']}"
    )

    print(
        f"Screening Status: "
        f"{result['screening_status']}"
    )

    candidates = result.get(
        "candidates",
        []
    )

    if not candidates:

        print("OFAC candidates: None")
        return

    print(
        f"Candidates found: "
        f"{len(candidates)}"
    )

    # -----------------------------------------------------
    # Display strongest candidates
    # -----------------------------------------------------

    for index, candidate in enumerate(
        candidates,
        start=1
    ):

        print(
            f"\n--- Candidate {index} ---"
        )

        print(
            f"OFAC UID: "
            f"{candidate.get('ofac_uid')}"
        )

        print(
            f"OFAC Name: "
            f"{candidate.get('ofac_name')}"
        )

        print(
            f"Matched Name: "
            f"{candidate.get('matched_name')}"
        )

        print(
            f"SDN Type: "
            f"{candidate.get('sdn_type')}"
        )

        print(
            f"Name Similarity: "
            f"{candidate.get('name_similarity')}"
        )

        print(
            f"DOB Match: "
            f"{candidate.get('dob_match')}"
        )

        print(
            f"Country Match: "
            f"{candidate.get('country_match')}"
        )

        print(
            f"Candidate Score: "
            f"{candidate.get('candidate_score')}"
        )

        print(
            f"Assessment: "
            f"{candidate.get('assessment')}"
        )

        print(
            f"Programs: "
            f"{candidate.get('programs')}"
        )


# ---------------------------------------------------------
# Main Test
# ---------------------------------------------------------

def main():

    print(
        "\n========================================"
    )

    print(
        "PHASE 5 — SANCTIONS SCREENING TEST"
    )

    print(
        "========================================\n"
    )

    # -----------------------------------------------------
    # Load applicants
    # -----------------------------------------------------

    applicants = load_applicants()

    print(
        f"Applicants loaded: "
        f"{len(applicants)}"
    )

    print(
        f"OFAC file: {OFAC_FILE}"
    )

    # -----------------------------------------------------
    # Create screener
    # -----------------------------------------------------

    screener = SanctionsScreener(
        str(OFAC_FILE)
    )

    # -----------------------------------------------------
    # Screen every applicant
    # -----------------------------------------------------

    for applicant in applicants:

        applicant_id = applicant.get(
            "applicant_id"
        )

        print(
            f"\n\nProcessing {applicant_id}..."
        )

        screening_input = (
            build_screening_input(
                applicant
            )
        )

        result = screener.screen(
            screening_input
        )

        print_result(result)

        print(
            "\n----------------------------------------"
        )


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":

    main()
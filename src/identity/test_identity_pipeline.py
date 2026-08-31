import json
from pathlib import Path

from src.documents_processing.document_processor import DocumentProcessor
from src.identity.identity_verifier import IdentityVerifier


BASE_DIR = Path(__file__).resolve().parents[2]

APPLICANTS_FILE = (
    BASE_DIR
    / "synthetic_documents"
    / "applicants.json"
)

DOCUMENTS_DIR = (
    BASE_DIR
    / "synthetic_documents"
)


def load_applicants():
    """Load expected applicant data."""

    with open(APPLICANTS_FILE, "r") as file:
        return json.load(file)


def process_applicant(applicant, processor):
    """
    Process all documents belonging to one applicant.
    """

    applicant_id = applicant["applicant_id"]

    applicant_dir = DOCUMENTS_DIR / applicant_id

    documents = [
        "aadhar.png",
        "pan.png",
        "address_proof.png",
    ]

    extracted_documents = []

    print(f"  Applicant directory: {applicant_dir}")

    for document_name in documents:

        image_path = applicant_dir / document_name

        print(f"  Checking: {image_path}")

        if not image_path.exists():
            print(f"  Missing: {document_name}")
            continue

        print(f"  Starting OCR: {document_name}")

        result = processor.process_document(
            str(image_path)
        )

        print(f"  Finished OCR: {document_name}")

        extracted_documents.append(result)

    print(
        f"  Finished applicant: {applicant_id}"
    )

    return extracted_documents


def build_actual_identity(extracted_documents):
    """
    Combine extracted fields from multiple documents
    into one identity record.
    """

    actual = {
        "name": None,
        "dob": None,
        "address": None,
    }

    for document in extracted_documents:

        fields = document["fields"]

        if not actual["name"] and fields.get("name"):
            actual["name"] = fields["name"]

        if not actual["dob"] and fields.get("dob"):
            actual["dob"] = fields["dob"]

        if not actual["address"] and fields.get("address"):
            actual["address"] = fields["address"]

    return actual


def main():

    applicants = load_applicants()

    # -------------------------------------------------
    # Initialize expensive components ONCE
    # -------------------------------------------------

    processor = DocumentProcessor()
    verifier = IdentityVerifier()

    print(
        "\n========================================"
    )
    print(
        "REAL PHASE 2 → PHASE 4 PIPELINE"
    )
    print(
        "========================================\n"
    )

    for applicant in applicants:

        applicant_id = applicant["applicant_id"]

        print(
            f"\nProcessing {applicant_id}..."
        )

        # -------------------------------------------------
        # Phase 2
        # -------------------------------------------------

        documents = process_applicant(
            applicant,
            processor
        )

        print(
            f"Documents processed: {len(documents)}"
        )

        # -------------------------------------------------
        # Combine extracted document fields
        # -------------------------------------------------

        actual = build_actual_identity(
            documents
        )

        # -------------------------------------------------
        # Expected customer identity
        # -------------------------------------------------

        expected = {
            "applicant_id": applicant_id,
            "name": applicant["name"],
            "dob": applicant["dob"],
            "address": applicant["address"],
        }

        # -------------------------------------------------
        # Phase 4
        # -------------------------------------------------

        result = verifier.verify(
            expected,
            actual
        )

        print("\nIdentity Result:")

        print(
            json.dumps(
                result,
                indent=2
            )
        )

        print(
            "\n----------------------------------------"
        )


if __name__ == "__main__":
    main()
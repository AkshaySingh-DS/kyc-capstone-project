from pathlib import Path
import json
import os

from paddleocr import PaddleOCR


class DocumentProcessor:
    """
    Simple KYC document processor.

    Image
        ↓
    Document Type Detection
        ↓
    OCR
        ↓
    Field Extraction
        ↓
    Field Validation
        ↓
    Normalized JSON
    """

    def __init__(self):
        # PaddleOCR configuration
        self.ocr = PaddleOCR(
            lang="en",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False
        )

    # =========================================================
    # MAIN PROCESSOR
    # =========================================================

    def process_document(self, image_path: str) -> dict:
        """
        Process one document image.

        Returns:
            Normalized JSON containing:
            - document name
            - raw OCR text
            - extracted fields
            - validation result
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Document not found: {image_path}"
            )

        # -----------------------------------------------------
        # 1. OCR
        # -----------------------------------------------------

        result = self.ocr.predict(str(image_path))

        text_lines = []

        for page in result:

            data = page.json

            if isinstance(data, str):
                data = json.loads(data)

            rec_texts = data.get(
                "res", {}
            ).get(
                "rec_texts", []
            )

            text_lines.extend(rec_texts)

        full_text = "\n".join(text_lines)

        # -----------------------------------------------------
        # 2. Extract fields
        # -----------------------------------------------------

        fields = self.extract_fields(full_text)

        # -----------------------------------------------------
        # 3. Validate extracted fields
        # -----------------------------------------------------

        validation = self.validate_fields(fields)

        # -----------------------------------------------------
        # 4. Return normalized result
        # -----------------------------------------------------

        return {
            "document": image_path.name,
            "raw_text": full_text,
            "fields": fields,
            "validation": validation
        }

    # =========================================================
    # FIELD EXTRACTION
    # =========================================================

    def extract_fields(self, text: str) -> dict:
        """
        Extract basic KYC fields from OCR text.

        This is intentionally rule-based.
        No LLM is used.
        """

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]

        result = {
            "document_type": self.detect_document_type(text),
            "name": None,
            "dob": None,
            "document_id": None,
            "applicant_id": None,
            "address": None
        }

        for i, line in enumerate(lines):

            lower = line.lower()

            # -------------------------------------------------
            # Applicant ID
            # -------------------------------------------------

            if lower.startswith("applicant id"):

                result["applicant_id"] = self.get_value(
                    lines,
                    i
                )

            # -------------------------------------------------
            # Date of Birth
            # -------------------------------------------------

            elif (
                lower.startswith("date of birth")
                or lower.startswith("dob")
            ):

                result["dob"] = self.get_value(
                    lines,
                    i
                )

            # -------------------------------------------------
            # Document ID
            # -------------------------------------------------

            elif (
                lower.startswith("document id")
                or lower.startswith(
                    "fictional document number"
                )
                or lower.startswith(
                    "reference number"
                )
            ):

                result["document_id"] = self.get_value(
                    lines,
                    i
                )

            # -------------------------------------------------
            # Name
            # -------------------------------------------------

            elif (
                lower.startswith("name")
                or lower.startswith(
                    "applicant name"
                )
                or lower.startswith(
                    "customer name"
                )
            ):

                result["name"] = self.get_value(
                    lines,
                    i
                )

            # -------------------------------------------------
            # Address
            # -------------------------------------------------

            elif lower.startswith("address"):

                result["address"] = self.get_address(
                    lines,
                    i
                )

        return result

    # =========================================================
    # GENERIC VALUE EXTRACTION
    # =========================================================

    @staticmethod
    def get_value(lines: list, index: int):
        """
        Extract a value from:

            Name: Rahul Sharma

        or:

            Name:
            Rahul Sharma
        """

        current_line = lines[index]

        # -----------------------------------------------------
        # Value exists on the same line
        # -----------------------------------------------------

        if ":" in current_line:

            value = current_line.split(
                ":",
                1
            )[1].strip()

            if value:
                return value

        # -----------------------------------------------------
        # Value exists on the next line
        # -----------------------------------------------------

        if index + 1 < len(lines):

            next_line = lines[index + 1].strip()

            if not DocumentProcessor.is_label(
                next_line
            ):
                return next_line

        return None

    # =========================================================
    # ADDRESS EXTRACTION
    # =========================================================

    @staticmethod
    def get_address(lines: list, index: int):
        """
        Extract address from synthetic documents.

        Handles:

            Address: 123 Example Street, Pune

        and OCR output such as:

            Address:
            123 Example Street,
            Applicant ID:
            APP-001
            Pune, 411001, India
        """

        current_line = lines[index]

        # -----------------------------------------------------
        # Normal case:
        #
        # Address: 123 Example Street, Pune
        # -----------------------------------------------------

        if ":" in current_line:

            value = current_line.split(
                ":",
                1
            )[1].strip()

            if value:
                return value

        # -----------------------------------------------------
        # OCR split-line case
        # -----------------------------------------------------

        address_parts = []

        i = index + 1

        while i < len(lines):

            current = lines[i].strip()

            lower = current.lower()

            # -------------------------------------------------
            # Applicant ID may appear inside OCR address block.
            #
            # Skip:
            #
            # Applicant ID:
            # APP-001
            # -------------------------------------------------

            if lower.startswith("applicant id"):

                i += 2
                continue

            # -------------------------------------------------
            # Stop at unrelated document fields
            # -------------------------------------------------

            if (
                lower.startswith("account number")
                or lower.startswith("previous balance")
                or lower.startswith("new charges")
                or lower.startswith("total amount")
                or lower.startswith("due date")
                or lower.startswith("this document")
            ):

                break

            # -------------------------------------------------
            # Stop at another known label
            # -------------------------------------------------

            if DocumentProcessor.is_label(
                current
            ):

                break

            address_parts.append(current)

            i += 1

        if address_parts:
            return " ".join(address_parts)

        return None

    # =========================================================
    # FIELD VALIDATION
    # =========================================================

    @staticmethod
    def validate_fields(fields: dict) -> dict:
        """
        Validate whether OCR successfully extracted
        the expected fields.

        IMPORTANT:
        This does NOT perform KYC decision-making.

        It only checks extraction completeness.
        """

        errors = []

        document_type = fields.get(
            "document_type"
        )

        # -----------------------------------------------------
        # Applicant ID
        # -----------------------------------------------------

        if not fields.get("applicant_id"):

            errors.append(
                "Missing applicant_id"
            )

        # -----------------------------------------------------
        # Document ID
        # -----------------------------------------------------

        if not fields.get("document_id"):

            errors.append(
                "Missing document_id"
            )

        # -----------------------------------------------------
        # Name
        # -----------------------------------------------------

        if not fields.get("name"):

            errors.append(
                "Missing name"
            )

        # -----------------------------------------------------
        # Document Type
        # -----------------------------------------------------

        if document_type == "UNKNOWN":

            errors.append(
                "Unknown document type"
            )

        # -----------------------------------------------------
        # DOB
        #
        # Required for identity documents
        # -----------------------------------------------------

        if document_type in [
            "PAN",
            "AADHAAR_LIKE"
        ]:

            if not fields.get("dob"):

                errors.append(
                    "Missing date of birth"
                )

        # -----------------------------------------------------
        # Address
        #
        # Required for:
        # - Aadhaar-like document
        # - Address proof
        # -----------------------------------------------------

        if document_type in [
            "AADHAAR_LIKE",
            "ADDRESS_PROOF"
        ]:

            if not fields.get("address"):

                errors.append(
                    "Missing address"
                )

        # -----------------------------------------------------
        # Final validation result
        # -----------------------------------------------------

        return {
            "valid": len(errors) == 0,
            "errors": errors
        }

    # =========================================================
    # DOCUMENT TYPE DETECTION
    # =========================================================

    @staticmethod
    def detect_document_type(text: str) -> str:
        """
        Detect synthetic document type using
        the synthetic document ID.
        """

        text_lower = text.lower()

        if "syn-pan" in text_lower:

            return "PAN"

        if "syn-aad" in text_lower:

            return "AADHAAR_LIKE"

        if "syn-addr" in text_lower:

            return "ADDRESS_PROOF"

        return "UNKNOWN"

    # =========================================================
    # FIELD LABEL DETECTION
    # =========================================================

    @staticmethod
    def is_label(line: str) -> bool:
        """
        Determine whether a line is a known
        document field label.
        """

        labels = {
            "name",
            "applicant name",
            "customer name",
            "date of birth",
            "dob",
            "document id",
            "fictional document number",
            "reference number",
            "applicant id",
            "address",
            "gender",
            "account number",
            "date of issue",
            "issuing authority",
            "customer details",
            "reference information",
            "statement date"
        }

        return line.lower().strip() in labels


# =============================================================
# TEST
# =============================================================

if __name__ == "__main__":

    processor = DocumentProcessor()

    print("Current directory:")
    print(os.getcwd())

    # ---------------------------------------------------------
    # Change this path when testing another document
    # ---------------------------------------------------------

    image_path = (
        "/home/labuser/Desktop/demo/"
        "1801398-my-repo/"
        "synthetic_documents/"
        "APP-001/"
        "aadhar.png"
    )

    # ---------------------------------------------------------
    # Process document
    # ---------------------------------------------------------

    result = processor.process_document(
        image_path
    )

    # ---------------------------------------------------------
    # Display result
    # ---------------------------------------------------------

    print(
        "\n===== OCR + EXTRACTED DATA =====\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

import re


class FieldExtractor:
    """Extract structured KYC fields from OCR text."""

    def extract(self, text: str) -> dict:

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
            "address": None,
        }

        for i, line in enumerate(lines):

            lower = line.lower()

            # Applicant ID
            if lower.startswith("applicant id"):
                result["applicant_id"] = self.get_value(
                    lines, i
                )

            # Date of Birth
            elif (
                lower.startswith("date of birth")
                or lower.startswith("dob")
            ):
                result["dob"] = self.get_value(
                    lines, i
                )

            # Document ID
            elif (
                lower.startswith("document id")
                or lower.startswith("fictional document number")
                or lower.startswith("reference number")
            ):
                result["document_id"] = self.get_value(
                    lines, i
                )

            # Name
            elif (
                lower.startswith("name")
                or lower.startswith("applicant name")
                or lower.startswith("customer name")
            ):
                result["name"] = self.get_value(
                    lines, i
                )

            # Address
            elif lower.startswith("address"):
                result["address"] = self.get_address(
                    lines, i
                )

        return result

    # =========================================================
    # VALUE EXTRACTION
    # =========================================================

    @staticmethod
    def get_value(lines: list, index: int):
        """Extract a field value from the same or next line."""

        current_line = lines[index]

        # Example:
        # Name: Rahul Sharma

        if ":" in current_line:

            value = current_line.split(
                ":", 1
            )[1].strip()

            if value:
                return value

        # Example:
        # Name:
        # Rahul Sharma

        if index + 1 < len(lines):

            next_line = lines[index + 1].strip()

            if not FieldExtractor.is_label(next_line):
                return next_line

        return None

    # =========================================================
    # ADDRESS EXTRACTION
    # =========================================================

    @staticmethod
    def get_address(lines: list, index: int):
        """
        Extract address from single-line or
        multi-line OCR output.
        """

        current_line = lines[index]

        # Example:
        # Address: 123 Example Street, Pune

        if ":" in current_line:

            value = current_line.split(
                ":", 1
            )[1].strip()

            if value:
                return value

        address_parts = []

        i = index + 1

        while i < len(lines):

            current = lines[i].strip()
            lower = current.lower()

            # Skip Applicant ID block
            if lower.startswith("applicant id"):

                i += 2
                continue

            # Stop at unrelated fields
            if (
                lower.startswith("account number")
                or lower.startswith("previous balance")
                or lower.startswith("new charges")
                or lower.startswith("total amount")
                or lower.startswith("due date")
                or lower.startswith("this document")
            ):
                break

            # Stop at another known label
            if FieldExtractor.is_label(current):
                break

            address_parts.append(current)

            i += 1

        if address_parts:
            return " ".join(address_parts)

        return None

    # =========================================================
    # DOCUMENT TYPE
    # =========================================================

    @staticmethod
    def detect_document_type(text: str) -> str:
        """Detect the synthetic document type."""

        text_lower = text.lower()

        if "syn-pan" in text_lower:
            return "PAN"

        if "syn-aad" in text_lower:
            return "AADHAAR_LIKE"

        if "syn-addr" in text_lower:
            return "ADDRESS_PROOF"

        return "UNKNOWN"

    # =========================================================
    # LABEL DETECTION
    # =========================================================

    @staticmethod
    def is_label(line: str) -> bool:

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
            "statement date",
        }

        return line.lower().strip() in labels
from pathlib import Path

from .ocr import OCRProcessor
from .field_extractor import FieldExtractor


class DocumentProcessor:
    """
    Coordinates the document processing pipeline.

    Image
      ↓
    OCR
      ↓
    Field Extraction
      ↓
    Field Validation
      ↓
    Structured Result
    """

    def __init__(self):

        self.ocr_processor = OCRProcessor()

        self.field_extractor = FieldExtractor()

    # =========================================================
    # MAIN PROCESSOR
    # =========================================================

    def process_document(self, image_path: str) -> dict:
        """
        Process one KYC document.

        Returns:
            {
                "document": "...",
                "raw_text": "...",
                "fields": {...},
                "validation": {...}
            }
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Document not found: {image_path}"
            )

        # -----------------------------------------------------
        # 1. OCR
        # -----------------------------------------------------

        raw_text = self.ocr_processor.extract_text(
            str(image_path)
        )

        # -----------------------------------------------------
        # 2. Field Extraction
        # -----------------------------------------------------

        fields = self.field_extractor.extract(
            raw_text
        )

        # -----------------------------------------------------
        # 3. Field Validation
        # -----------------------------------------------------

        validation = self.validate_fields(
            fields
        )

        # -----------------------------------------------------
        # 4. Return structured result
        # -----------------------------------------------------

        return {
            "document": image_path.name,
            "raw_text": raw_text,
            "fields": fields,
            "validation": validation,
        }

    # =========================================================
    # FIELD VALIDATION
    # =========================================================

    @staticmethod
    def validate_fields(fields: dict) -> dict:
        """
        Validate extraction completeness.

        This does NOT make a KYC decision.
        """

        errors = []

        document_type = fields.get(
            "document_type"
        )

        # Applicant ID
        if not fields.get("applicant_id"):
            errors.append(
                "Missing applicant_id"
            )

        # Document ID
        if not fields.get("document_id"):
            errors.append(
                "Missing document_id"
            )

        # Name
        if not fields.get("name"):
            errors.append(
                "Missing name"
            )

        # Document type
        if document_type == "UNKNOWN":
            errors.append(
                "Unknown document type"
            )

        # DOB
        if document_type in [
            "PAN",
            "AADHAAR_LIKE",
        ]:
            if not fields.get("dob"):
                errors.append(
                    "Missing date of birth"
                )

        # Address
        if document_type in [
            "AADHAAR_LIKE",
            "ADDRESS_PROOF",
        ]:
            if not fields.get("address"):
                errors.append(
                    "Missing address"
                )

        return {
            "valid": len(errors) == 0,
            "errors": errors,
        }
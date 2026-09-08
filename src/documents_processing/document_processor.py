from pathlib import Path
from .ocr import OCRProcessor
from .field_extractor import FieldExtractor
from src.utils.logger import logger

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

        try:

            image_path = Path(image_path)

            logger.info(
                f"Document Processing started | "
                f"file={image_path.name}"
            )    
            
            # -----------------------------------------------------
            # 1. OCR
            # -----------------------------------------------------

            logger.info(
                f"OCR has started | "
                f"file={image_path.name}"
            )

            raw_text = self.ocr_processor.extract_text(
                str(image_path)
            )

            logger.info(
                f"OCR has completed | "
                f"file={image_path.name} | "
                f"text_detected={True if raw_text else False}"
            )

            # -----------------------------------------------------
            # 2. Field Extraction
            # -----------------------------------------------------

            fields = self.field_extractor.extract(
                raw_text
            )

            logger.info(
                f"FIELD EXTRACTION completed | "
                f"Document_type= {fields.get('document_type')} | "
                f"fields_found={[field for field in fields.keys()]}"
            )

            # -----------------------------------------------------
            # 3. Field Validation
            # -----------------------------------------------------

            validation = self.validate_fields(
                fields
            )

            logger.info(
                f"FIELD validation completed | "
                f"fields_valid={validation.get('valid')} | "
            )

            # -----------------------------------------------------
            # 4. Return structured result
            # -----------------------------------------------------

            logger.info(
                f"Document Processing completed | "
            )   
            
            return {
                "document": image_path.name,
                "raw_text": raw_text,
                "fields": fields,
                "validation": validation,
            }
        
        except Exception as e:
            logger.exception(
                f"Document Not Found| "
                f"Document_type= {fields.get('document_type')} "
            )
            

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
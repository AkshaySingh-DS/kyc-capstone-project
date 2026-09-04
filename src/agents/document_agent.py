from pathlib import Path

from src.agents.state import KYCState
from src.documents_processing.document_processor import DocumentProcessor
from src.multimodal.document_visual_analyzer import DocumentVisualAnalyzer


class DocumentVerificationAgent:
    """
    Document Verification Agent.

    Responsibilities:
    1. Check that required KYC documents exist.
    2. Run existing document/OCR processing.
    3. Run multimodal visual analysis.
    4. Return the document verification result.

    The agent does not make a final KYC decision.

    NOTE:
    This agent currently works with the shared KYCState structure.
    """

    REQUIRED_DOCUMENTS = {
        "aadhar": "aadhar.png",
        "pan": "pan.png",
        "address_proof": "address_proof.png",
    }

    def __init__(self):
        self.document_processor = DocumentProcessor()
        self.visual_analyzer = DocumentVisualAnalyzer()

    def run(self, state: KYCState) -> dict:
        """
        Execute document verification for one applicant.

        Parameters
        ----------
        state:
            KYCState containing at least:
                {
                    "applicant_id": "...",
                    "document_paths": {...}
                }

        Returns
        -------
        dict
            Document verification result containing:
                - document_result
                - next_action
                - messages
        """

        applicant_id = state["applicant_id"]

        # -------------------------------------------------
        # 1. Locate applicant documents
        # -------------------------------------------------

        document_paths = state.get("document_paths", {})

        documents = {}

        for document_type, filename in self.REQUIRED_DOCUMENTS.items():

            path = document_paths.get(document_type)

            if path:
                documents[document_type] = path

            else:
                default_path = (
                    Path("synthetic_documents")
                    / applicant_id
                    / filename
                )

                documents[document_type] = str(default_path)

        # -------------------------------------------------
        # 2. Check required documents
        # -------------------------------------------------

        missing_documents = []

        for document_type, path in documents.items():

            if not Path(path).exists():
                missing_documents.append(document_type)

        # -------------------------------------------------
        # 2A. Handle missing documents
        # -------------------------------------------------

        if missing_documents:

            result = {
                "status": "INCOMPLETE",
                "documents_found": [
                    document_type
                    for document_type, path in documents.items()
                    if Path(path).exists()
                ],
                "missing_documents": missing_documents,
                "ocr_results": {},
                "visual_analysis": [],
            }

            return {
                "document_result": result,
                "next_action": "MORE_DOCUMENTS",
                "messages": [
                    (
                        f"Document Agent: missing documents "
                        f"for {applicant_id}: "
                        f"{', '.join(missing_documents)}"
                    )
                ],
            }

        # -------------------------------------------------
        # 3. Process documents using OCR/document processor
        # -------------------------------------------------

        ocr_results = {}

        for document_type, path in documents.items():

            try:

                ocr_results[document_type] = (
                    self.document_processor.process_document(path)
                )

            except Exception as e:

                ocr_results[document_type] = {
                    "error": str(e)
                }

        # -------------------------------------------------
        # 4. Visual analysis
        # -------------------------------------------------

        visual_analysis = []

        for document_type, path in documents.items():

            try:

                analysis = self.visual_analyzer.analyze(path)

                visual_analysis.append({
                    "document_type": document_type,
                    **analysis,
                })

            except Exception as e:

                visual_analysis.append({
                    "document_type": document_type,
                    "document": Path(path).name,
                    "assessment": "ANALYSIS_UNCERTAIN",
                    "observations": [],
                    "tampering_indicators": [],
                    "error": str(e),
                })

        # -------------------------------------------------
        # 5. Build final document result
        # -------------------------------------------------

        result = {
            "status": "COMPLETE",
            "documents_found": list(documents.keys()),
            "missing_documents": [],
            "ocr_results": ocr_results,
            "visual_analysis": visual_analysis,
        }

        # -------------------------------------------------
        # 6. Return agent output
        # -------------------------------------------------

        return {
            "document_result": result,
            "next_action": "IDENTITY",
            "messages": [
                (
                    f"Document Agent: all required documents "
                    f"available for {applicant_id}"
                )
            ],
        }


# =========================================================
# Simple Manual Test
# =========================================================

if __name__ == "__main__":

    print("\n===== DOCUMENT AGENT TEST =====\n")

    applicant_id = input(
        "Enter applicant ID (e.g. APP-001): "
    ).strip().upper()

    if not applicant_id:

        print("Applicant ID cannot be empty.")
        raise SystemExit(1)

    state = {
        "applicant_id": applicant_id,
        "document_paths": {},
        "photo_paths": {},
    }

    agent = DocumentVerificationAgent()

    try:

        result = agent.run(state)

    except Exception as e:

        print("\nDocument Agent execution failed:")
        print(str(e))
        raise SystemExit(1)

    print("\n===== DOCUMENT AGENT RESULT =====\n")

    print(result)
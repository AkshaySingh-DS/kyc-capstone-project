from pathlib import Path
from typing import Dict, Optional
import json

from src.identity.identity_verifier import IdentityVerifier
from src.multimodal.face_verifier import FaceVerifier


class IdentityVerificationAgent:
    """
    Phase 7 - Identity Verification Agent.

    Combines:

    1. Phase 3 deterministic identity verification
       - Name
       - Date of birth
       - Address

    2. Phase 6 multimodal face verification
       - ID photo
       - Selfie

    This agent provides evidence for the downstream
    Decision Agent. It does NOT make the final KYC decision.
    """

    def __init__(self):

        self.identity_verifier = IdentityVerifier()
        self.face_verifier = FaceVerifier()

    # =====================================================
    # BUILD ACTUAL IDENTITY DATA
    # =====================================================

    @staticmethod
    def build_actual_identity(
        ocr_results: dict
    ) -> dict:
        """
        Build a single identity record from OCR results.

        Priority:

        Name:
            Aadhaar -> PAN -> Address Proof

        DOB:
            Aadhaar -> PAN

        Address:
            Aadhaar -> Address Proof
        """

        aadhar = ocr_results.get(
            "aadhar",
            {}
        )

        pan = ocr_results.get(
            "pan",
            {}
        )

        address_proof = ocr_results.get(
            "address_proof",
            {}
        )

        aadhar_fields = aadhar.get(
            "fields",
            {}
        )

        pan_fields = pan.get(
            "fields",
            {}
        )

        address_fields = address_proof.get(
            "fields",
            {}
        )

        # -------------------------------------------------
        # Prefer Aadhaar where available.
        # -------------------------------------------------

        name = (
            aadhar_fields.get("name")
            or pan_fields.get("name")
            or address_fields.get("name")
        )

        dob = (
            aadhar_fields.get("dob")
            or pan_fields.get("dob")
        )

        address = (
            aadhar_fields.get("address")
            or address_fields.get("address")
        )

        applicant_id = (
            aadhar_fields.get("applicant_id")
            or pan_fields.get("applicant_id")
            or address_fields.get("applicant_id")
        )

        return {
            "applicant_id": applicant_id,
            "name": name,
            "dob": dob,
            "address": address,
        }

    # =====================================================
    # MAIN AGENT
    # =====================================================

    def run(
        self,
        applicant_id: str,
        expected_profile: dict,
        document_result: dict,
        photo_paths: Optional[Dict[str, str]] = None,
    ) -> dict:
        """
        Run identity verification.

        Parameters
        ----------
        applicant_id:
            Applicant identifier.

        expected_profile:
            Expected applicant identity information.

        document_result:
            Output from Document Verification Agent.

        photo_paths:
            Optional dictionary containing:

                {
                    "id_photo": "...",
                    "selfie": "..."
                }

        Returns
        -------
        dict
            Identity verification result.
        """

        # -------------------------------------------------
        # 1. Check document completeness
        # -------------------------------------------------

        if document_result.get("status") != "COMPLETE":

            return {
                "identity_result": {
                    "applicant_id": applicant_id,
                    "status": "INCOMPLETE",
                    "identity_verification": {},
                    "face_verification": {},
                },
                "next_action": "MORE_DOCUMENTS",
                "message": (
                    "Identity Agent: required documents "
                    "are incomplete."
                ),
            }

        # -------------------------------------------------
        # 2. Get OCR results
        # -------------------------------------------------

        ocr_results = document_result.get(
            "ocr_results",
            {}
        )

        if not ocr_results:

            return {
                "identity_result": {
                    "applicant_id": applicant_id,
                    "status": "ANALYSIS_UNCERTAIN",
                    "identity_verification": {},
                    "face_verification": {},
                },
                "next_action": "SANCTIONS",
                "message": (
                    "Identity Agent: OCR results unavailable."
                ),
            }

        # -------------------------------------------------
        # 3. Build actual identity from documents
        # -------------------------------------------------

        actual_identity = (
            self.build_actual_identity(
                ocr_results
            )
        )

        # -------------------------------------------------
        # 4. Make sure applicant ID is available
        # -------------------------------------------------

        if not expected_profile.get("applicant_id"):

            expected_profile = {
                **expected_profile,
                "applicant_id": applicant_id,
            }

        # -------------------------------------------------
        # 5. Run deterministic identity verification
        # -------------------------------------------------

        try:

            identity_verification = (
                self.identity_verifier.verify(
                    expected=expected_profile,
                    actual=actual_identity,
                )
            )

        except Exception as e:

            identity_verification = {
                "applicant_id": applicant_id,
                "status": "ANALYSIS_UNCERTAIN",
                "error": (
                    f"Identity verification failed: {str(e)}"
                ),
            }

        # -------------------------------------------------
        # 6. Face verification
        # -------------------------------------------------

        face_verification = {}

        if photo_paths:

            id_photo = photo_paths.get(
                "id_photo"
            )

            selfie = photo_paths.get(
                "selfie"
            )

            if id_photo and selfie:

                if (
                    Path(id_photo).is_file()
                    and Path(selfie).is_file()
                ):

                    try:

                        face_verification = (
                            self.face_verifier.verify(
                                id_photo_path=id_photo,
                                selfie_path=selfie,
                            )
                        )

                    except Exception as e:

                        face_verification = {
                            "assessment": (
                                "ANALYSIS_UNCERTAIN"
                            ),
                            "observations": [],
                            "error": (
                                f"Face verification failed: "
                                f"{str(e)}"
                            ),
                        }

                else:

                    face_verification = {
                        "assessment": (
                            "ANALYSIS_UNCERTAIN"
                        ),
                        "observations": [],
                        "error": (
                            "ID photo or selfie file "
                            "does not exist."
                        ),
                    }

        # -------------------------------------------------
        # 7. Determine identity agent status
        # -------------------------------------------------

        identity_status = identity_verification.get(
            "status",
            "ANALYSIS_UNCERTAIN"
        )

        face_status = face_verification.get(
            "assessment"
        )

        # -------------------------------------------------
        # Identity Agent does not make final KYC decision.
        #
        # It reports evidence and sends the workflow to
        # Sanctions Agent.
        # -------------------------------------------------

        if identity_status == "MATCH":

            overall_status = "PASS"

        elif identity_status in (
            "REVIEW",
            "MISMATCH",
        ):

            overall_status = "REVIEW"

        else:

            overall_status = "ANALYSIS_UNCERTAIN"

        # -------------------------------------------------
        # Face verification can influence the evidence
        # status, but final KYC disposition is handled by
        # the Decision Agent.
        # -------------------------------------------------

        if face_status == "REVIEW":

            overall_status = "REVIEW"

        elif face_status == "ANALYSIS_UNCERTAIN":

            if overall_status == "PASS":
                overall_status = "ANALYSIS_UNCERTAIN"

        # -------------------------------------------------
        # 8. Build final identity result
        # -------------------------------------------------

        result = {
            "applicant_id": applicant_id,
            "status": overall_status,
            "identity_verification": identity_verification,
            "face_verification": face_verification,
            "actual_identity": actual_identity,
        }

        # -------------------------------------------------
        # 9. Return agent output
        # -------------------------------------------------

        return {
            "identity_result": result,
            "next_action": "SANCTIONS",
            "message": (
                f"Identity Agent: verification completed "
                f"for {applicant_id}."
            ),
        }


# =========================================================
# Manual Test
# =========================================================

if __name__ == "__main__":

    print("\n===== IDENTITY AGENT TEST =====\n")

    agent = IdentityVerificationAgent()

    # -----------------------------------------------------
    # Select applicant dynamically
    # -----------------------------------------------------

    applicant_id = input(
        "Enter applicant ID (e.g. APP-001): "
    ).strip().upper()

    if not applicant_id:

        print("Applicant ID cannot be empty.")
        raise SystemExit(1)

    # -----------------------------------------------------
    # Load applicant profile
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
    # Find selected applicant
    # -----------------------------------------------------

    expected_profile = next(
        (
            applicant
            for applicant in applicants
            if applicant.get("applicant_id") == applicant_id
        ),
        None
    )

    if expected_profile is None:

        print(
            f"\nApplicant '{applicant_id}' "
            "was not found in applicants.json."
        )

        raise SystemExit(1)

    # -----------------------------------------------------
    # Run Document Agent
    #
    # IMPORTANT:
    #
    # DocumentVerificationAgent.run() expects a KYC state,
    # not applicant_id/document_directory/document_files
    # as separate keyword arguments.
    # -----------------------------------------------------

    try:

        from .document_agent import (
            DocumentVerificationAgent
        )

        document_agent = (
            DocumentVerificationAgent()
        )

        document_state = {
            "applicant_id": applicant_id,
            "document_paths": {},
            "photo_paths": {},
        }

        document_agent_output = (
            document_agent.run(document_state)
        )

    except Exception as e:

        print(
            "\nDocument Agent execution failed:"
        )

        print(str(e))

        raise SystemExit(1)

    # -----------------------------------------------------
    # Extract actual document_result
    #
    # Document Agent returns:
    #
    # {
    #     "document_result": {...},
    #     "next_action": "...",
    #     "messages": [...]
    # }
    # -----------------------------------------------------

    document_result = (
        document_agent_output.get(
            "document_result",
            {}
        )
    )

    # -----------------------------------------------------
    # Display Document Agent status
    # -----------------------------------------------------

    print("\n===== DOCUMENT AGENT STATUS =====\n")

    print(
        f"Status: "
        f"{document_result.get('status')}"
    )

    print(
        f"Next Action: "
        f"{document_agent_output.get('next_action')}"
    )

    # -----------------------------------------------------
    # Stop if documents are incomplete
    # -----------------------------------------------------

    if document_result.get("status") != "COMPLETE":

        print(
            "\nDocument verification is incomplete."
        )

        print(
            "Missing documents:"
        )

        for document in document_result.get(
            "missing_documents",
            []
        ):

            print(f"  - {document}")

        raise SystemExit(0)

    # -----------------------------------------------------
    # Build photo paths dynamically
    # -----------------------------------------------------

    photo_directory = (
        f"synthetic_photos/{applicant_id}"
    )

    photo_paths = {
        "id_photo": (
            f"{photo_directory}/id_photo.png"
        ),
        "selfie": (
            f"{photo_directory}/selfie.png"
        ),
    }

    # -----------------------------------------------------
    # Check photo files
    # -----------------------------------------------------

    missing_photos = [
        path
        for path in photo_paths.values()
        if not Path(path).exists()
    ]

    if missing_photos:

        print("\nMissing photo files:")

        for path in missing_photos:

            print(f"  - {path}")

        raise SystemExit(1)

    # -----------------------------------------------------
    # Run Identity Agent
    # -----------------------------------------------------

    try:

        result = agent.run(
            applicant_id=applicant_id,
            expected_profile=expected_profile,
            document_result=document_result,
            photo_paths=photo_paths,
        )

    except Exception as e:

        print(
            "\nIdentity Agent execution failed:"
        )

        print(str(e))

        raise SystemExit(1)

    # -----------------------------------------------------
    # Display result
    # -----------------------------------------------------

    print(
        "\n===== IDENTITY AGENT RESULT =====\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )
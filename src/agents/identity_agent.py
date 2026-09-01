from pathlib import Path
from typing import Dict, Optional

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
                "status": "INCOMPLETE",
                "identity_verification": {},
                "face_verification": {},
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
                "status": "ANALYSIS_UNCERTAIN",
                "identity_verification": {},
                "face_verification": {},
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

        # Make sure applicant ID is available.
        if not expected_profile.get("applicant_id"):

            expected_profile = {
                **expected_profile,
                "applicant_id": applicant_id,
            }

        # -------------------------------------------------
        # 4. Run deterministic identity verification
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
        # 5. Face verification
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
        # 6. Determine identity agent status
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

        # A face mismatch/review should be visible in the
        # identity result, but final disposition is handled
        # later by Decision Agent.

        if face_status == "REVIEW":

            overall_status = "REVIEW"

        elif face_status == "ANALYSIS_UNCERTAIN":

            if overall_status == "PASS":
                overall_status = "ANALYSIS_UNCERTAIN"

        # -------------------------------------------------
        # 7. Build final agent result
        # -------------------------------------------------

        result = {
            "applicant_id": applicant_id,
            "status": overall_status,
            "identity_verification": identity_verification,
            "face_verification": face_verification,
            "actual_identity": actual_identity,
        }

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

    applicant_id = "APP-001"

    # Expected identity profile.
    expected_profile = {
        "applicant_id": "APP-001",
        "name": "Rahul Sharma",
        "dob": "12-03-1995",
        "address": (
            "123 Example Street, Pune, 411001"
        ),
    }

    # Minimal document-agent-style input.
    #
    # In the actual LangGraph workflow this will come
    # directly from Document Verification Agent.
    document_result = {
        "status": "COMPLETE",
        "ocr_results": {
            "aadhar": {
                "fields": {
                    "name": "RAHUL SHARMA",
                    "dob": "12-03-1995",
                    "address": (
                        "123 Example Street, Pune, 411001"
                    ),
                    "applicant_id": "APP-001",
                }
            },
            "pan": {
                "fields": {
                    "name": "Rahul Sharma",
                    "dob": "12-03-1995",
                    "applicant_id": "APP-001",
                }
            },
            "address_proof": {
                "fields": {
                    "name": "Rahul Sharma",
                    "address": (
                        "123 Example Street, Pune, 411001, India"
                    ),
                    "applicant_id": "APP-001",
                }
            },
        },
    }

    photo_paths = {
        "id_photo": (
            "synthetic_photos/"
            "APP-021/"
            "id_photo.png"
        ),
        "selfie": (
            "synthetic_photos/"
            "APP-021/"
            "selfie.png"
        ),
    }

    result = agent.run(
        applicant_id=applicant_id,
        expected_profile=expected_profile,
        document_result=document_result,
        photo_paths=photo_paths,
    )

    print("\n===== IDENTITY AGENT RESULT =====\n")

    import json

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

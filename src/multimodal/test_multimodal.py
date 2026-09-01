import json
from pathlib import Path

from .document_visual_analyzer import DocumentVisualAnalyzer
from .face_verifier import FaceVerifier


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

DOCUMENTS_DIR = Path("synthetic_documents")
PHOTOS_DIR = Path("synthetic_photos")


# ---------------------------------------------------------
# Main Test
# ---------------------------------------------------------

def main():

    print("\n========================================")
    print("PHASE 6 MULTIMODAL VERIFICATION TEST")
    print("========================================\n")

    # -----------------------------------------------------
    # Validate directories
    # -----------------------------------------------------

    if not DOCUMENTS_DIR.exists():

        print(
            f"Documents directory not found: "
            f"{DOCUMENTS_DIR}"
        )

        return

    if not PHOTOS_DIR.exists():

        print(
            f"Photos directory not found: "
            f"{PHOTOS_DIR}"
        )

        return

    # -----------------------------------------------------
    # Initialize components
    # -----------------------------------------------------

    print("Initializing vision models...")

    document_analyzer = DocumentVisualAnalyzer()
    face_verifier = FaceVerifier()

    print("Vision models ready.\n")

    # -----------------------------------------------------
    # Find applicants
    # -----------------------------------------------------

    applicants = sorted(
        [
            directory
            for directory in PHOTOS_DIR.iterdir()
            if directory.is_dir()
            and directory.name.startswith("APP-")
        ]
    )

    if not applicants:

        print("No applicant directories found.")

        return

    print(
        f"Applicants found: {len(applicants)}"
    )

    results = []

    # =====================================================
    # PROCESS APPLICANTS
    # =====================================================

    for applicant_dir in applicants:

        applicant_id = applicant_dir.name

        print("\n========================================")
        print(f"Processing {applicant_id}")
        print("========================================")

        applicant_result = {
            "applicant_id": applicant_id,
            "document_analysis": [],
            "face_verification": None
        }

        # =================================================
        # DOCUMENT VISUAL ANALYSIS
        # =================================================

        print("\n--- Document Visual Analysis ---")

        document_dir = (
            DOCUMENTS_DIR /
            applicant_id
        )

        document_files = [
            "aadhar.png",
            "pan.png",
            "address_proof.png"
        ]

        if not document_dir.exists():

            print(
                "Document directory not found."
            )

        else:

            for filename in document_files:

                document_path = (
                    document_dir /
                    filename
                )

                if not document_path.exists():

                    print(
                        f"Skipping {filename}: "
                        "file not found."
                    )

                    continue

                print(
                    f"\nAnalyzing: "
                    f"{document_path.name}"
                )

                try:

                    result = (
                        document_analyzer.analyze(
                            str(document_path)
                        )
                    )

                    applicant_result[
                        "document_analysis"
                    ].append(result)

                    print(
                        f"Assessment: "
                        f"{result.get('assessment')}"
                    )

                    indicators = result.get(
                        "tampering_indicators",
                        []
                    )

                    if indicators:

                        print(
                            "Tampering indicators:"
                        )

                        for indicator in indicators:

                            print(
                                f"  - {indicator}"
                            )

                except Exception as e:

                    print(
                        f"Document analysis failed: "
                        f"{e}"
                    )

        # =================================================
        # FACE VERIFICATION
        # =================================================

        print("\n--- Face Verification ---")

        id_photo = (
            applicant_dir /
            "id_photo.png"
        )

        selfie = (
            applicant_dir /
            "selfie.png"
        )

        if not id_photo.exists():

            print(
                "Skipping: id_photo.png not found."
            )

        elif not selfie.exists():

            print(
                "Skipping: selfie.png not found."
            )

        else:

            print(
                f"ID Photo: {id_photo.name}"
            )

            print(
                f"Selfie: {selfie.name}"
            )

            try:

                face_result = (
                    face_verifier.verify(
                        str(id_photo),
                        str(selfie)
                    )
                )

                applicant_result[
                    "face_verification"
                ] = face_result

                print(
                    f"Similarity: "
                    f"{face_result.get('face_similarity', 'N/A')}"
                )

                print(
                    f"Assessment: "
                    f"{face_result.get('assessment')}"
                )

            except Exception as e:

                print(
                    f"Face verification failed: "
                    f"{e}"
                )

        # -------------------------------------------------
        # Store result
        # -------------------------------------------------

        results.append(
            applicant_result
        )

    # =====================================================
    # FINAL SUMMARY
    # =====================================================

    print("\n\n========================================")
    print("PHASE 6 MULTIMODAL SUMMARY")
    print("========================================\n")

    document_clear = 0
    document_review = 0
    document_uncertain = 0

    face_consistent = 0
    face_review = 0
    face_uncertain = 0

    for result in results:

        applicant_id = result[
            "applicant_id"
        ]

        # -------------------------------------------------
        # Document assessment
        # -------------------------------------------------

        documents = result[
            "document_analysis"
        ]

        doc_assessments = []

        for document in documents:

            assessment = document.get(
                "assessment"
            )

            if assessment:

                doc_assessments.append(
                    assessment
                )

        if doc_assessments:

            if all(
                assessment ==
                "NO_OBVIOUS_TAMPERING"
                for assessment in doc_assessments
            ):

                document_clear += 1

            elif any(
                assessment ==
                "POTENTIAL_TAMPERING_INDICATORS"
                for assessment in doc_assessments
            ):

                document_review += 1

            else:

                document_uncertain += 1

        # -------------------------------------------------
        # Face assessment
        # -------------------------------------------------

        face_result = result[
            "face_verification"
        ]

        face_assessment = "Not analyzed"

        if face_result:

            face_assessment = face_result.get(
                "assessment",
                "ANALYSIS_UNCERTAIN"
            )

            if face_assessment == "CONSISTENT":

                face_consistent += 1

            elif face_assessment == "REVIEW":

                face_review += 1

            else:

                face_uncertain += 1

        # -------------------------------------------------
        # Applicant summary
        # -------------------------------------------------

        if doc_assessments:

            document_summary = ", ".join(
                doc_assessments
            )

        else:

            document_summary = "Not analyzed"

        print(
            f"{applicant_id}"
        )

        print(
            f"  Documents: "
            f"{document_summary}"
        )

        print(
            f"  Face: "
            f"{face_assessment}"
        )

        print()

    # =====================================================
    # SUMMARY COUNTS
    # =====================================================

    print("----------------------------------------")

    print(
        f"Applicants processed : "
        f"{len(results)}"
    )

    print(
        f"Documents clear      : "
        f"{document_clear}"
    )

    print(
        f"Documents review     : "
        f"{document_review}"
    )

    print(
        f"Documents uncertain  : "
        f"{document_uncertain}"
    )

    print(
        f"Faces consistent     : "
        f"{face_consistent}"
    )

    print(
        f"Faces review         : "
        f"{face_review}"
    )

    print(
        f"Faces uncertain      : "
        f"{face_uncertain}"
    )

    print("----------------------------------------")

    # =====================================================
    # COMPLETE JSON
    # =====================================================

    print(
        "\n===== COMPLETE MULTIMODAL RESULTS =====\n"
    )

    print(
        json.dumps(
            results,
            indent=2,
            ensure_ascii=False
        )
    )


# ---------------------------------------------------------
# Entry Point
# ---------------------------------------------------------

if __name__ == "__main__":

    main()

from rapidfuzz import fuzz

from .normalizer import (
    normalize_name,
    normalize_address,
    normalize_dob,
)


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

NAME_WEIGHT = 0.40
DOB_WEIGHT = 0.30
ADDRESS_WEIGHT = 0.30

MATCH_THRESHOLD = 0.90
REVIEW_THRESHOLD = 0.75

# Minimum individual field thresholds
NAME_MATCH_THRESHOLD = 0.85
ADDRESS_MATCH_THRESHOLD = 0.75


# ---------------------------------------------------------
# Identity Verifier
# ---------------------------------------------------------

class IdentityVerifier:
    """
    Deterministic KYC identity verification.

    Python is used for:
    - Name normalization
    - DOB normalization
    - Address normalization
    - Fuzzy matching
    - Confidence calculation
    - Final decision

    No LLM is used here.
    """

    # =========================================================
    # NAME COMPARISON
    # =========================================================

    def compare_name(
        self,
        expected: str,
        actual: str
    ) -> float:
        """
        Compare two names using fuzzy matching.

        Returns:
            Similarity score between 0 and 1.
        """

        expected = normalize_name(expected)
        actual = normalize_name(actual)

        if not expected or not actual:
            return 0.0

        score = fuzz.token_sort_ratio(
            expected,
            actual
        )

        return round(score / 100, 2)

    # =========================================================
    # ADDRESS COMPARISON
    # =========================================================

    def compare_address(
        self,
        expected: str,
        actual: str
    ) -> float:
        """
        Compare two addresses using fuzzy matching.

        Returns:
            Similarity score between 0 and 1.
        """

        expected = normalize_address(expected)
        actual = normalize_address(actual)

        if not expected or not actual:
            return 0.0

        score = fuzz.token_set_ratio(
            expected,
            actual
        )

        return round(score / 100, 2)

    # =========================================================
    # DOB COMPARISON
    # =========================================================

    def compare_dob(
        self,
        expected: str,
        actual: str
    ) -> bool:
        """
        Compare two dates of birth after normalization.
        """

        expected = normalize_dob(expected)
        actual = normalize_dob(actual)

        if not expected or not actual:
            return False

        return expected == actual

    # =========================================================
    # MAIN VERIFICATION
    # =========================================================

    def verify(
        self,
        expected: dict,
        actual: dict
    ) -> dict:
        """
        Perform deterministic identity verification.

        expected:
            Expected applicant profile.

        actual:
            Fields extracted from submitted documents.
        """

        # -----------------------------------------------------
        # Compare individual fields
        # -----------------------------------------------------

        name_similarity = self.compare_name(
            expected.get("name"),
            actual.get("name")
        )

        dob_match = self.compare_dob(
            expected.get("dob"),
            actual.get("dob")
        )

        address_similarity = self.compare_address(
            expected.get("address"),
            actual.get("address")
        )

        # -----------------------------------------------------
        # Convert DOB result to numeric score
        # -----------------------------------------------------

        dob_score = 1.0 if dob_match else 0.0

        # -----------------------------------------------------
        # Calculate overall confidence
        # -----------------------------------------------------

        identity_confidence = (
            name_similarity * NAME_WEIGHT
            + dob_score * DOB_WEIGHT
            + address_similarity * ADDRESS_WEIGHT
        )

        identity_confidence = round(
            identity_confidence,
            2
        )

        # -----------------------------------------------------
        # Determine final status
        # -----------------------------------------------------

        status = self._determine_status(
            name_similarity=name_similarity,
            dob_match=dob_match,
            address_similarity=address_similarity,
            identity_confidence=identity_confidence,
        )

        # -----------------------------------------------------
        # Return verification result
        # -----------------------------------------------------

        return {
            "applicant_id": expected.get("applicant_id"),
            "name_similarity": name_similarity,
            "dob_match": dob_match,
            "address_similarity": address_similarity,
            "identity_confidence": identity_confidence,
            "status": status,
        }

    # =========================================================
    # DECISION LOGIC
    # =========================================================

    @staticmethod
    def _determine_status(
        name_similarity: float,
        dob_match: bool,
        address_similarity: float,
        identity_confidence: float,
    ) -> str:
        """
        Determine final identity verification status.

        MATCH:
            Strong agreement across identity attributes.

        REVIEW:
            Some discrepancy exists and should be
            manually reviewed.

        MISMATCH:
            Critical identity information conflicts.
        """

        # -----------------------------------------------------
        # DOB mismatch is treated as a critical conflict.
        # -----------------------------------------------------

        if not dob_match:
            return "MISMATCH"

        # -----------------------------------------------------
        # Significant name mismatch is treated as a
        # critical identity conflict.
        # -----------------------------------------------------

        if name_similarity < NAME_MATCH_THRESHOLD:
            return "MISMATCH"

        # -----------------------------------------------------
        # Address discrepancy should not automatically
        # reject the applicant because addresses can change
        # or have formatting/OCR differences.
        #
        # Send it for manual review.
        # -----------------------------------------------------

        if address_similarity < ADDRESS_MATCH_THRESHOLD:
            return "REVIEW"

        # -----------------------------------------------------
        # Strong match
        # -----------------------------------------------------

        if identity_confidence >= MATCH_THRESHOLD:
            return "MATCH"

        # -----------------------------------------------------
        # Moderate confidence
        # -----------------------------------------------------

        if identity_confidence >= REVIEW_THRESHOLD:
            return "REVIEW"

        # -----------------------------------------------------
        # Low confidence
        # -----------------------------------------------------

        return "MISMATCH"


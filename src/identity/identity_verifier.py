from rapidfuzz import fuzz

from normalizer import (
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


# ---------------------------------------------------------
# Identity Verifier
# ---------------------------------------------------------

class IdentityVerifier:

    def compare_name(self, expected: str, actual: str) -> float:
        """
        Compare two names using fuzzy matching.

        Returns a similarity score between 0 and 1.
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

    def compare_address(
        self,
        expected: str,
        actual: str
    ) -> float:
        """
        Compare two addresses using fuzzy matching.

        Returns a similarity score between 0 and 1.
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

    def verify(
        self,
        expected: dict,
        actual: dict
    ) -> dict:
        """
        Perform deterministic identity verification.

        expected:
            Expected applicant profile from applicants.json.

        actual:
            Fields extracted from the submitted document.
        """

        # -------------------------------------------------
        # Compare individual fields
        # -------------------------------------------------

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

        # Convert DOB result to numeric score
        dob_score = 1.0 if dob_match else 0.0

        # -------------------------------------------------
        # Calculate overall confidence
        # -------------------------------------------------

        identity_confidence = (
            name_similarity * NAME_WEIGHT
            + dob_score * DOB_WEIGHT
            + address_similarity * ADDRESS_WEIGHT
        )

        identity_confidence = round(
            identity_confidence,
            2
        )

        # -------------------------------------------------
        # Determine final status
        # -------------------------------------------------

        if (
            identity_confidence >= MATCH_THRESHOLD
            and dob_match
            and name_similarity >= 0.85
        ):
            status = "MATCH"

        elif identity_confidence >= REVIEW_THRESHOLD:
            status = "REVIEW"

        else:
            status = "MISMATCH"

        # -------------------------------------------------
        # Return verification result
        # -------------------------------------------------

        return {
            "applicant_id": expected.get("applicant_id"),
            "name_similarity": name_similarity,
            "dob_match": dob_match,
            "address_similarity": address_similarity,
            "identity_confidence": identity_confidence,
            "status": status,
        }
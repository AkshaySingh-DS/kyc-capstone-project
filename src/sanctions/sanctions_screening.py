from rapidfuzz import fuzz

from .ofac_parser import OFACParser
from src.identity.normalizer import normalize_name


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Minimum similarity required to create an OFAC candidate.
#
# This prevents weak surname-only matches such as:
#
#   Rahul Sharma
#   Rakesh Sharma
#
# from automatically becoming REVIEW candidates.
NAME_CANDIDATE_THRESHOLD = 0.85

# Strong name match.
STRONG_NAME_THRESHOLD = 0.90

# Overall candidate thresholds.
POTENTIAL_MATCH_THRESHOLD = 0.90


# ---------------------------------------------------------
# Sanctions Screener
# ---------------------------------------------------------

class SanctionsScreener:
    """
    Simple deterministic OFAC sanctions screening.

    Pipeline:

        Applicant
            ↓
        Normalize identity
            ↓
        Candidate generation
            ↓
        Fuzzy name comparison
            ↓
        DOB comparison
            ↓
        Country comparison
            ↓
        Candidate scoring
            ↓
        CLEAR / REVIEW / POTENTIAL_MATCH

    No LLM is used.
    """

    def __init__(self, ofac_xml_path: str):

        parser = OFACParser(
            ofac_xml_path
        )

        print(
            "Loading OFAC sanctions list..."
        )

        self.records = parser.parse()

        print(
            f"OFAC records loaded: "
            f"{len(self.records)}"
        )

    # =========================================================
    # NAME COMPARISON
    # =========================================================

    @staticmethod
    def compare_name(
        applicant_name: str,
        ofac_name: str
    ) -> float:
        """
        Compare two names using fuzzy matching.

        Returns:
            Similarity between 0 and 1.
        """

        applicant_name = normalize_name(
            applicant_name
        )

        ofac_name = normalize_name(
            ofac_name
        )

        if not applicant_name or not ofac_name:
            return 0.0

        score = fuzz.token_sort_ratio(
            applicant_name,
            ofac_name
        )

        return round(
            score / 100,
            2
        )

    # =========================================================
    # DOB COMPARISON
    # =========================================================

    @staticmethod
    def compare_dob(
        applicant_dob: str,
        ofac_dobs: list
    ):
        """
        Compare applicant DOB against OFAC DOB values.

        Returns:

            True  -> DOB matches
            False -> DOB available but mismatches
            None  -> DOB unavailable
        """

        if not applicant_dob:
            return None

        if not ofac_dobs:
            return None

        applicant_normalized = (
            SanctionsScreener._normalize_date(
                applicant_dob
            )
        )

        for ofac_dob in ofac_dobs:

            if not ofac_dob:
                continue

            ofac_normalized = (
                SanctionsScreener._normalize_date(
                    ofac_dob
                )
            )

            if (
                applicant_normalized
                == ofac_normalized
            ):
                return True

        return False

    # =========================================================
    # COUNTRY COMPARISON
    # =========================================================

    @staticmethod
    def compare_country(
        applicant_country: str,
        ofac_record: dict
    ):
        """
        Compare applicant country against OFAC
        country information.

        Returns:

            True  -> country matches
            False -> country available but mismatches
            None  -> country unavailable
        """

        if not applicant_country:
            return None

        applicant_country = (
            applicant_country
            .strip()
            .lower()
        )

        countries = []

        countries.extend(
            ofac_record.get(
                "countries",
                []
            )
        )

        countries.extend(
            ofac_record.get(
                "nationalities",
                []
            )
        )

        countries.extend(
            ofac_record.get(
                "citizenships",
                []
            )
        )

        if not countries:
            return None

        for country in countries:

            if not country:
                continue

            if (
                country.strip().lower()
                == applicant_country
            ):
                return True

        return False

    # =========================================================
    # CANDIDATE GENERATION
    # =========================================================

    def find_candidates(
        self,
        applicant_name: str,
        limit: int = 5
    ) -> list:
        """
        Generate possible OFAC candidates.

        Important:

        Weak name matches are ignored.

        This prevents cases such as:

            Rahul Sharma
            Rakesh Sharma

        from becoming sanctions candidates simply
        because they share the surname "Sharma".

        Both primary OFAC names and aliases are checked.
        """

        candidates = []

        for record in self.records:

            # -------------------------------------------------
            # Primary OFAC name
            # -------------------------------------------------

            primary_name = record.get(
                "name",
                ""
            )

            best_name = primary_name

            best_score = self.compare_name(
                applicant_name,
                primary_name
            )

            # -------------------------------------------------
            # Check aliases
            # -------------------------------------------------

            for alias in record.get(
                "aliases",
                []
            ):

                alias_score = self.compare_name(
                    applicant_name,
                    alias
                )

                if alias_score > best_score:

                    best_score = alias_score
                    best_name = alias

            # -------------------------------------------------
            # Ignore weak matches
            # -------------------------------------------------

            if (
                best_score
                < NAME_CANDIDATE_THRESHOLD
            ):
                continue

            candidates.append(
                {
                    "record": record,
                    "matched_name": best_name,
                    "name_similarity": best_score,
                }
            )

        # -----------------------------------------------------
        # Strongest candidates first
        # -----------------------------------------------------

        candidates.sort(
            key=lambda candidate:
                candidate["name_similarity"],
            reverse=True
        )

        return candidates[:limit]

    # =========================================================
    # CANDIDATE ASSESSMENT
    # =========================================================

    def assess_candidate(
        self,
        applicant: dict,
        candidate: dict
    ) -> dict:
        """
        Assess one OFAC candidate using:

        - Name
        - DOB
        - Country

        This does NOT declare that a person is sanctioned.

        It only determines whether the candidate should
        be considered for further review.
        """

        record = candidate["record"]

        name_similarity = candidate[
            "name_similarity"
        ]

        # -----------------------------------------------------
        # DOB
        # -----------------------------------------------------

        dob_match = self.compare_dob(
            applicant.get("dob"),
            record.get(
                "dates_of_birth",
                []
            )
        )

        # -----------------------------------------------------
        # Country
        # -----------------------------------------------------

        country_match = self.compare_country(
            applicant.get("country"),
            record
        )

        # -----------------------------------------------------
        # Candidate score
        # -----------------------------------------------------

        score = name_similarity

        # Strong supporting evidence:
        # matching DOB.
        if dob_match is True:

            score += 0.05

        # DOB mismatch is useful evidence against
        # the candidate.
        elif dob_match is False:

            score -= 0.10

        # Matching country provides some
        # additional support.
        if country_match is True:

            score += 0.05

        # Country mismatch slightly reduces
        # confidence.
        elif country_match is False:

            score -= 0.05

        # Keep score inside 0-1.
        score = max(
            0.0,
            min(1.0, score)
        )

        score = round(
            score,
            2
        )

        # -----------------------------------------------------
        # Determine assessment
        # -----------------------------------------------------

        # Very strong name + supporting attribute.
        if (
            name_similarity
            >= STRONG_NAME_THRESHOLD
            and (
                dob_match is True
                or country_match is True
            )
        ):

            assessment = (
                "POTENTIAL_MATCH"
            )

        # Strong name but supporting attributes
        # are unavailable or inconsistent.
        elif (
            name_similarity
            >= NAME_CANDIDATE_THRESHOLD
        ):

            assessment = "REVIEW"

        else:

            assessment = "CLEAR"

        # -----------------------------------------------------
        # Return candidate result
        # -----------------------------------------------------

        return {
            "ofac_uid": record.get(
                "uid"
            ),
            "ofac_name": record.get(
                "name"
            ),
            "matched_name": candidate.get(
                "matched_name"
            ),
            "sdn_type": record.get(
                "sdn_type"
            ),
            "programs": record.get(
                "programs",
                []
            ),
            "name_similarity": name_similarity,
            "dob_match": dob_match,
            "country_match": country_match,
            "candidate_score": score,
            "assessment": assessment,
        }

    # =========================================================
    # SCREEN APPLICANT
    # =========================================================

    def screen(
        self,
        applicant: dict
    ) -> dict:
        """
        Screen one applicant against the OFAC list.

        Example input:

            {
                "applicant_id": "APP-001",
                "name": "Rahul Sharma",
                "dob": "1995-03-12",
                "country": "India"
            }
        """

        applicant_id = applicant.get(
            "applicant_id"
        )

        applicant_name = applicant.get(
            "name"
        )

        # -----------------------------------------------------
        # Validate applicant
        # -----------------------------------------------------

        if not applicant_name:

            return {
                "applicant_id": applicant_id,
                "screening_status": "ERROR",
                "candidates": [],
                "message": (
                    "Applicant name is missing"
                )
            }

        # -----------------------------------------------------
        # Generate candidates
        # -----------------------------------------------------

        candidates = self.find_candidates(
            applicant_name
        )

        # -----------------------------------------------------
        # No meaningful candidate
        # -----------------------------------------------------

        if not candidates:

            return {
                "applicant_id": applicant_id,
                "screening_status": "CLEAR",
                "candidates": []
            }

        # -----------------------------------------------------
        # Assess candidates
        # -----------------------------------------------------

        assessments = []

        for candidate in candidates:

            assessment = (
                self.assess_candidate(
                    applicant,
                    candidate
                )
            )

            assessments.append(
                assessment
            )

        # -----------------------------------------------------
        # Determine overall screening status
        # -----------------------------------------------------

        potential_matches = [
            result
            for result in assessments
            if result["assessment"]
            == "POTENTIAL_MATCH"
        ]

        reviews = [
            result
            for result in assessments
            if result["assessment"]
            == "REVIEW"
        ]

        if potential_matches:

            screening_status = (
                "POTENTIAL_MATCH"
            )

        elif reviews:

            screening_status = "REVIEW"

        else:

            screening_status = "CLEAR"

        # -----------------------------------------------------
        # Final result
        # -----------------------------------------------------

        return {
            "applicant_id": applicant_id,
            "screening_status": screening_status,
            "candidates": assessments
        }

    # =========================================================
    # DATE NORMALIZATION
    # =========================================================

    @staticmethod
    def _normalize_date(
        date_value: str
    ):
        """
        Normalize common date formats.

        Examples:

            1995-03-12
            12-03-1995
            12 Mar 1995
            26 Dec 1955
        """

        if not date_value:
            return None

        value = (
            date_value
            .strip()
            .lower()
        )

        # -----------------------------------------------------
        # YYYY-MM-DD
        # -----------------------------------------------------

        if (
            len(value) == 10
            and value[4] == "-"
        ):

            return value

        # -----------------------------------------------------
        # DD-MM-YYYY
        # -----------------------------------------------------

        if (
            len(value) == 10
            and value[2] == "-"
        ):

            day = value[0:2]
            month = value[3:5]
            year = value[6:10]

            return (
                f"{year}-"
                f"{month}-"
                f"{day}"
            )

        # -----------------------------------------------------
        # DD Mon YYYY
        #
        # Example:
        # 26 Dec 1955
        # -----------------------------------------------------

        months = {
            "jan": "01",
            "feb": "02",
            "mar": "03",
            "apr": "04",
            "may": "05",
            "jun": "06",
            "jul": "07",
            "aug": "08",
            "sep": "09",
            "oct": "10",
            "nov": "11",
            "dec": "12",
        }

        parts = value.split()

        if len(parts) == 3:

            day = parts[0]

            month = months.get(
                parts[1][:3]
            )

            year = parts[2]

            if (
                month
                and day.isdigit()
                and year.isdigit()
            ):

                return (
                    f"{year}-"
                    f"{month}-"
                    f"{int(day):02d}"
                )

        # -----------------------------------------------------
        # Unknown date format
        # -----------------------------------------------------

        return value

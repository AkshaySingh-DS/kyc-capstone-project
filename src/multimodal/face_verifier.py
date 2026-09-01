import base64
import json
import re
from pathlib import Path

from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_ID = (
    "meta-llama/"
    "llama-4-maverick-17b-128e-instruct-fp8"
)

WATSONX_URL = (
    "https://us-south.ml.cloud.ibm.com"
)

PROJECT_ID = "skills-network"

# Demo threshold only.
#
# >= 0.80 -> CONSISTENT
# <  0.80 -> REVIEW
#
# This is NOT a production biometric threshold.
SIMILARITY_THRESHOLD = 0.80


# ---------------------------------------------------------
# Face Verifier
# ---------------------------------------------------------

class FaceVerifier:
    """
    Multimodal face consistency checker.

    Compares an ID photograph and a selfie using
    Llama 4 Maverick vision capabilities.

    This component is designed for the synthetic
    KYC demonstration.

    It does NOT perform production-grade biometric
    identity verification.
    """

    def __init__(
        self,
        similarity_threshold: float = SIMILARITY_THRESHOLD
    ):

        credentials = Credentials(
            url=WATSONX_URL
        )

        self.model = ModelInference(
            model_id=MODEL_ID,
            credentials=credentials,
            project_id=PROJECT_ID,
            params={
                "temperature": 0,
                "max_tokens": 300
            }
        )

        self.similarity_threshold = (
            similarity_threshold
        )

    # =====================================================
    # IMAGE ENCODING
    # =====================================================

    @staticmethod
    def encode_image(
        image_path: str
    ) -> str:
        """
        Convert an image to Base64.
        """

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not image_path.is_file():

            raise ValueError(
                f"Path is not a file: {image_path}"
            )

        with open(
            image_path,
            "rb"
        ) as image_file:

            return base64.b64encode(
                image_file.read()
            ).decode("utf-8")

    # =====================================================
    # JSON EXTRACTION
    # =====================================================

    @staticmethod
    def extract_json(
        response_text: str
    ) -> dict:
        """
        Extract JSON from the model response.

        Handles:

        - normal JSON
        - markdown code fences
        - <python_start> / <python_end>
        - additional text around JSON
        """

        if not response_text:

            raise ValueError(
                "Empty model response"
            )

        text = response_text.strip()

        # Remove common wrappers.
        text = re.sub(
            r"<python_start>|<python_end>",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = re.sub(
            r"```json|```",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = text.strip()

        # Try complete response.
        try:

            result = json.loads(
                text
            )

            if isinstance(result, dict):

                return result

        except json.JSONDecodeError:
            pass

        # Try JSON object inside response.
        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:

            json_text = text[
                start:end + 1
            ]

            try:

                result = json.loads(
                    json_text
                )

                if isinstance(result, dict):

                    return result

            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Unable to parse JSON from model response"
        )

    # =====================================================
    # FALLBACK PARSER
    # =====================================================

    @staticmethod
    def extract_from_malformed_response(
        response_text: str
    ):
        """
        Recover useful information from an almost-valid
        JSON response.

        Example:

        {
            "face_similarity": 0.95,
            ["facial features are highly consistent"]
        }

        The model has provided useful information but
        produced invalid JSON.

        We recover the similarity and observation rather
        than immediately marking the analysis uncertain.
        """

        if not response_text:

            return None

        text = response_text.strip()

        # -------------------------------------------------
        # Extract similarity
        # -------------------------------------------------

        similarity_match = re.search(
            r'"face_similarity"\s*:\s*([01](?:\.\d+)?)',
            text,
            flags=re.IGNORECASE
        )

        if not similarity_match:

            return None

        similarity = float(
            similarity_match.group(1)
        )

        # -------------------------------------------------
        # Extract quoted observation strings
        # -------------------------------------------------

        observations = []

        after_similarity = text[
            similarity_match.end():
        ]

        quoted_strings = re.findall(
            r'"([^"]{10,})"',
            after_similarity
        )

        ignored_values = {
            "assessment",
            "observations",
            "face_similarity"
        }

        for value in quoted_strings:

            if value in ignored_values:
                continue

            observations.append(
                value.strip()
            )

        return {
            "face_similarity": similarity,
            "observations": observations
        }

    # =====================================================
    # NORMALIZE RESULT
    # =====================================================

    @staticmethod
    def normalize_similarity(
        similarity
    ) -> float:
        """
        Validate and normalize the similarity score.
        """

        try:

            similarity = float(
                similarity
            )

        except (
            TypeError,
            ValueError
        ):

            raise ValueError(
                "Invalid face similarity score."
            )

        if not 0.0 <= similarity <= 1.0:

            raise ValueError(
                "Face similarity must be between "
                "0.0 and 1.0."
            )

        return round(
            similarity,
            2
        )

    # =====================================================
    # VERIFY
    # =====================================================

    def verify(
        self,
        id_photo_path: str,
        selfie_path: str
    ) -> dict:
        """
        Compare an ID photograph and a selfie.

        Returns a normalized result containing:

        - image names
        - similarity score
        - assessment
        - observations
        """

        id_photo_path = Path(
            id_photo_path
        )

        selfie_path = Path(
            selfie_path
        )

        # -------------------------------------------------
        # Validate input files
        # -------------------------------------------------

        if not id_photo_path.exists():

            return {
                "id_photo": id_photo_path.name,
                "selfie": selfie_path.name,
                "assessment": "ANALYSIS_UNCERTAIN",
                "observations": [],
                "error": (
                    f"ID photo not found: "
                    f"{id_photo_path}"
                )
            }

        if not selfie_path.exists():

            return {
                "id_photo": id_photo_path.name,
                "selfie": selfie_path.name,
                "assessment": "ANALYSIS_UNCERTAIN",
                "observations": [],
                "error": (
                    f"Selfie not found: "
                    f"{selfie_path}"
                )
            }

        # -------------------------------------------------
        # Encode images
        # -------------------------------------------------

        try:

            id_image = self.encode_image(
                str(id_photo_path)
            )

            selfie_image = self.encode_image(
                str(selfie_path)
            )

        except Exception as e:

            return {
                "id_photo": id_photo_path.name,
                "selfie": selfie_path.name,
                "assessment": "ANALYSIS_UNCERTAIN",
                "observations": [],
                "error": str(e)
            }

        # -------------------------------------------------
        # Vision prompt
        # -------------------------------------------------

        prompt = """
You are a visual identity consistency assistant
for a synthetic KYC demonstration.

Two images are provided:

IMAGE 1:
Applicant ID photograph.

IMAGE 2:
Applicant selfie.

Compare the visible facial characteristics between
the two images.

Consider:

- overall facial structure
- face shape
- eyes
- nose
- mouth
- other stable visible facial characteristics

Provide a visual similarity score between 0.0 and 1.0.

0.0 means very different facial appearance.

1.0 means very similar facial appearance.

IMPORTANT:

This is ONLY a demonstration.

Do NOT make a legal, KYC, or identity decision.

Do NOT claim that the person is definitely the same
or definitely different.

Return ONLY valid JSON.

Use EXACTLY this structure:

{
  "face_similarity": 0.95,
  "observations": [
    "Facial structure and features appear highly similar."
  ]
}

IMPORTANT:

The key "observations" MUST be included.

Do not return an array directly after face_similarity.

Do not include markdown.

Do not include explanations outside the JSON.
"""

        # -------------------------------------------------
        # Multimodal message
        # -------------------------------------------------

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                + id_image
                            )
                        }
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": (
                                "data:image/png;base64,"
                                + selfie_image
                            )
                        }
                    }
                ]
            }
        ]

        # -------------------------------------------------
        # Call vision model
        # -------------------------------------------------

        try:

            response = self.model.chat(
                messages=messages
            )

            raw_response = (
                response["choices"][0]
                ["message"]["content"]
                .strip()
            )

        except Exception as e:

            return {
                "id_photo": id_photo_path.name,
                "selfie": selfie_path.name,
                "assessment": "ANALYSIS_UNCERTAIN",
                "observations": [],
                "error": (
                    f"Vision model failed: {str(e)}"
                )
            }

        # -------------------------------------------------
        # Parse model response
        # -------------------------------------------------

        result = None

        try:

            result = self.extract_json(
                raw_response
            )

        except ValueError:

            # -------------------------------------------------
            # Try recovery from malformed JSON.
            # -------------------------------------------------

            result = (
                self.extract_from_malformed_response(
                    raw_response
                )
            )

            if result is None:

                return {
                    "id_photo": id_photo_path.name,
                    "selfie": selfie_path.name,
                    "assessment": "ANALYSIS_UNCERTAIN",
                    "observations": [],
                    "raw_response": raw_response,
                    "error": (
                        "Unable to extract face similarity "
                        "from model response."
                    )
                }

        # -------------------------------------------------
        # Validate similarity
        # -------------------------------------------------

        try:

            similarity = self.normalize_similarity(
                result.get(
                    "face_similarity"
                )
            )

        except ValueError as e:

            return {
                "id_photo": id_photo_path.name,
                "selfie": selfie_path.name,
                "assessment": "ANALYSIS_UNCERTAIN",
                "observations": [],
                "raw_response": raw_response,
                "error": str(e)
            }

        # -------------------------------------------------
        # Determine assessment
        #
        # Assessment is calculated by our application,
        # not trusted from the LLM.
        # -------------------------------------------------

        if (
            similarity
            >= self.similarity_threshold
        ):

            assessment = "CONSISTENT"

        else:

            assessment = "REVIEW"

        # -------------------------------------------------
        # Normalize observations
        # -------------------------------------------------

        observations = result.get(
            "observations",
            []
        )

        if not isinstance(
            observations,
            list
        ):

            observations = [
                str(observations)
            ]

        observations = [
            str(item).strip()
            for item in observations
            if str(item).strip()
        ]

        # -------------------------------------------------
        # Final result
        # -------------------------------------------------

        return {
            "id_photo": id_photo_path.name,
            "selfie": selfie_path.name,
            "face_similarity": similarity,
            "assessment": assessment,
            "observations": observations
        }
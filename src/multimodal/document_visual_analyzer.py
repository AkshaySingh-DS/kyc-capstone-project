import base64
import json
import mimetypes
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


# ---------------------------------------------------------
# Document Visual Analyzer
# ---------------------------------------------------------

class DocumentVisualAnalyzer:
    """
    Analyze KYC document images using
    Llama 4 Maverick vision capabilities.

    IMPORTANT:

    This component only identifies potential
    visual indicators.

    It does NOT determine whether a document
    is definitely genuine or forged.
    """

    def __init__(self):

        credentials = Credentials(
            url=WATSONX_URL
        )

        self.model = ModelInference(
            model_id=MODEL_ID,
            credentials=credentials,
            project_id=PROJECT_ID,
            params={
                "temperature": 0,
                "max_tokens": 400
            }
        )

    # =====================================================
    # IMAGE ENCODING
    # =====================================================

    @staticmethod
    def encode_image(
        image_path: str
    ) -> str:
        """
        Convert image to Base64.
        """

        image_path = Path(
            image_path
        )

        if not image_path.exists():

            raise FileNotFoundError(
                f"Document not found: "
                f"{image_path}"
            )

        if not image_path.is_file():

            raise ValueError(
                f"Document path is not a file: "
                f"{image_path}"
            )

        with open(
            image_path,
            "rb"
        ) as image_file:

            return base64.b64encode(
                image_file.read()
            ).decode("utf-8")

    # =====================================================
    # MIME TYPE
    # =====================================================

    @staticmethod
    def get_mime_type(
        image_path: str
    ) -> str:
        """
        Determine the MIME type of the image.
        """

        image_path = Path(
            image_path
        )

        mime_type, _ = mimetypes.guess_type(
            image_path.name
        )

        if not mime_type:

            mime_type = "image/png"

        return mime_type

    # =====================================================
    # JSON EXTRACTION
    # =====================================================

    @staticmethod
    def extract_json(
        response_text: str
    ) -> dict:
        """
        Extract JSON from the model response.

        Handles responses such as:

            {
                ...
            }

        and:

            ```json
            {
                ...
            }
            ```
        """

        if not response_text:

            raise ValueError(
                "Empty model response"
            )

        text = response_text.strip()

        # -------------------------------------------------
        # Remove markdown code fences
        # -------------------------------------------------

        text = re.sub(
            r"```json",
            "",
            text,
            flags=re.IGNORECASE
        )

        text = text.replace(
            "```",
            ""
        ).strip()

        # -------------------------------------------------
        # Direct JSON
        # -------------------------------------------------

        try:

            return json.loads(
                text
            )

        except json.JSONDecodeError:
            pass

        # -------------------------------------------------
        # Find JSON object inside response
        # -------------------------------------------------

        start = text.find("{")
        end = text.rfind("}")

        if start != -1 and end != -1:

            json_text = text[
                start:end + 1
            ]

            try:

                return json.loads(
                    json_text
                )

            except json.JSONDecodeError:
                pass

        raise ValueError(
            "Unable to parse JSON from model response"
        )

    # =====================================================
    # NORMALIZE RESULT
    # =====================================================

    @staticmethod
    def normalize_result(
        result: dict
    ) -> dict:
        """
        Ensure the model response follows
        our expected Phase 6 structure.
        """

        observations = result.get(
            "observations",
            []
        )

        tampering_indicators = result.get(
            "tampering_indicators",
            []
        )

        assessment = result.get(
            "assessment",
            "ANALYSIS_UNCERTAIN"
        )

        # -------------------------------------------------
        # Ensure lists
        # -------------------------------------------------

        if not isinstance(
            observations,
            list
        ):

            observations = [
                str(observations)
            ]

        if not isinstance(
            tampering_indicators,
            list
        ):

            tampering_indicators = [
                str(tampering_indicators)
            ]

        # -------------------------------------------------
        # Restrict assessment values
        # -------------------------------------------------

        allowed_assessments = {
            "NO_OBVIOUS_TAMPERING",
            "POTENTIAL_TAMPERING_INDICATORS",
            "ANALYSIS_UNCERTAIN"
        }

        if assessment not in allowed_assessments:

            assessment = (
                "ANALYSIS_UNCERTAIN"
            )

        return {
            "assessment": assessment,
            "observations": observations,
            "tampering_indicators": (
                tampering_indicators
            )
        }

    # =====================================================
    # DOCUMENT ANALYSIS
    # =====================================================

    def analyze(
        self,
        image_path: str
    ) -> dict:
        """
        Analyze one document image.

        The image path is supplied dynamically,
        allowing this method to work with any
        applicant/document.
        """

        image_path = Path(
            image_path
        )

        # -------------------------------------------------
        # Validate document
        # -------------------------------------------------

        if not image_path.exists():

            raise FileNotFoundError(
                f"Document not found: "
                f"{image_path}"
            )

        if not image_path.is_file():

            raise ValueError(
                f"Document path is not a file: "
                f"{image_path}"
            )

        print(
            f"\nAnalyzing document: "
            f"{image_path.name}"
        )

        # -------------------------------------------------
        # Encode image
        # -------------------------------------------------

        encoded_image = (
            self.encode_image(
                str(image_path)
            )
        )

        # -------------------------------------------------
        # Determine MIME type
        # -------------------------------------------------

        mime_type = (
            self.get_mime_type(
                str(image_path)
            )
        )

        # -------------------------------------------------
        # Vision prompt
        # -------------------------------------------------

        prompt = """
You are a document visual analysis assistant for a
synthetic KYC demonstration.

Analyze ONLY the visible characteristics of the supplied
document image.

Your purpose is to identify UNEXPECTED visual
characteristics that could potentially indicate that a
document has been altered.

Look for things such as:

- inconsistent fonts within otherwise similar fields
- unusual text alignment
- inconsistent spacing
- text that appears visually overlaid
- suspicious borders or boxes around a particular field
- inconsistent image quality between different areas
- visible editing or manipulation artifacts
- photograph areas that appear unusually altered
- portions of the document that visually differ from
  surrounding content

IMPORTANT — SYNTHETIC DOCUMENT RULES:

This project intentionally uses synthetic documents.

Therefore, the following are EXPECTED and MUST NOT be
classified as tampering indicators:

- "SYNTHETIC"
- "SYNTHETIC - FOR DEMO ONLY"
- "DEMO"
- "(DEMO)"
- "(Example)"
- "Fictional Identity Card"
- "Fictional Demo Authority"
- placeholder or illustrated photographs
- fictional document numbers
- fictional applicant information

These characteristics identify the document as synthetic
for our demonstration. They are NOT evidence that the
document was visually altered.

IMPORTANT:

1. Do NOT determine that the document is definitely forged.

2. Do NOT make a legal, regulatory, or KYC decision.

3. Do NOT treat the synthetic/demo labels listed above as
   tampering indicators.

4. Report only unexpected visual characteristics.

5. If the document appears visually consistent and there
   are no unexpected alteration indicators, use:

   "NO_OBVIOUS_TAMPERING"

6. If there are genuine unexpected visual characteristics
   that could warrant further investigation, use:

   "POTENTIAL_TAMPERING_INDICATORS"

7. If the visual characteristics cannot be reliably
   determined, use:

   "ANALYSIS_UNCERTAIN"

8. Keep observations concise and factual.

9. Do not invent visual problems that are not visible
   in the image.

10. Return ONLY valid JSON.

Use EXACTLY this structure:

{
  "observations": [
    "observation 1",
    "observation 2"
  ],
  "tampering_indicators": [],
  "assessment": "NO_OBVIOUS_TAMPERING"
}

The assessment MUST be exactly one of:

"NO_OBVIOUS_TAMPERING"

"POTENTIAL_TAMPERING_INDICATORS"

"ANALYSIS_UNCERTAIN"

Remember:

Synthetic content is expected in this demonstration.

Do not confuse synthetic content with evidence of
document tampering.
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
                                f"data:{mime_type};base64,"
                                f"{encoded_image}"
                            )
                        }
                    }
                ]
            }
        ]

        # -------------------------------------------------
        # Call Llama 4 Maverick
        # -------------------------------------------------

        try:

            response = self.model.chat(
                messages=messages
            )

        except Exception as e:

            return {
                "document": image_path.name,
                "assessment": (
                    "ANALYSIS_UNCERTAIN"
                ),
                "observations": [],
                "tampering_indicators": [],
                "error": (
                    f"Vision model failed: "
                    f"{str(e)}"
                )
            }

        # -------------------------------------------------
        # Extract model response
        # -------------------------------------------------

        try:

            raw_response = (
                response["choices"][0]
                ["message"]["content"]
            )

            raw_response = (
                raw_response.strip()
            )

        except (
            KeyError,
            IndexError,
            TypeError
        ):

            return {
                "document": image_path.name,
                "assessment": (
                    "ANALYSIS_UNCERTAIN"
                ),
                "observations": [],
                "tampering_indicators": [],
                "error": (
                    "Unexpected response format "
                    "from vision model."
                )
            }

        # -------------------------------------------------
        # Parse JSON
        # -------------------------------------------------

        try:

            parsed_result = (
                self.extract_json(
                    raw_response
                )
            )

            normalized_result = (
                self.normalize_result(
                    parsed_result
                )
            )

            return {
                "document": image_path.name,
                **normalized_result
            }

        except ValueError:

            return {
                "document": image_path.name,
                "assessment": (
                    "ANALYSIS_UNCERTAIN"
                ),
                "observations": [],
                "tampering_indicators": [],
                "raw_response": raw_response,
                "error": (
                    "Model returned invalid JSON."
                )
            }


# =========================================================
# Manual Test
# =========================================================

if __name__ == "__main__":

    analyzer = DocumentVisualAnalyzer()

    image_path = input(
        "\nEnter document image path: "
    ).strip()

    if not image_path:

        print(
            "No document path supplied."
        )

        raise SystemExit(1)

    result = analyzer.analyze(
        image_path
    )

    print(
        "\n===== DOCUMENT VISUAL ANALYSIS =====\n"
    )

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

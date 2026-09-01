import json
from pathlib import Path

import face_recognition


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

# Lower face distance means the faces are more similar.
# 0.60 is a commonly used starting point.
FACE_DISTANCE_THRESHOLD = 0.60


# ---------------------------------------------------------
# Face Verifier
# ---------------------------------------------------------

class FaceVerifier:
    """
    Simple face verification component.

    Compares two photographs using face embeddings.

    Example:

        ID Photo
            ↓
        Face Embedding
            ↓
          Compare
            ↑
        Face Embedding
            ↑
        Selfie

    This component does NOT make an identity or KYC
    decision. It only reports face similarity.
    """

    # =====================================================
    # LOAD IMAGE
    # =====================================================

    @staticmethod
    def load_image(image_path: str):
        """
        Load an image from disk.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        if not image_path.is_file():
            raise ValueError(
                f"Image path is not a file: {image_path}"
            )

        return face_recognition.load_image_file(
            str(image_path)
        )

    # =====================================================
    # EXTRACT FACE EMBEDDING
    # =====================================================

    @staticmethod
    def get_face_encoding(
        image,
        image_path: str
    ):
        """
        Detect a face and generate its embedding.

        For this demo we expect exactly one face
        in each image.
        """

        face_locations = (
            face_recognition.face_locations(
                image
            )
        )

        if len(face_locations) == 0:

            raise ValueError(
                f"No face detected in: "
                f"{image_path}"
            )

        if len(face_locations) > 1:

            raise ValueError(
                f"Multiple faces detected in: "
                f"{image_path}. "
                "Expected one face."
            )

        encodings = (
            face_recognition.face_encodings(
                image,
                face_locations
            )
        )

        if not encodings:

            raise ValueError(
                f"Unable to generate face embedding "
                f"for: {image_path}"
            )

        return encodings[0]

    # =====================================================
    # VERIFY FACES
    # =====================================================

    def verify(
        self,
        id_photo_path: str,
        selfie_path: str
    ) -> dict:
        """
        Compare an ID photograph with a selfie.

        Returns:

            face_distance
            similarity
            assessment
        """

        id_photo_path = Path(
            id_photo_path
        )

        selfie_path = Path(
            selfie_path
        )

        print(
            f"\nComparing:"
            f"\n  ID Photo : {id_photo_path.name}"
            f"\n  Selfie   : {selfie_path.name}"
        )

        # -------------------------------------------------
        # Load images
        # -------------------------------------------------

        id_image = self.load_image(
            str(id_photo_path)
        )

        selfie_image = self.load_image(
            str(selfie_path)
        )

        # -------------------------------------------------
        # Generate embeddings
        # -------------------------------------------------

        id_encoding = self.get_face_encoding(
            id_image,
            str(id_photo_path)
        )

        selfie_encoding = self.get_face_encoding(
            selfie_image,
            str(selfie_path)
        )

        # -------------------------------------------------
        # Calculate face distance
        # -------------------------------------------------

        distance = face_recognition.face_distance(
            [id_encoding],
            selfie_encoding
        )[0]

        distance = float(distance)

        # -------------------------------------------------
        # Convert distance to simple similarity score
        #
        # This is a demo-friendly normalized score.
        # It should NOT be interpreted as a probability.
        # -------------------------------------------------

        similarity = max(
            0.0,
            min(
                1.0,
                1.0 - distance
            )
        )

        similarity = round(
            similarity,
            2
        )

        # -------------------------------------------------
        # Determine assessment
        # -------------------------------------------------

        if distance <= FACE_DISTANCE_THRESHOLD:

            assessment = "CONSISTENT"

        else:

            assessment = "REVIEW"

        # -------------------------------------------------
        # Return result
        # -------------------------------------------------

        return {
            "id_photo": id_photo_path.name,
            "selfie": selfie_path.name,
            "face_distance": round(
                distance,
                3
            ),
            "face_similarity": similarity,
            "assessment": assessment
        }


# =========================================================
# Manual Test
# =========================================================

if __name__ == "__main__":

    verifier = FaceVerifier()

    id_photo = input(
        "\nEnter ID photo path: "
    ).strip()

    selfie = input(
        "Enter selfie path: "
    ).strip()

    try:

        result = verifier.verify(
            id_photo,
            selfie
        )

        print(
            "\n===== FACE VERIFICATION =====\n"
        )

        print(
            json.dumps(
                result,
                indent=2
            )
        )

    except Exception as e:

        print(
            "\n===== FACE VERIFICATION ERROR =====\n"
        )

        print(str(e))

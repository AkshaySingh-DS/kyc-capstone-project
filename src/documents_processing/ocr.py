import json
from pathlib import Path
from paddleocr import PaddleOCR
from src.utils.logger import logger


class OCRProcessor:
    """Handles OCR processing using PaddleOCR."""

    def __init__(self):

        self.ocr = PaddleOCR(
            lang="en",
            enable_mkldnn=False,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )

    def extract_text(self, image_path: str) -> str:
        """
        Run OCR on a document image and return
        the extracted text as a single string.
        """

        image_path = Path(image_path)

        if not image_path.exists():
            raise FileNotFoundError(
                f"Document not found: {image_path}"
            )
        
        logger.info(
            f"Paddle OCR is being initialized..."
        )
        
        result = self.ocr.predict(str(image_path))

        text_lines = []

        for page in result:

            data = page.json

            if isinstance(data, str):
                data = json.loads(data)

            rec_texts = (
                data.get("res", {})
                .get("rec_texts", [])
            )

            text_lines.extend(rec_texts)

        return "\n".join(text_lines)
import json

from src.documents_processing.document_processor import DocumentProcessor


def main():

    processor = DocumentProcessor()

    image_path = (
        "synthetic_documents/"
        "APP-001/"
        "address_proof.png"
    )

    result = processor.process_document(
        image_path
    )

    print("\n===== DOCUMENT PROCESSING RESULT =====\n")

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )


if __name__ == "__main__":
    main()
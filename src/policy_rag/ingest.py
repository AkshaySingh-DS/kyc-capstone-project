from pathlib import Path
import json

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma

from langchain_ibm import WatsonxEmbeddings
from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

JSON_FILE = (
    BASE_DIR
    / "data"
    / "regulations"
    / "rbi"
    / "extracted"
    / "kyc_relevant_sections.json"
)

CHROMA_DIR = BASE_DIR / "data" / "chroma"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

COLLECTION_NAME = "rbi_kyc_policy"


# ---------------------------------------------------------
# Watsonx Embeddings
# ---------------------------------------------------------

embed_params = {
    EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
    EmbedTextParamsMetaNames.RETURN_OPTIONS: {
        "input_text": True
    },
}

embeddings = WatsonxEmbeddings(
    model_id="ibm/granite-embedding-278m-multilingual",
    url="https://us-south.ml.cloud.ibm.com",
    project_id="skills-network",
    params=embed_params
)


# ---------------------------------------------------------
# Load RBI sections
# ---------------------------------------------------------

def load_documents():

    with open(JSON_FILE, "r", encoding="utf-8") as file:
        data = json.load(file)

    documents = []

    for section in data["sections"]:

        document = Document(
            page_content=section["text"],
            metadata={
                "section_id": section["section_id"],
                "title": section["title"],
                "pages": ",".join(
                    str(page)
                    for page in section["source_pages"]
                ),
                "paragraphs": ",".join(
                    section["paragraphs"]
                ),
                "source": data["source_document"]["title"],
                "document_id": data["source_document"]["document_id"]
            }
        )

        documents.append(document)

    return documents


# ---------------------------------------------------------
# Create Chroma Vector Store
# ---------------------------------------------------------

def ingest():

    print("Loading RBI KYC policy...")

    documents = load_documents()

    print(f"Documents loaded: {len(documents)}")

    print("Creating Chroma vector store...")

    Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name=COLLECTION_NAME
    )

    print("\nRBI policy ingestion completed.")
    print(f"Collection : {COLLECTION_NAME}")
    print(f"Documents  : {len(documents)}")
    print(f"Database   : {CHROMA_DIR}")


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":
    ingest()
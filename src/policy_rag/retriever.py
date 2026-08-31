from pathlib import Path

import chromadb

from ibm_watsonx_ai.metanames import EmbedTextParamsMetaNames
from langchain_ibm import WatsonxEmbeddings


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

CHROMA_DIR = BASE_DIR / "data" / "chroma"


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

COLLECTION_NAME = "rbi_kyc_policy"

WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID = "skills-network"

EMBEDDING_MODEL = "ibm/granite-embedding-278m-multilingual"


# ---------------------------------------------------------
# Retriever
# ---------------------------------------------------------

class RBIPolicyRetriever:

    def __init__(self):

        # WatsonX embedding configuration
        embed_params = {
            EmbedTextParamsMetaNames.TRUNCATE_INPUT_TOKENS: 3,
            EmbedTextParamsMetaNames.RETURN_OPTIONS: {
                "input_text": True
            },
        }

        self.embeddings = WatsonxEmbeddings(
            model_id=EMBEDDING_MODEL,
            url=WATSONX_URL,
            project_id=PROJECT_ID,
            params=embed_params
        )

        # Connect to existing ChromaDB
        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIR)
        )

        self.collection = self.client.get_collection(
            name=COLLECTION_NAME
        )

    def search(self, question: str, top_k: int = 3):
        """
        Search the RBI policy collection using
        WatsonX embeddings.
        """

        # Create query embedding
        query_embedding = self.embeddings.embed_query(
            question
        )

        # Search ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )

        return results


# ---------------------------------------------------------
# Simple test
# ---------------------------------------------------------

if __name__ == "__main__":

    retriever = RBIPolicyRetriever()

    question = input(
        "\nAsk an RBI KYC question: "
    ).strip()

    results = retriever.search(question)

    print("\n===== RELEVANT RBI SECTIONS =====\n")

    if not results["documents"][0]:

        print("No relevant RBI sections found.")

    else:

        for i in range(len(results["documents"][0])):

            print(f"--- Result {i + 1} ---")

            metadata = results["metadatas"][0][i]

            print("Title:")
            print(metadata.get("title", "N/A"))

            print("\nSection:")
            print(metadata.get("section_id", "N/A"))

            print("\nPages:")
            print(metadata.get("pages", "N/A"))

            print("\nContent:")
            print(results["documents"][0][i][:1000])

            print("\n")

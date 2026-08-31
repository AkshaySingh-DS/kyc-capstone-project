from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials

from retriever import RBIPolicyRetriever


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_ID = "ibm/granite-4-h-small"

WATSONX_URL = "https://us-south.ml.cloud.ibm.com"

PROJECT_ID = "skills-network"


# ---------------------------------------------------------
# Policy Answer
# ---------------------------------------------------------

class RBIPolicyAnswer:

    def __init__(self):

        # Watsonx credentials
        # Authentication is already configured in the
        # Skills Network lab environment.
        credentials = Credentials(
            url=WATSONX_URL
        )

        # Watsonx LLM
        self.model = ModelInference(
            model_id=MODEL_ID,
            credentials=credentials,
            project_id=PROJECT_ID,
            params={
                "temperature": 0,
                "max_tokens": 500
            }
        )

        # RBI policy retriever
        self.retriever = RBIPolicyRetriever()

    def answer(self, question: str, top_k: int = 3) -> dict:
        """
        Retrieve relevant RBI policy sections and generate
        an answer using only the retrieved evidence.
        """

        # -------------------------------------------------
        # 1. Retrieve relevant policy sections
        # -------------------------------------------------

        results = self.retriever.search(
            question,
            top_k=top_k
        )

        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]

        if not documents:
            return {
                "question": question,
                "answer": "Insufficient evidence in the retrieved RBI policy.",
                "sources": []
            }

        # -------------------------------------------------
        # 2. Build context
        # -------------------------------------------------

        context_parts = []

        for i, document in enumerate(documents):

            metadata = metadatas[i]

            context_parts.append(
                f"""
SOURCE {i + 1}

Title:
{metadata.get("title", "Unknown")}

Section:
{metadata.get("section_id", "Unknown")}

Pages:
{metadata.get("pages", "Unknown")}

Content:
{document}
"""
            )

        context = "\n".join(context_parts)

        # -------------------------------------------------
        # 3. Strict RAG prompt
        # -------------------------------------------------

        prompt = f"""
You are an RBI KYC policy assistant.

Answer the user's question ONLY using the retrieved
RBI policy evidence provided below.

IMPORTANT RULES:

1. Do not invent RBI requirements.
2. Do not use outside knowledge.
3. Do not make assumptions.
4. If the evidence is insufficient, say:
   "Insufficient evidence in the retrieved RBI policy."
5. Keep the answer concise and factual.
6. Mention the relevant section and page when available.

USER QUESTION:
{question}

RETRIEVED RBI POLICY EVIDENCE:
{context}

ANSWER:
"""

        # -------------------------------------------------
        # 4. Call Watsonx
        # -------------------------------------------------

        try:

            response = self.model.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

            answer = response[
                "choices"
            ][0][
                "message"
            ][
                "content"
            ].strip()

        except Exception as e:

            return {
                "question": question,
                "answer": f"LLM generation failed: {str(e)}",
                "sources": []
            }

        # -------------------------------------------------
        # 5. Build sources
        # -------------------------------------------------

        sources = []

        for metadata in metadatas:

            sources.append({
                "section": metadata.get("section_id"),
                "title": metadata.get("title"),
                "pages": metadata.get("pages")
            })

        return {
            "question": question,
            "answer": answer,
            "sources": sources
        }


# ---------------------------------------------------------
# Simple test
# ---------------------------------------------------------

if __name__ == "__main__":

    policy_answer = RBIPolicyAnswer()

    question = input(
        "\nAsk an RBI KYC policy question: "
    )

    result = policy_answer.answer(question)

    print("\n===== POLICY ANSWER =====\n")

    print(result["answer"])

    print("\n===== SOURCES =====\n")

    for source in result["sources"]:

        print(
            f"Section: {source['section']} | "
            f"Pages: {source['pages']}"
        )
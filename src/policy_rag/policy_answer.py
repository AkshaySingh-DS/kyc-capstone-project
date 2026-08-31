from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai import Credentials

from retriever import RBIPolicyRetriever


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

MODEL_ID  = "meta-llama/llama-4-maverick-17b-128e-instruct-fp8"
# MODEL_ID = "ibm/granite-4-h-small"
WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID = "skills-network"


# ---------------------------------------------------------
# Policy Answer
# ---------------------------------------------------------

class RBIPolicyAnswer:

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
                "max_tokens": 500
            }
        )

        self.retriever = RBIPolicyRetriever()

    def answer(self, question: str, top_k: int = 5) -> dict:

        # -------------------------------------------------
        # 1. Retrieve RBI policy evidence
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
--- RBI POLICY SOURCE {i + 1} ---

Title:
{metadata.get("title", "Unknown")}

Section:
{metadata.get("section_id", "Unknown")}

PDF Pages:
{metadata.get("pages", "Unknown")}

Policy Text:
{document}
"""
            )

        context = "\n".join(context_parts)

        # -------------------------------------------------
        # 3. Strict policy prompt
        # -------------------------------------------------

        prompt = f"""
You are an RBI KYC Policy Assistant.

Your task is to answer the user's question using ONLY
the RBI policy passages provided below.

IMPORTANT INSTRUCTIONS:

1. The provided RBI policy passages are the ONLY source
   of truth.

2. Read the passages carefully and determine whether they
   contain an answer to the user's question.

3. Understand the meaning of the question semantically.
   The wording of the question does not need to exactly
   match the wording in the policy.

4. If a retrieved policy passage directly answers the
   question, provide the answer based on that passage.

5. Do NOT invent, assume, or add any RBI requirement that
   is not present in the retrieved passages.

6. If the retrieved passages genuinely do not contain
   enough information to answer the question, respond:

   "Insufficient evidence in the retrieved RBI policy."

7. Keep the answer concise and factual.

8. Always mention the relevant RBI section and PDF page
   when the information is available.

USER QUESTION:
{question}

RETRIEVED RBI POLICY:

{context}

Now answer the user's question using ONLY the retrieved
RBI policy evidence.

ANSWER:
"""

        # -------------------------------------------------
        # 4. Generate answer using Watsonx
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

            answer = (
                response["choices"][0]["message"]["content"]
                .strip()
            )

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
# Test
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


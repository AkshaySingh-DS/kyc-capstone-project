import logging

from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models import ModelInference

from retriever import RBIPolicyRetriever


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

WATSONX_URL = "https://us-south.ml.cloud.ibm.com"
PROJECT_ID = "skills-network"
MODEL_ID = "ibm/granite-4-h-small"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------
# RBI Policy Answerer
# ---------------------------------------------------------

class RBIPolicyAnswerer:

    def __init__(self):

        # WatsonX credentials
        credentials = Credentials(
            url=WATSONX_URL
        )

        # WatsonX LLM
        self.model = ModelInference(
            model_id=MODEL_ID,
            credentials=credentials,
            project_id=PROJECT_ID,
            params={
                "temperature": 0,
                "max_tokens": 500
            }
        )

        # Existing RBI retriever
        self.retriever = RBIPolicyRetriever()

    def answer(self, question: str) -> str:

        # -------------------------------------------------
        # 1. Retrieve relevant RBI policy sections
        # -------------------------------------------------

        logger.info("Retrieving RBI policy evidence...")

        results = self.retriever.search(
            question,
            top_k=3
        )

        documents = results["documents"][0]
        metadatas = results["metadatas"][0]

        if not documents:
            return (
                "The retrieved RBI evidence is insufficient "
                "to answer this question."
            )

        # -------------------------------------------------
        # 2. Build evidence context
        # -------------------------------------------------

        evidence = []

        for i, document in enumerate(documents):

            metadata = metadatas[i]

            evidence.append(
                f"""
SOURCE {i + 1}

Section:
{metadata.get("title", "Unknown")}

Section ID:
{metadata.get("section_id", "Unknown")}

Source Pages:
{metadata.get("pages", "Unknown")}

Content:
{document}
"""
            )

        evidence_text = "\n".join(evidence)

        # -------------------------------------------------
        # 3. Strict RBI-grounded prompt
        # -------------------------------------------------

        prompt = f"""
You are an RBI KYC Policy Assistant.

Answer the user's question ONLY using the RBI policy
evidence provided below.

STRICT RULES:

1. Use only the provided RBI evidence.
2. Do not use your general knowledge.
3. Do not invent RBI rules.
4. Do not infer regulatory requirements that are not
   explicitly supported by the evidence.
5. If the evidence is insufficient, say:
   "The retrieved RBI evidence is insufficient to answer
   this question."
6. Mention the relevant RBI section.
7. Mention the source page numbers.
8. Keep the answer concise and clear.

RBI POLICY EVIDENCE
===================

{evidence_text}


USER QUESTION
=============

{question}


RESPONSE FORMAT
===============

Answer:
<answer based only on the retrieved evidence>

RBI Evidence:
<relevant section>

Source Pages:
<page numbers>

Confidence:
HIGH / MEDIUM / LOW
"""

        # -------------------------------------------------
        # 4. Call WatsonX Granite
        # -------------------------------------------------

        logger.info("Sending RBI evidence to WatsonX LLM...")

        try:

            response = self.model.chat(
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )

        except Exception as e:

            logger.error(
                f"WatsonX model inference failed: {e}"
            )

            return "Unable to generate the policy answer."

        # -------------------------------------------------
        # 5. Extract response
        # -------------------------------------------------

        try:

            answer = (
                response["choices"][0]
                ["message"]["content"]
                .strip()
            )

        except (KeyError, IndexError, TypeError) as e:

            logger.error(
                f"Unexpected WatsonX response format: {e}"
            )

            return "Unable to parse the policy answer."

        return answer


# ---------------------------------------------------------
# Simple CLI Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("\n========================================")
    print("      RBI KYC POLICY ASSISTANT")
    print("========================================")

    answerer = RBIPolicyAnswerer()

    question = input(
        "\nAsk an RBI KYC policy question: "
    ).strip()

    if not question:

        print("\nPlease enter a question.")

    else:

        print(
            "\n===== RBI POLICY ANSWER =====\n"
        )

        answer = answerer.answer(question)

        print(answer)

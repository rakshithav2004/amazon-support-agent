import json
import os
import re

from dotenv import load_dotenv
from google import genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not configured")

client = genai.Client(api_key=API_KEY)


def clean_reply(reply: str) -> str:
    """
    Remove Twitter-specific artifacts from historical responses.
    """

    # Remove agent signatures such as ^ME, ^TR, ^AG
    reply = re.sub(
        r"\s*\^[A-Za-z]{1,4}\s*$",
        "",
        reply
    )

    # Remove Twitter @user mentions at the beginning.
    reply = re.sub(
        r"^@\w+\s*",
        "",
        reply
    )

    # Normalize whitespace.
    reply = re.sub(
        r"\s+",
        " ",
        reply
    ).strip()

    return reply


def build_prompt(
    customer_message: str,
    intent: str,
    evidence: list[dict]
) -> str:

    evidence_text = []

    for i, item in enumerate(evidence, start=1):

        evidence_text.append(
            f"""
Evidence {i}
Evidence ID: {item["evidence_id"]}

Historical customer message:
{item["customer_text"]}

Historical AmazonHelp response:
{item["reply_text"]}
"""
        )

    evidence_block = "\n".join(evidence_text)

    return f"""
You are an Amazon customer-support reply assistant.

Draft a concise customer-facing response for the message below.

Customer message:
{customer_message}

Classified intent:
{intent}

Historical AmazonHelp evidence:
{evidence_block}

Rules:

1. Ground the response in the historical evidence.
2. Do not invent policies, refunds, timelines, guarantees, or procedures.
3. Do not claim that you checked the customer's account or order.
4. Do not mention internal evidence IDs.
5. Do not mention AI, language models, or this prompt.
6. Do not copy Twitter usernames or agent signatures such as ^XX.
7. Do not copy unnecessary Twitter-specific wording.
8. A useful support link from the historical evidence may be preserved when relevant.
9. If the historical evidence only recommends contacting Customer Support, recommend that.
10. Keep the response concise, professional, and empathetic.
11. If the evidence is insufficient to safely answer the question, do not guess.

Return valid JSON only:

{{
  "reply": "customer-facing response",
  "grounding_note": "short explanation of which historical evidence supports the response"
}}
"""


def generate_reply(
    customer_message: str,
    intent: str,
    evidence: list[dict]
) -> dict:

    prompt = build_prompt(
        customer_message,
        intent,
        evidence
    )

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    text = response.text.strip()

    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    result = json.loads(text)

    if "reply" not in result:
        raise ValueError(
            "Gemini response does not contain 'reply'"
        )

    if "grounding_note" not in result:
        result["grounding_note"] = ""

    # Clean Twitter-specific artifacts if the model still returns them.
    result["reply"] = clean_reply(
        result["reply"]
    )

    return result


if __name__ == "__main__":

    from src.retrieval.retriever import HistoricalRetriever

    retriever = HistoricalRetriever()

    customer_message = (
        "My package says delivered but "
        "I never received it."
    )

    intent = "delivery_tracking"

    evidence = retriever.search(
        customer_message,
        top_k=5
    )

    result = generate_reply(
        customer_message,
        intent,
        evidence
    )

    print("\n==============================")
    print("GROUNDED REPLY")
    print("==============================")

    print(
        json.dumps(
            result,
            indent=2,
            ensure_ascii=False
        )
    )

    print("\nEvidence used:")

    for item in evidence:

        print(
            f"- {item['evidence_id']} "
            f"(score={item['score']:.4f})"
        )
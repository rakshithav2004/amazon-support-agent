import os
import json

from dotenv import load_dotenv
from google import genai

from src.intents.taxonomy import INTENTS


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not configured")

client = genai.Client(api_key=API_KEY)


def build_prompt(message: str) -> str:

    intent_text = "\n".join(
        [
            f"- {name}: {details['description']}"
            for name, details in INTENTS.items()
        ]
    )

    return f"""
You are a customer support intent classifier.

Classify the customer's message into exactly ONE of these intents:

{intent_text}

Customer message:
"{message}"

Rules:
1. Choose exactly one intent from the list.
2. Base the decision on the customer's main problem.
3. Do not choose an intent only because a keyword appears.
4. Return valid JSON only.
5. The intent must exactly match one of the provided intent names.

Return:
{{
  "intent": "intent_name",
  "reason": "short explanation"
}}
"""


def classify(message: str) -> dict:

    prompt = build_prompt(message)

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    text = response.text.strip()

    # Handle accidental markdown code fences
    if text.startswith("```"):
        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    result = json.loads(text)

    if result["intent"] not in INTENTS:
        raise ValueError(
            f"Invalid intent returned by Gemini: {result['intent']}"
        )

    return result


if __name__ == "__main__":

    test_message = (
        "My package was supposed to arrive yesterday "
        "but I still haven't received it."
    )

    result = classify(test_message)

    print(json.dumps(result, indent=2, ensure_ascii=False))
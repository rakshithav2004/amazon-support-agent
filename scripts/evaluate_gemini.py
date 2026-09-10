import json
import os
import time

from sklearn.metrics import accuracy_score, classification_report
from google import genai
from dotenv import load_dotenv
from src.intents.taxonomy import INTENTS

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")

INPUT_FILE = "data/golden/golden_200.jsonl"
OUTPUT_FILE = "data/golden/gemini_predictions.jsonl"

BATCH_SIZE = 10
RATE_LIMIT_WAIT_SECONDS = 60

if not API_KEY:
    raise ValueError("GEMINI_API_KEY is not configured")

client = genai.Client(api_key=API_KEY)

def load_examples():
    rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_existing_predictions():

    if not os.path.exists(OUTPUT_FILE):
        return {}

    predictions = {}

    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            predictions[str(row["tweet_id"])] = row

    return predictions


# --------------------------------------------------
# Build intent list
# --------------------------------------------------

def build_intent_text():

    lines = []

    for name, details in INTENTS.items():

        lines.append(
            f"- {name}: {details['description']}"
        )

    return "\n".join(lines)


# --------------------------------------------------
# Batch classification
# --------------------------------------------------

def classify_batch(examples):

    intent_text = build_intent_text()

    messages = []

    for index, example in enumerate(examples, start=1):

        messages.append(
            f"""
Example {index}

Tweet ID:
{example["tweet_id"]}

Customer message:
{example["text"]}
"""
        )

    prompt = f"""
You are a customer support intent classification system.

Classify every customer message into exactly ONE of these intents:

{intent_text}

Rules:

1. Choose exactly one intent for every example.
2. Base the decision on the customer's MAIN problem.
3. Do not choose an intent only because a keyword appears.
4. Use only the provided intent names.
5. Return valid JSON only.
6. Return exactly one result for every example.
7. Keep the same order as the examples.
8. Use the tweet_id provided for each example.

Return this exact JSON structure:

{{
  "predictions": [
    {{
      "tweet_id": 123,
      "intent": "delivery_tracking",
      "reason": "Short explanation"
    }}
  ]
}}

Customer examples:

{''.join(messages)}
"""

    response = client.models.generate_content(
        model=MODEL,
        contents=prompt
    )

    text = response.text.strip()

    # Remove markdown code fences if Gemini adds them.
    if text.startswith("```"):

        text = text.replace("```json", "")
        text = text.replace("```", "")
        text = text.strip()

    result = json.loads(text)

    predictions = result["predictions"]

    if len(predictions) != len(examples):

        raise ValueError(
            f"Expected {len(examples)} predictions "
            f"but received {len(predictions)}"
        )

    return predictions

examples = load_examples()
existing = load_existing_predictions()

print(f"Total examples: {len(examples)}")
print(f"Already evaluated: {len(existing)}")

remaining = [
    example
    for example in examples
    if str(example["tweet_id"]) not in existing
]

print(f"Remaining: {len(remaining)}")


for start in range(0, len(remaining), BATCH_SIZE):

    batch = remaining[start:start + BATCH_SIZE]

    batch_number = start // BATCH_SIZE + 1
    total_batches = (
        len(remaining) + BATCH_SIZE - 1
    ) // BATCH_SIZE

    print(
        f"\nBatch {batch_number}/{total_batches}"
    )

    print(
        f"Evaluating {len(batch)} examples..."
    )

    while True:

        try:

            predictions = classify_batch(batch)

            prediction_by_id = {
                str(item["tweet_id"]): item
                for item in predictions
            }

            # Validate IDs.
            for example in batch:

                tweet_id = str(example["tweet_id"])

                if tweet_id not in prediction_by_id:

                    raise ValueError(
                        f"Missing prediction for tweet {tweet_id}"
                    )

            # Save every prediction.
            with open(
                OUTPUT_FILE,
                "a",
                encoding="utf-8"
            ) as f:

                for example in batch:

                    tweet_id = str(example["tweet_id"])

                    prediction = prediction_by_id[tweet_id]

                    predicted_intent = prediction["intent"]

                    if predicted_intent not in INTENTS:

                        raise ValueError(
                            f"Invalid intent returned by Gemini: "
                            f"{predicted_intent}"
                        )

                    row = {
                        "tweet_id": example["tweet_id"],
                        "text": example["text"],
                        "gold_intent": example["gold_intent"],
                        "predicted_intent": predicted_intent,
                        "reason": prediction.get(
                            "reason",
                            ""
                        )
                    }

                    f.write(
                        json.dumps(
                            row,
                            ensure_ascii=False
                        ) + "\n"
                    )

                    existing[tweet_id] = row

            print(
                f"Successfully completed "
                f"{len(batch)} examples."
            )

            break

        except Exception as e:

            error_text = str(e)

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                print(
                    "\nRate limit reached."
                )

                print(
                    f"Waiting "
                    f"{RATE_LIMIT_WAIT_SECONDS} seconds..."
                )

                time.sleep(
                    RATE_LIMIT_WAIT_SECONDS
                )

                continue

            print(
                f"\nERROR: {e}"
            )

            print(
                "Stopping. Completed predictions "
                "have been preserved."
            )

            raise

results = [
    existing[str(example["tweet_id"])]
    for example in examples
    if str(example["tweet_id"]) in existing
]


if not results:

    print(
        "\nNo predictions completed yet."
    )

    raise SystemExit


y_true = [
    row["gold_intent"]
    for row in results
]

y_pred = [
    row["predicted_intent"]
    for row in results
]


accuracy = accuracy_score(
    y_true,
    y_pred
)

print("\n==============================")
print("GEMINI RESULTS")
print("==============================")

print(
    f"Completed: "
    f"{len(results)}/{len(examples)}"
)

print(
    f"Accuracy: {accuracy:.4f}"
)

print("\nClassification Report:")

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0
    )
)

print(
    f"\nPredictions saved to: "
    f"{OUTPUT_FILE}"
)
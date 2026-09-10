import json
import os
import time

from sklearn.metrics import accuracy_score, classification_report

from src.intents.gemini_classifier import classify


INPUT_FILE = "data/golden/golden_200.jsonl"
OUTPUT_FILE = "data/golden/gemini_predictions.jsonl"

# Wait between normal API requests.
# Gemini free tier currently allows only a limited number
# of requests per minute, so we use a conservative delay.
DELAY_SECONDS = 5

# How long to wait after a rate-limit error.
RATE_LIMIT_WAIT_SECONDS = 60


def load_examples():
    rows = []

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
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


examples = load_examples()
existing = load_existing_predictions()

print(f"Total examples: {len(examples)}")
print(f"Already evaluated: {len(existing)}")

for index, example in enumerate(examples, start=1):

    tweet_id = str(example["tweet_id"])

    # Skip examples that were already successfully evaluated.
    if tweet_id in existing:
        continue

    print(f"\n[{index}/{len(examples)}] Evaluating tweet {tweet_id}")

    while True:

        try:

            result = classify(example["text"])

            row = {
                "tweet_id": example["tweet_id"],
                "text": example["text"],
                "gold_intent": example["intent"],
                "predicted_intent": result["intent"],
                "reason": result.get("reason", "")
            }

            # Save immediately after every successful prediction.
            with open(OUTPUT_FILE, "a", encoding="utf-8") as f:

                f.write(
                    json.dumps(
                        row,
                        ensure_ascii=False
                    ) + "\n"
                )

            existing[tweet_id] = row

            print(f"Gold:      {example['intent']}")
            print(f"Predicted: {result['intent']}")

            # Wait before the next request.
            time.sleep(DELAY_SECONDS)

            break

        except Exception as e:

            error_text = str(e)

            # Gemini free-tier rate limit.
            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:

                print("\nRate limit reached.")
                print(
                    f"Waiting {RATE_LIMIT_WAIT_SECONDS} seconds "
                    "before retrying..."
                )

                time.sleep(RATE_LIMIT_WAIT_SECONDS)

                # Retry the SAME example.
                continue

            # Any other error.
            print(f"\nERROR: {e}")
            print("Stopping so completed results are preserved.")

            break

    # Stop the outer loop if the current example failed
    # for a reason other than rate limiting.
    if tweet_id not in existing:
        break


# --------------------------------------------------
# Calculate metrics for completed predictions
# --------------------------------------------------

results = list(existing.values())

if not results:

    print("\nNo predictions completed yet.")
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
    f"Completed: {len(results)}/{len(examples)}"
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
    f"\nPredictions saved to: {OUTPUT_FILE}"
)
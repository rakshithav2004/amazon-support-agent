import json
import pandas as pd
import os


INPUT_FILE = "data/golden/golden_200.jsonl"
OUTPUT_FILE = "data/golden/golden_review.csv"


rows = []

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    for line in f:
        if line.strip():
            rows.append(json.loads(line))


df = pd.DataFrame(rows)

# Create a separate column for the human-verified label.
df["gold_intent"] = ""

# Keep only the columns needed for manual review.
review_df = df[
    [
        "tweet_id",
        "text",
        "intent",
        "gold_intent"
    ]
].copy()

review_df = review_df.rename(
    columns={
        "intent": "suggested_intent"
    }
)

os.makedirs("data/golden", exist_ok=True)

review_df.to_csv(
    OUTPUT_FILE,
    index=False,
    encoding="utf-8-sig"
)

print("Golden review file created.")
print(f"Examples: {len(review_df)}")
print(f"Saved to: {OUTPUT_FILE}")
import csv
import json
import os


INPUT_FILE = "data/golden/golden_review.csv"
OUTPUT_FILE = "data/golden/golden_200.jsonl"


print("Reading golden review CSV...")

rows = []

with open(
    INPUT_FILE,
    "r",
    encoding="utf-8",
    newline=""
) as f:

    reader = csv.reader(f)

    header = next(reader)

    print("Original header:")
    print(header)

    for line_number, row in enumerate(reader, start=2):

        if len(row) < 4:
            print(
                f"Skipping invalid row {line_number}: "
                f"only {len(row)} fields"
            )
            continue

        # First field
        tweet_id = row[0]

        # Last two fields
        suggested_intent = row[-2]
        gold_intent = row[-1]

        # Everything between them belongs to tweet text
        text = ",".join(row[1:-2])

        rows.append({
            "tweet_id": int(tweet_id),
            "text": text,
            "suggested_intent": suggested_intent.strip(),
            "gold_intent": gold_intent.strip()
        })


print(f"\nLoaded rows: {len(rows)}")


# Check for missing gold labels
missing_labels = [
    row for row in rows
    if not row["gold_intent"]
]

if missing_labels:

    print(
        f"\nWARNING: {len(missing_labels)} "
        "rows have no gold_intent."
    )

else:

    print("All rows have gold_intent.")


# Save JSONL
os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    for row in rows:

        f.write(
            json.dumps(
                row,
                ensure_ascii=False
            ) + "\n"
        )


print(f"\nSaved: {OUTPUT_FILE}")


# Show distribution
from collections import Counter

distribution = Counter(
    row["gold_intent"]
    for row in rows
)

print("\nGold intent distribution:")

for intent, count in distribution.most_common():

    print(
        f"{intent}: {count}"
    )


print("\nDone!")
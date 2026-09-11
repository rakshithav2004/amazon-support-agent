import csv
import json
import os


INPUT_FILE = "data/golden/reply_eval_60.jsonl"
OUTPUT_FILE = "data/golden/human_ratings_60.csv"


def main():

    rows = []

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "tweet_id",
            "customer_text",
            "generated_reply",
            "grounding",
            "correctness",
            "helpfulness",
            "unsupported_claims"
        ])

        for row in rows:

            writer.writerow([
                row["tweet_id"],
                row["customer_text"],
                row["reply"],
                "",
                "",
                "",
                ""
            ])

    print(
        f"Created: {OUTPUT_FILE}"
    )

    print(
        f"Examples: {len(rows)}"
    )


if __name__ == "__main__":
    main()
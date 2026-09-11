import csv
import json
import os


INPUT_FILE = "data/golden/reply_eval_60.jsonl"
OUTPUT_FILE = "data/golden/human_ratings_60.csv"


def load_examples():
    rows = []

    if not os.path.exists(INPUT_FILE):
        raise FileNotFoundError(
            f"Input file not found: {INPUT_FILE}"
        )

    with open(
        INPUT_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(json.loads(line))

    return rows


def load_existing_ratings():

    existing = {}

    if not os.path.exists(OUTPUT_FILE):
        return existing

    with open(
        OUTPUT_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            grounding = row.get(
                "grounding",
                ""
            ).strip()

            if grounding:

                existing[
                    str(row["tweet_id"])
                ] = row

    return existing


def get_scores():

    while True:

        value = input(
            "\nEnter scores "
            "(Grounding Correctness Helpfulness Unsupported): "
        ).strip()

        parts = value.split()

        if len(parts) != 4:

            print(
                "Please enter exactly 4 numbers."
            )
            print(
                "Example: 5 4 4 5"
            )
            continue

        if not all(
            part in {"1", "2", "3", "4", "5"}
            for part in parts
        ):

            print(
                "Each score must be between 1 and 5."
            )
            continue

        return [int(part) for part in parts]


def save_ratings(existing):

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8",
        newline=""
    ) as f:

        fieldnames = [
            "tweet_id",
            "customer_text",
            "generated_reply",
            "grounding",
            "correctness",
            "helpfulness",
            "unsupported_claims"
        ]

        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for row in existing.values():
            writer.writerow(row)


def main():

    examples = load_examples()

    if len(examples) != 60:

        raise ValueError(
            f"Expected 60 examples, "
            f"found {len(examples)}"
        )

    existing = load_existing_ratings()

    print("\n==========================================")
    print("HUMAN EVALUATION - 60 EXAMPLES")
    print("==========================================")

    print("\nEnter 4 scores on one line:")

    print(
        "Grounding Correctness Helpfulness Unsupported"
    )

    print(
        "Example: 5 4 4 5"
    )

    print("\nRating guide:")

    print(
        "1 = poor"
    )

    print(
        "3 = acceptable"
    )

    print(
        "5 = excellent"
    )

    print(
        "\nRatings are saved after every example."
    )

    print(
        "You can press Ctrl+C and resume later."
    )

    for position, example in enumerate(
        examples,
        start=1
    ):

        tweet_id = str(
            example["tweet_id"]
        )

        if tweet_id in existing:

            print(
                f"\n[{position}/60] "
                f"Tweet {tweet_id} already rated."
            )

            continue

        print("\n==========================================")

        print(
            f"[{position}/60] Tweet ID: {tweet_id}"
        )

        print("==========================================")

        print("\nCUSTOMER:")
        print(
            example["customer_text"]
        )

        print("\nGENERATED REPLY:")
        print(
            example["reply"]
        )

        grounding, correctness, helpfulness, unsupported = (
            get_scores()
        )

        existing[tweet_id] = {
            "tweet_id": tweet_id,
            "customer_text": example[
                "customer_text"
            ],
            "generated_reply": example[
                "reply"
            ],
            "grounding": grounding,
            "correctness": correctness,
            "helpfulness": helpfulness,
            "unsupported_claims": unsupported
        }

        save_ratings(existing)

        print(
            f"\nSaved. Progress: "
            f"{len(existing)}/60"
        )

    print("\n==========================================")
    print("HUMAN EVALUATION COMPLETE")
    print("==========================================")

    print(
        f"Rated examples: {len(existing)}/60"
    )

    print(
        f"File: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
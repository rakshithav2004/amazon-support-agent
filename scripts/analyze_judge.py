import json
import statistics


INPUT_FILE = "data/golden/judge_scores_30.jsonl"


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
                rows.append(json.loads(line))

    print("\n==============================")
    print("JUDGE SUMMARY")
    print("==============================")

    print(f"Examples: {len(rows)}")

    metrics = [
        "grounding",
        "correctness",
        "helpfulness",
        "unsupported_claims"
    ]

    for metric in metrics:

        values = [
            row[metric]
            for row in rows
        ]

        print(
            f"{metric.replace('_', ' ').title()}: "
            f"{statistics.mean(values):.2f}/5"
        )

    # Overall average across all four dimensions.
    overall_values = []

    for row in rows:

        score = (
            row["grounding"]
            + row["correctness"]
            + row["helpfulness"]
            + row["unsupported_claims"]
        ) / 4

        overall_values.append(score)

    print(
        f"Overall average: "
        f"{statistics.mean(overall_values):.2f}/5"
    )

    # Show weaker responses.
    print("\n==============================")
    print("LOW-SCORING EXAMPLES")
    print("==============================")

    sorted_rows = sorted(
        rows,
        key=lambda row: (
            row["grounding"]
            + row["correctness"]
            + row["helpfulness"]
            + row["unsupported_claims"]
        ) / 4
    )

    for row in sorted_rows[:5]:

        overall = (
            row["grounding"]
            + row["correctness"]
            + row["helpfulness"]
            + row["unsupported_claims"]
        ) / 4

        print("\nTweet:", row["tweet_id"])
        print(f"Overall: {overall:.2f}/5")
        print("Customer:", row["customer_text"])
        print("Reply:", row["generated_reply"])
        print("Reason:", row["overall_reason"])


if __name__ == "__main__":
    main()
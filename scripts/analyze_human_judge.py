import csv
import json
import os

from scipy.stats import spearmanr


HUMAN_FILE = "data/golden/human_ratings_60.csv"
JUDGE_FILE = "data/golden/judge_scores_60.jsonl"


METRICS = [
    "grounding",
    "correctness",
    "helpfulness",
    "unsupported_claims"
]


def load_human():

    rows = {}

    with open(
        HUMAN_FILE,
        "r",
        encoding="utf-8",
        newline=""
    ) as f:

        reader = csv.DictReader(f)

        for row in reader:

            rows[str(row["tweet_id"])] = {
                metric: int(row[metric])
                for metric in METRICS
            }

    return rows


def load_judge():

    rows = {}

    with open(
        JUDGE_FILE,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if not line:
                continue

            row = json.loads(line)

            rows[str(row["tweet_id"])] = {
                metric: int(row[metric])
                for metric in METRICS
            }

    return rows


def main():

    human = load_human()
    judge = load_judge()

    ids = sorted(
        set(human) & set(judge)
    )

    print("\n==============================")
    print("HUMAN vs LLM JUDGE ANALYSIS")
    print("==============================")

    print(
        f"Common examples: {len(ids)}"
    )

    all_human = []
    all_judge = []

    for metric in METRICS:

        h = [
            human[i][metric]
            for i in ids
        ]

        j = [
            judge[i][metric]
            for i in ids
        ]

        exact = sum(
            a == b
            for a, b in zip(h, j)
        ) / len(ids)

        within_one = sum(
            abs(a - b) <= 1
            for a, b in zip(h, j)
        ) / len(ids)

        correlation, p_value = spearmanr(
            h,
            j
        )

        all_human.extend(h)
        all_judge.extend(j)

        print(
            f"\n{metric.replace('_', ' ').title()}"
        )

        print(
            f"Exact agreement: "
            f"{exact:.3f}"
        )

        print(
            f"Agreement within ±1: "
            f"{within_one:.3f}"
        )

        print(
            f"Spearman correlation: "
            f"{correlation:.3f}"
        )

        print(
            f"Spearman p-value: "
            f"{p_value:.4f}"
        )

    overall_exact = sum(
        a == b
        for a, b in zip(
            all_human,
            all_judge
        )
    ) / len(all_human)

    overall_within_one = sum(
        abs(a - b) <= 1
        for a, b in zip(
            all_human,
            all_judge
        )
    ) / len(all_human)

    overall_corr, overall_p = spearmanr(
        all_human,
        all_judge
    )

    print(
        "\n=============================="
    )

    print(
        f"Overall exact agreement: "
        f"{overall_exact:.3f}"
    )

    print(
        f"Overall agreement within ±1: "
        f"{overall_within_one:.3f}"
    )

    print(
        f"Overall Spearman correlation: "
        f"{overall_corr:.3f}"
    )

    print(
        f"Overall Spearman p-value: "
        f"{overall_p:.4f}"
    )


if __name__ == "__main__":
    main()
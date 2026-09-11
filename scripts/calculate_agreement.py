import csv
import json
import os

from sklearn.metrics import cohen_kappa_score


HUMAN_FILE = "data/golden/human_ratings_60.csv"
JUDGE_FILE = "data/golden/judge_scores_60.jsonl"


def load_human_ratings():

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
                "grounding": int(row["grounding"]),
                "correctness": int(row["correctness"]),
                "helpfulness": int(row["helpfulness"]),
                "unsupported_claims": int(
                    row["unsupported_claims"]
                )
            }

    return rows


def load_judge_ratings():

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
                "grounding": row["grounding"],
                "correctness": row["correctness"],
                "helpfulness": row["helpfulness"],
                "unsupported_claims": row[
                    "unsupported_claims"
                ]
            }

    return rows


def main():

    human = load_human_ratings()
    judge = load_judge_ratings()

    common_ids = sorted(
        set(human.keys()) & set(judge.keys())
    )

    print("\n==============================")
    print("HUMAN vs LLM JUDGE AGREEMENT")
    print("==============================")

    print(
        f"Human ratings: {len(human)}"
    )

    print(
        f"Judge ratings: {len(judge)}"
    )

    print(
        f"Common examples: {len(common_ids)}"
    )

    metrics = [
        "grounding",
        "correctness",
        "helpfulness",
        "unsupported_claims"
    ]

    kappas = []

    for metric in metrics:

        human_scores = [
            human[tweet_id][metric]
            for tweet_id in common_ids
        ]

        judge_scores = [
            judge[tweet_id][metric]
            for tweet_id in common_ids
        ]

        kappa = cohen_kappa_score(
            human_scores,
            judge_scores,
            weights="quadratic"
        )

        kappas.append(kappa)

        print(
            f"\n{metric.replace('_', ' ').title()}"
        )

        print(
            f"Weighted Cohen's kappa: "
            f"{kappa:.3f}"
        )

    # Overall average kappa.
    average_kappa = sum(kappas) / len(kappas)

    print(
        "\n=============================="
    )

    print(
        f"Average weighted kappa: "
        f"{average_kappa:.3f}"
    )

    print(
        "=============================="
    )


if __name__ == "__main__":
    main()
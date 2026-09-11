import json
import os


JUDGE_FILE = "data/golden/judge_scores_60.jsonl"
REPLY_FILE = "data/golden/reply_eval_60.jsonl"

OUTPUT_FILE = "data/golden/failure_analysis.jsonl"


def load_jsonl(path):

    rows = []

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:

        for line in f:

            line = line.strip()

            if line:
                rows.append(
                    json.loads(line)
                )

    return rows


def main():

    judge_rows = load_jsonl(
        JUDGE_FILE
    )

    reply_rows = load_jsonl(
        REPLY_FILE
    )

    replies = {
        str(row["tweet_id"]): row
        for row in reply_rows
    }

    results = []

    for row in judge_rows:

        tweet_id = str(
            row["tweet_id"]
        )

        overall = (
            row["grounding"]
            + row["correctness"]
            + row["helpfulness"]
            + row["unsupported_claims"]
        ) / 4

        reply = replies.get(
            tweet_id,
            {}
        )

        results.append(
            {
                "tweet_id": tweet_id,
                "overall": overall,
                "grounding": row["grounding"],
                "correctness": row["correctness"],
                "helpfulness": row["helpfulness"],
                "unsupported_claims": row[
                    "unsupported_claims"
                ],
                "customer_text": row[
                    "customer_text"
                ],
                "generated_reply": row[
                    "generated_reply"
                ],
                "reason": row.get(
                    "overall_reason",
                    ""
                ),
                "gold_intent": reply.get(
                    "gold_intent",
                    ""
                )
            }
        )

    results.sort(
        key=lambda x: x["overall"]
    )

    worst = results[:10]

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for row in worst:

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

    print("\n==============================")
    print("TOP 10 LOWEST-SCORING EXAMPLES")
    print("==============================")

    for index, row in enumerate(
        worst,
        start=1
    ):

        print(
            f"\n#{index}"
        )

        print(
            f"Tweet ID: {row['tweet_id']}"
        )

        print(
            f"Intent: {row['gold_intent']}"
        )

        print(
            f"Overall: {row['overall']:.2f}/5"
        )

        print(
            f"Grounding: "
            f"{row['grounding']}/5"
        )

        print(
            f"Correctness: "
            f"{row['correctness']}/5"
        )

        print(
            f"Helpfulness: "
            f"{row['helpfulness']}/5"
        )

        print(
            f"Unsupported claims: "
            f"{row['unsupported_claims']}/5"
        )

        print(
            f"\nCustomer:\n"
            f"{row['customer_text']}"
        )

        print(
            f"\nGenerated reply:\n"
            f"{row['generated_reply']}"
        )

        print(
            f"\nJudge reasoning:\n"
            f"{row['reason']}"
        )

        print(
            "\n----------------------------------"
        )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
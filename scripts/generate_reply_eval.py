import json
import os
import time

from src.retrieval.retriever import HistoricalRetriever
from src.generation.reply_generator import generate_reply


INPUT_FILE = "data/golden/golden_200.jsonl"
OUTPUT_FILE = "data/golden/reply_eval_60.jsonl"

SAMPLE_SIZE = 60
TOP_K = 5

RATE_LIMIT_WAIT_SECONDS = 20


def load_jsonl(path):

    rows = []

    if not os.path.exists(path):
        return rows

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

    examples = load_jsonl(
        INPUT_FILE
    )[:SAMPLE_SIZE]

    existing_rows = load_jsonl(
        OUTPUT_FILE
    )

    existing_ids = {
        str(row["tweet_id"])
        for row in existing_rows
    }

    remaining = [
        example
        for example in examples
        if str(example["tweet_id"])
        not in existing_ids
    ]

    print(
        f"Target examples: {len(examples)}"
    )

    print(
        f"Already generated: {len(existing_rows)}"
    )

    print(
        f"Remaining: {len(remaining)}"
    )

    if not remaining:

        print(
            "\nAll 60 replies already exist."
        )

        return

    retriever = HistoricalRetriever()

    with open(
        OUTPUT_FILE,
        "a",
        encoding="utf-8"
    ) as f:

        for index, example in enumerate(
            remaining,
            start=1
        ):

            print(
                f"\n[{index}/{len(remaining)}] "
                f"Tweet {example['tweet_id']}"
            )

            evidence = retriever.search(
                example["text"],
                top_k=TOP_K
            )

            # Retry the same example after rate limiting.
            while True:

                try:

                    reply_result = generate_reply(
                        customer_message=example["text"],
                        intent=example["gold_intent"],
                        evidence=evidence
                    )

                    break

                except Exception as e:

                    error_text = str(e)

                    if (
                        "429" in error_text
                        or "RESOURCE_EXHAUSTED" in error_text
                    ):

                        print(
                            "\nGemini rate limit reached."
                        )

                        print(
                            f"Waiting "
                            f"{RATE_LIMIT_WAIT_SECONDS} "
                            "seconds before retrying..."
                        )

                        time.sleep(
                            RATE_LIMIT_WAIT_SECONDS
                        )

                        continue

                    raise

            row = {
                "tweet_id": example["tweet_id"],
                "customer_text": example["text"],
                "gold_intent": example["gold_intent"],
                "reply": reply_result["reply"],
                "grounding_note": reply_result.get(
                    "grounding_note",
                    ""
                ),
                "evidence": [
                    {
                        "evidence_id": item["evidence_id"],
                        "customer_text": item["customer_text"],
                        "reply_text": item["reply_text"],
                        "score": item["score"]
                    }
                    for item in evidence
                ]
            }

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

            f.flush()

            print(
                f"Generated reply: "
                f"{reply_result['reply']}"
            )

            # Small pause between successful requests.
            time.sleep(5)

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
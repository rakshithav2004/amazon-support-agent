import json
import os

from src.retrieval.retriever import HistoricalRetriever
from src.generation.reply_generator import generate_reply


INPUT_FILE = "data/golden/golden_200.jsonl"
OUTPUT_FILE = "data/golden/reply_eval_30.jsonl"

SAMPLE_SIZE = 30
TOP_K = 5


def load_golden():

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

    return rows


def main():

    examples = load_golden()

    # Use a fixed sample so the experiment is reproducible.
    examples = examples[:SAMPLE_SIZE]

    print(
        f"Generating replies for "
        f"{len(examples)} examples..."
    )

    retriever = HistoricalRetriever()

    with open(
        OUTPUT_FILE,
        "w",
        encoding="utf-8"
    ) as f:

        for index, example in enumerate(
            examples,
            start=1
        ):

            print(
                f"\n[{index}/{len(examples)}] "
                f"Tweet {example['tweet_id']}"
            )

            evidence = retriever.search(
                example["text"],
                top_k=TOP_K
            )

            reply_result = generate_reply(
                customer_message=example["text"],
                intent=example["gold_intent"],
                evidence=evidence
            )

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

            print(
                f"Generated reply: "
                f"{reply_result['reply']}"
            )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
import json
import os
import time

from dotenv import load_dotenv
from google import genai


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite"
)

INPUT_FILE = "data/golden/reply_eval_60.jsonl"
OUTPUT_FILE = "data/golden/judge_scores_60.jsonl"

RATE_LIMIT_WAIT_SECONDS = 20


if not API_KEY:
    raise ValueError(
        "GEMINI_API_KEY is not configured"
    )


client = genai.Client(
    api_key=API_KEY
)


def build_prompt(example):

    evidence_text = []

    for item in example["evidence"]:

        evidence_text.append(
            f"""
Evidence ID: {item["evidence_id"]}

Historical customer:
{item["customer_text"]}

Historical AmazonHelp reply:
{item["reply_text"]}
"""
        )

    return f"""
You are evaluating an AI customer-support response.

Customer message:
{example["customer_text"]}

Intent:
{example["gold_intent"]}

Generated response:
{example["reply"]}

Historical evidence:
{''.join(evidence_text)}

Evaluate the generated response using these four criteria.

1. Grounding

5 = Fully supported by the historical evidence.
4 = Mostly supported with only minor interpretation.
3 = Partially supported.
2 = Mostly unsupported.
1 = Contradicts or ignores the evidence.

2. Correctness

5 = Correctly addresses the customer's actual issue.
4 = Mostly correct.
3 = Partially correct.
2 = Significant misunderstanding.
1 = Incorrect response.

3. Helpfulness

5 = Clear, concise, actionable and appropriate.
4 = Helpful with a minor weakness.
3 = Somewhat helpful.
2 = Limited usefulness.
1 = Not helpful.

4. Unsupported claims

5 = No unsupported claims.
4 = Very minor unsupported wording.
3 = Some unsupported content.
2 = Several unsupported claims.
1 = Major fabricated claims or guarantees.

Return valid JSON only:

{{
  "grounding": 1,
  "correctness": 1,
  "helpfulness": 1,
  "unsupported_claims": 1,
  "overall_reason": "short explanation"
}}
"""


def judge(example):

    prompt = build_prompt(
        example
    )

    while True:

        try:

            response = client.models.generate_content(
                model=MODEL,
                contents=prompt
            )

            text = response.text.strip()

            if text.startswith("```"):

                text = text.replace(
                    "```json",
                    ""
                )

                text = text.replace(
                    "```",
                    ""
                )

                text = text.strip()

            return json.loads(text)

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
    )

    existing = load_jsonl(
        OUTPUT_FILE
    )

    existing_ids = {
        str(row["tweet_id"])
        for row in existing
    }

    remaining = [
        example
        for example in examples
        if str(example["tweet_id"])
        not in existing_ids
    ]

    print(
        f"Total examples: {len(examples)}"
    )

    print(
        f"Already judged: {len(existing)}"
    )

    print(
        f"Remaining: {len(remaining)}"
    )

    if not remaining:

        print(
            "\nAll examples have already been judged."
        )

        return

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

            scores = judge(
                example
            )

            row = {
                "tweet_id": example["tweet_id"],
                "customer_text": example[
                    "customer_text"
                ],
                "generated_reply": example[
                    "reply"
                ],
                "grounding": scores[
                    "grounding"
                ],
                "correctness": scores[
                    "correctness"
                ],
                "helpfulness": scores[
                    "helpfulness"
                ],
                "unsupported_claims": scores[
                    "unsupported_claims"
                ],
                "overall_reason": scores.get(
                    "overall_reason",
                    ""
                )
            }

            f.write(
                json.dumps(
                    row,
                    ensure_ascii=False
                ) + "\n"
            )

            f.flush()

            print(
                f"Grounding: "
                f"{row['grounding']}"
            )

            print(
                f"Correctness: "
                f"{row['correctness']}"
            )

            print(
                f"Helpfulness: "
                f"{row['helpfulness']}"
            )

            print(
                f"Unsupported claims: "
                f"{row['unsupported_claims']}"
            )

    print(
        f"\nSaved to: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()
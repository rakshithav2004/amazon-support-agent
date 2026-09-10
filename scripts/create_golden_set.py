import pandas as pd
import json
import os
import re

INPUT_FILE = "data/processed/amazon_threads.csv"
OUTPUT_FILE = "data/golden/golden_200.jsonl"

df = pd.read_csv(INPUT_FILE, low_memory=False)

# Customer messages only
df = df[df["inbound"] == True].copy()
df["text"] = df["text"].fillna("").astype(str)

RULES = {
    "delivery_tracking": [
        "delivery", "delivered", "shipping", "shipment",
        "tracking", "package", "parcel", "arrive"
    ],
    "return_refund": [
        "refund", "refunded", "return", "returned",
        "money back"
    ],
    "payment_billing": [
        "payment", "charged", "charge", "billing",
        "invoice", "credit card", "debit card"
    ],
    "account_login": [
        "account", "login", "password", "sign in",
        "logged in", "hacked"
    ],
    "prime_membership": [
        "prime", "membership", "subscription"
    ],
    "digital_content": [
        "prime video", "kindle", "video", "movie",
        "music", "ebook", "digital"
    ],
    "cancellation": [
        "cancel", "cancellation", "cancelled"
    ],
    "product_issue": [
        "damaged", "broken", "defective",
        "wrong item", "wrong product"
    ],
    "order_issue": [
        "order", "ordered", "preorder", "purchase"
    ]
}


def matches(text, keywords):
    text = text.lower()

    return any(
        re.search(r"\b" + re.escape(keyword) + r"\b", text)
        for keyword in keywords
    )


records = []
used_tweet_ids = set()

# 20 candidates for each of the 9 specific intents
for intent, keywords in RULES.items():

    mask = df["text"].apply(
        lambda x: matches(x, keywords)
    )

    candidate_df = df[mask].copy()

    # Remove duplicates already selected
    candidate_df = candidate_df[
        ~candidate_df["tweet_id"].isin(used_tweet_ids)
    ]

    sample_size = min(20, len(candidate_df))

    sample = candidate_df.sample(
        n=sample_size,
        random_state=42
    )

    for _, row in sample.iterrows():

        tweet_id = int(row["tweet_id"])

        records.append({
            "tweet_id": tweet_id,
            "text": row["text"],
            "intent": intent,
            "label_source": "keyword_candidate"
        })

        used_tweet_ids.add(tweet_id)


# Add 20 random examples for OTHER
remaining_df = df[
    ~df["tweet_id"].isin(used_tweet_ids)
].copy()

other_sample = remaining_df.sample(
    n=20,
    random_state=123
)

for _, row in other_sample.iterrows():

    records.append({
        "tweet_id": int(row["tweet_id"]),
        "text": row["text"],
        "intent": "other",
        "label_source": "random_candidate"
    })


os.makedirs("data/golden", exist_ok=True)

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:

    for record in records:
        f.write(
            json.dumps(
                record,
                ensure_ascii=False
            ) + "\n"
        )

print("Golden-set candidate creation complete.")
print(f"Total candidates: {len(records)}")
print(f"Saved to: {OUTPUT_FILE}")
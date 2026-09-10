import pandas as pd
import re
from collections import Counter

FILE = "data/processed/amazon_threads.csv"

df = pd.read_csv(FILE)

print(f"Total customer messages: {len(df):,}")

# Only analyze customer messages
customers = df[df["inbound"] == True].copy()

print(f"Customer messages: {len(customers):,}")

# Basic statistics
customers["text_length"] = customers["text"].fillna("").str.len()

print("\nMessage length:")
print(customers["text_length"].describe())

# Look for common English keywords
text = customers["text"].fillna("").str.lower()

keywords = {
    "delivery": [
        "delivery", "delivered", "shipping", "shipment",
        "package", "parcel", "tracking", "arrive"
    ],
    "refund": [
        "refund", "refunded", "money back", "reimburse"
    ],
    "return": [
        "return", "send back", "returned"
    ],
    "order": [
        "order", "ordered", "purchase"
    ],
    "payment": [
        "payment", "charged", "charge", "billing",
        "credit card", "debit card"
    ],
    "account": [
        "account", "login", "password", "sign in"
    ],
    "prime": [
        "prime", "membership", "subscription"
    ],
    "video": [
        "prime video", "video", "movie", "episode", "stream"
    ],
    "product": [
        "product", "item", "device", "broken", "damaged"
    ],
    "cancel": [
        "cancel", "cancellation"
    ]
}

print("\nKeyword frequency:")
results = []

for category, words in keywords.items():
    count = 0

    for word in words:
        count += text.str.contains(
            re.escape(word),
            regex=True,
            na=False
        ).sum()

    results.append((category, count))

for category, count in sorted(results, key=lambda x: x[1], reverse=True):
    print(f"{category:12} {count:,}")

# Show random examples
print("\nSample customer messages:\n")

sample = customers.sample(
    min(30, len(customers)),
    random_state=42
)

for i, row in sample.iterrows():
    print(f"[{row['tweet_id']}] {row['text']}")
    print()
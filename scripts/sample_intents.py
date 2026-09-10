import pandas as pd

FILE = "data/processed/amazon_threads.csv"

df = pd.read_csv(
    FILE,
    low_memory=False
)

customers = df[df["inbound"] == True].copy()

# Keyword groups used only for exploration.
categories = {
    "delivery_tracking": [
        "delivery", "delivered", "shipping",
        "shipment", "tracking", "package",
        "parcel", "arrive"
    ],
    "return_refund": [
        "refund", "refunded", "return",
        "returned", "money back"
    ],
    "payment_billing": [
        "payment", "charged", "charge",
        "billing", "credit card", "debit card"
    ],
    "account_login": [
        "account", "login", "password",
        "sign in", "logged in"
    ],
    "prime_membership": [
        "prime", "membership", "subscription"
    ],
    "digital_content": [
        "prime video", "video", "kindle",
        "movie", "music", "episode"
    ],
    "cancellation": [
        "cancel", "cancellation"
    ],
    "product_issue": [
        "damaged", "broken", "defective",
        "wrong product", "wrong item"
    ],
    "order_issue": [
        "order", "ordered", "purchase"
    ]
}

text = customers["text"].fillna("").astype(str).str.lower()

for category, keywords in categories.items():

    mask = text.str.contains(
        "|".join(keywords),
        regex=True,
        na=False
    )

    examples = customers[mask].sample(
        min(8, mask.sum()),
        random_state=42
    )

    print("\n" + "=" * 80)
    print(category.upper())
    print("=" * 80)

    for _, row in examples.iterrows():
        print(f"[{row['tweet_id']}] {row['text']}")
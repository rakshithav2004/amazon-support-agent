import pandas as pd
import os


INPUT_FILE = "data/processed/amazon_retrieval_corpus.csv"
OUTPUT_FILE = "data/processed/amazon_retrieval_corpus_clean.csv"


print("Loading retrieval corpus...")

df = pd.read_csv(
    INPUT_FILE,
    low_memory=False
)

print(f"Original records: {len(df):,}")


# --------------------------------------------------
# Basic cleaning
# --------------------------------------------------

df["customer_text"] = (
    df["customer_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)

df["reply_text"] = (
    df["reply_text"]
    .fillna("")
    .astype(str)
    .str.strip()
)


# --------------------------------------------------
# Remove very short / generic replies
# --------------------------------------------------

generic_phrases = [
    "こちらこそありがとうございました",
    "ありがとうございました",
    "ご利用ありがとうございます",
    "お問い合わせありがとうございます",
    "よろしくお願いします",
    "thank you",
    "thanks",
    "you're welcome",
    "you are welcome"
]


def is_low_value_reply(text):

    normalized = text.lower().strip()

    # Too short to contain useful resolution information.
    if len(normalized) < 35:
        return True

    # Generic acknowledgements.
    for phrase in generic_phrases:

        if phrase.lower() in normalized:

            # Keep it only if it also contains meaningful support content.
            support_words = [
                "help",
                "order",
                "refund",
                "return",
                "delivery",
                "account",
                "payment",
                "cancel",
                "troubleshoot",
                "http",
                "https",
                "amazon"
            ]

            if not any(
                word in normalized
                for word in support_words
            ):
                return True

    return False


before_count = len(df)

df = df[
    ~df["reply_text"].apply(is_low_value_reply)
].copy()


# --------------------------------------------------
# Remove duplicate evidence
# --------------------------------------------------

df = df.drop_duplicates(
    subset=["customer_text", "reply_text"]
)


# --------------------------------------------------
# Save
# --------------------------------------------------

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n==============================")
print("CLEAN RETRIEVAL CORPUS")
print("==============================")

print(
    f"Before filtering: {before_count:,}"
)

print(
    f"After filtering:  {len(df):,}"
)

print(
    f"Removed:          {before_count - len(df):,}"
)

print(
    f"\nSaved to: {OUTPUT_FILE}"
)


print("\nSample evidence:")

print(
    df[
        [
            "evidence_id",
            "customer_text",
            "reply_text"
        ]
    ]
    .head(5)
    .to_string(index=False)
)
import json
from pathlib import Path


GOLDEN_FILE = Path(
    "data/golden/golden_200.jsonl"
)


def load_golden_set():
    rows = []

    if not GOLDEN_FILE.exists():
        raise FileNotFoundError(
            f"Golden set not found: {GOLDEN_FILE}"
        )

    with open(
        GOLDEN_FILE,
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


def get_golden_count():
    return len(load_golden_set())


if __name__ == "__main__":
    rows = load_golden_set()

    print(
        f"Golden examples: {len(rows)}"
    )
import json
import random
from pathlib import Path

from datasets import load_dataset


SEED = 42
EVAL_SIZE = 100

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

TRAIN_OUT = PROCESSED_DIR / "alpaca_train.jsonl"
EVAL_OUT = PROCESSED_DIR / "alpaca_eval.jsonl"


def normalize_example(example, idx):
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output_text = (example.get("output") or "").strip()

    if not instruction or not output_text:
        return None

    return {
        "id": f"alpaca_{idx:06d}",
        "instruction": instruction,
        "input": input_text,
        "output": output_text,
    }


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset("yahma/alpaca-cleaned", split="train")
    normalized = []

    for idx, ex in enumerate(dataset):
        row = normalize_example(ex, idx)
        if row is not None:
            normalized.append(row)

    random.shuffle(normalized)

    eval_rows = normalized[:EVAL_SIZE]
    train_rows = normalized[EVAL_SIZE:]

    write_jsonl(EVAL_OUT, eval_rows)
    write_jsonl(TRAIN_OUT, train_rows)

    print(f"\nTotal normalized examples: {len(normalized)}")
    print(f"Train examples written: {len(train_rows)} -> {TRAIN_OUT}")
    print(f"Eval examples written: {len(eval_rows)} -> {EVAL_OUT}")


if __name__ == "__main__":
    main()
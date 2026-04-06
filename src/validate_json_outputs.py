import json
from pathlib import Path


SEED_PATH = Path("data/raw/json_prompt_seed.json")


def main():
    if not SEED_PATH.exists():
        raise FileNotFoundError(f"Seed file not found: {SEED_PATH}")

    with open(SEED_PATH, "r", encoding="utf-8") as f:
        rows = json.load(f)

    if not isinstance(rows, list):
        raise ValueError("Seed file must contain a JSON list.")

    required_keys = {"id", "task_type", "instruction", "input", "expected_schema"}

    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i} is not a JSON object.")

        missing = required_keys - set(row.keys())
        if missing:
            raise ValueError(f"Row {i} is missing keys: {missing}")

        if not isinstance(row["expected_schema"], dict):
            raise ValueError(f"Row {i} expected_schema must be an object.")

    print(f"Validation passed: {len(rows)} seed prompts found in {SEED_PATH}")


if __name__ == "__main__":
    main()
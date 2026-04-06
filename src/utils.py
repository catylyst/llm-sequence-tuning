from pathlib import Path
import yaml


def load_yaml(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def format_example(example):
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()
    output_text = (example.get("output") or "").strip()

    if input_text:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
            f"{output_text}"
        )
    else:
        prompt = (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Response:\n"
            f"{output_text}"
        )

    return {"text": prompt}
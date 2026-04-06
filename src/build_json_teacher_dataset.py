import json
import random
from pathlib import Path


SEED = 42
TRAIN_RATIO = 0.8

SEED_PATH = Path("data/raw/json_prompt_seed.json")
TRAIN_OUT = Path("data/processed/json_teacher_train.jsonl")
EVAL_OUT = Path("data/processed/json_eval.jsonl")


def load_seed_prompts():
    with open(SEED_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_local_teacher_output(task_type, instruction, input_text):
    """
    Temporary local placeholder generator.
    Builds the full pipeline.
    Later will replace this function with a real teacher-model call.
    """

    if task_type == "json_extraction":
        if "Sarah Chen" in input_text:
            return {
                "person": "Sarah Chen",
                "organization": "OpenAI",
                "location": "San Francisco",
                "date": "2025-03-12"
            }
        if "Michael Torres" in input_text:
            return {
                "customer_name": "Michael Torres",
                "product": "wireless keyboards",
                "quantity": 3,
                "shipping_state": "Texas"
            }

    elif task_type == "schema_constrained_generation":
        if "student profile" in instruction.lower():
            return {
                "name": "Emily Carter",
                "age": 22,
                "major": "Computer Science",
                "gpa": 3.8,
                "enrolled": True
            }
        return {
            "customer_name": "James Patel",
            "party_size": 4,
            "reservation_time": "2026-04-10T19:00:00",
            "indoor_seating": False
        }

    elif task_type == "exact_label_classification":
        if "battery life is acceptable" in input_text:
            return {
                "label": "neutral",
                "rationale": "The text contains both mildly positive and negative sentiment."
            }
        return {
            "label": "urgent",
            "rationale": "The production API outage is blocking all customer transactions."
        }

    elif task_type == "json_repair":
        if "skills" in input_text:
            return {
                "name": "Alice",
                "age": 29,
                "skills": ["python", "sql"]
            }
        return {
            "name": "Carlos",
            "department": "finance",
            "active": True
        }

    elif task_type == "tool_call_argument_generation":
        if "Austin" in input_text:
            return {
                "city": "Austin",
                "state": "Texas",
                "unit": "fahrenheit"
            }
        return {
            "title": "Project Sync",
            "date": "2026-04-10",
            "time": "14:30",
            "duration_minutes": 45
        }

    return {"error": "Unhandled task type"}


def is_valid_json_output(output_obj):
    try:
        json.dumps(output_obj)
        return True
    except Exception:
        return False


def normalize_example(seed_row):
    teacher_output = build_local_teacher_output(
        task_type=seed_row["task_type"],
        instruction=seed_row["instruction"],
        input_text=seed_row["input"]
    )

    if not is_valid_json_output(teacher_output):
        return None

    return {
        "id": seed_row["id"],
        "task_type": seed_row["task_type"],
        "instruction": seed_row["instruction"],
        "input": seed_row["input"],
        "output": json.dumps(teacher_output, ensure_ascii=False),
        "expected_schema": seed_row["expected_schema"]
    }


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main():
    random.seed(SEED)

    seed_rows = load_seed_prompts()
    processed = []

    for row in seed_rows:
        normalized = normalize_example(row)
        if normalized is not None:
            processed.append(normalized)

    random.shuffle(processed)

    split_idx = int(len(processed) * TRAIN_RATIO)
    train_rows = processed[:split_idx]
    eval_rows = processed[split_idx:]

    write_jsonl(TRAIN_OUT, train_rows)
    write_jsonl(EVAL_OUT, eval_rows)

    print(f"Total JSON teacher examples: {len(processed)}")
    print(f"Train written: {len(train_rows)} -> {TRAIN_OUT}")
    print(f"Eval written: {len(eval_rows)} -> {EVAL_OUT}")


if __name__ == "__main__":
    main()
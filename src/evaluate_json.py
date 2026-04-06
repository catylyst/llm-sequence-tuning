import argparse
import json
from collections import Counter
from pathlib import Path

from utils import load_yaml, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, choices=["checkpoint0", "checkpoint1", "checkpoint2"])
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def try_parse_json(text):
    try:
        return True, json.loads(text), None
    except Exception as e:
        return False, None, str(e)


def type_matches(value, expected_type):
    if expected_type == "string":
        return isinstance(value, str)
    if expected_type == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected_type == "number":
        return (isinstance(value, int) or isinstance(value, float)) and not isinstance(value, bool)
    if expected_type == "boolean":
        return isinstance(value, bool)
    if expected_type == "array[string]":
        return isinstance(value, list) and all(isinstance(x, str) for x in value)
    return True


def schema_compliant(parsed_obj, expected_schema):
    if not isinstance(parsed_obj, dict):
        return False, "not_object"

    expected_keys = set(expected_schema.keys())
    actual_keys = set(parsed_obj.keys())

    if expected_keys != actual_keys:
        if actual_keys - expected_keys:
            return False, "extra_fields"
        if expected_keys - actual_keys:
            return False, "missing_fields"
        return False, "key_mismatch"

    for key, expected_type in expected_schema.items():
        if not type_matches(parsed_obj[key], expected_type):
            return False, f"wrong_type:{key}"

    return True, None


def normalize_for_exact_match(obj):
    if isinstance(obj, dict):
        return {k: normalize_for_exact_match(obj[k]) for k in sorted(obj.keys())}
    if isinstance(obj, list):
        return [normalize_for_exact_match(x) for x in obj]
    return obj


def compute_field_level_scores(reference_obj, predicted_obj):
    """
    Simple micro-style field comparison for dict outputs.
    Counts key-level exact matches.
    """
    if not isinstance(reference_obj, dict) or not isinstance(predicted_obj, dict):
        return 0, 0, 0

    ref_keys = set(reference_obj.keys())
    pred_keys = set(predicted_obj.keys())

    true_positive = 0
    for key in ref_keys & pred_keys:
        if reference_obj[key] == predicted_obj[key]:
            true_positive += 1

    false_positive = len(pred_keys - ref_keys)
    false_negative = len(ref_keys - pred_keys)

    for key in ref_keys & pred_keys:
        if reference_obj[key] != predicted_obj[key]:
            false_positive += 1
            false_negative += 1

    return true_positive, false_positive, false_negative


def categorize_parse_error(text, error_message):
    text = (text or "").strip()

    if not text:
        return "empty_output"
    if text.count("{") != text.count("}") or text.count("[") != text.count("]"):
        return "unbalanced_brackets"
    if "Expecting property name enclosed in double quotes" in (error_message or ""):
        return "invalid_quotes_or_keys"
    if "Extra data" in (error_message or ""):
        return "extra_text"
    if len(text) < 5:
        return "truncated_output"
    return "invalid_json_other"


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    input_path = Path("outputs") / args.checkpoint / "json_outputs.jsonl"
    metrics_path = Path("outputs/metrics") / f"{args.checkpoint}_json_metrics.json"
    details_path = Path("outputs/metrics") / f"{args.checkpoint}_json_details.jsonl"
    ensure_dir(metrics_path.parent)

    rows = read_jsonl(input_path)

    total = len(rows)
    valid_json_count = 0
    schema_ok_count = 0
    exact_match_count = 0

    tp_total = 0
    fp_total = 0
    fn_total = 0

    error_counter = Counter()
    detail_rows = []

    for row in rows:
        reference_output = row.get("reference_output", "")
        model_output = row.get("model_output", "")

        expected_schema = row.get("expected_schema")
        if expected_schema is None:
            # fallback in case generation output file did not include it
            expected_schema = {}

        ref_valid, ref_obj, ref_err = try_parse_json(reference_output)
        pred_valid, pred_obj, pred_err = try_parse_json(model_output)

        row_result = {
            "id": row.get("id"),
            "checkpoint": row.get("checkpoint"),
            "task_type": row.get("task_type"),
            "json_valid": pred_valid,
            "schema_compliant": False,
            "exact_match": False,
            "error_type": None,
        }

        if pred_valid:
            valid_json_count += 1

            schema_ok, schema_error = schema_compliant(pred_obj, expected_schema)
            row_result["schema_compliant"] = schema_ok

            if schema_ok:
                schema_ok_count += 1

            if ref_valid and normalize_for_exact_match(pred_obj) == normalize_for_exact_match(ref_obj):
                exact_match_count += 1
                row_result["exact_match"] = True

            if ref_valid:
                tp, fp, fn = compute_field_level_scores(ref_obj, pred_obj)
                tp_total += tp
                fp_total += fp
                fn_total += fn

            if not schema_ok:
                error_counter[schema_error or "schema_failure"] += 1
                row_result["error_type"] = schema_error or "schema_failure"
        else:
            error_type = categorize_parse_error(model_output, pred_err)
            error_counter[error_type] += 1
            row_result["error_type"] = error_type

        detail_rows.append(row_result)

    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    field_f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics = {
        "checkpoint": args.checkpoint,
        "total_examples": total,
        "json_validity_rate": valid_json_count / total if total else 0.0,
        "schema_compliance_rate": schema_ok_count / total if total else 0.0,
        "exact_match_rate": exact_match_count / total if total else 0.0,
        "field_level_precision": precision,
        "field_level_recall": recall,
        "field_level_f1": field_f1,
        "error_taxonomy": dict(error_counter),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    with open(details_path, "w", encoding="utf-8") as f:
        for row in detail_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
import argparse
import json
from pathlib import Path

import evaluate
from utils import ensure_dir, load_yaml


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


def task_completion_rate(rows):
    completed = 0
    for row in rows:
        output = (row.get("model_output") or "").strip()
        if output:
            completed += 1
    return completed / len(rows) if rows else 0.0


def average_output_length(rows):
    lengths = []
    for row in rows:
        output = (row.get("model_output") or "").strip()
        lengths.append(len(output.split()))
    return sum(lengths) / len(lengths) if lengths else 0.0


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    input_path = Path("outputs") / args.checkpoint / "alpaca_outputs.jsonl"
    metrics_path = Path("outputs/metrics") / f"{args.checkpoint}_alpaca_metrics.json"
    ensure_dir(metrics_path.parent)

    rows = read_jsonl(input_path)

    predictions = [(row.get("model_output") or "").strip() for row in rows]
    references = [(row.get("reference_output") or "").strip() for row in rows]

    rouge = evaluate.load("rouge")
    bertscore = evaluate.load("bertscore")

    rouge_scores = rouge.compute(
        predictions=predictions,
        references=references
    )

    bert_scores = bertscore.compute(
        predictions=predictions,
        references=references,
        lang="en"
    )

    avg_bertscore_precision = sum(bert_scores["precision"]) / len(bert_scores["precision"]) if bert_scores["precision"] else 0.0
    avg_bertscore_recall = sum(bert_scores["recall"]) / len(bert_scores["recall"]) if bert_scores["recall"] else 0.0
    avg_bertscore_f1 = sum(bert_scores["f1"]) / len(bert_scores["f1"]) if bert_scores["f1"] else 0.0

    metrics = {
        "checkpoint": args.checkpoint,
        "total_examples": len(rows),
        "rouge1": rouge_scores.get("rouge1", 0.0),
        "rouge2": rouge_scores.get("rouge2", 0.0),
        "rougeL": rouge_scores.get("rougeL", 0.0),
        "rougeLsum": rouge_scores.get("rougeLsum", 0.0),
        "bertscore_precision": avg_bertscore_precision,
        "bertscore_recall": avg_bertscore_recall,
        "bertscore_f1": avg_bertscore_f1,
        "average_output_length_words": average_output_length(rows),
        "task_completion_rate": task_completion_rate(rows),
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
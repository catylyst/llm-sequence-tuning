import json
from pathlib import Path


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def safe_load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def summarize_judge_file(path):
    rows = read_jsonl(path)
    total = len(rows)

    wins = {}
    for row in rows:
        winner = row.get("canonical_winner", "Tie")
        wins[winner] = wins.get(winner, 0) + 1

    return {
        "file": str(path),
        "total_pairs": total,
        "wins": wins,
    }


def main():
    metrics_dir = Path("outputs/metrics")
    judge_dir = Path("outputs/judge_scores")
    summary_out = metrics_dir / "final_summary.json"

    summary = {
        "alpaca_metrics": {},
        "json_metrics": {},
        "judge_summaries": [],
    }

    for checkpoint in ["checkpoint0", "checkpoint1", "checkpoint2"]:
        alpaca_path = metrics_dir / f"{checkpoint}_alpaca_metrics.json"
        json_path = metrics_dir / f"{checkpoint}_json_metrics.json"

        if alpaca_path.exists():
            summary["alpaca_metrics"][checkpoint] = safe_load_json(alpaca_path)

        if json_path.exists():
            summary["json_metrics"][checkpoint] = safe_load_json(json_path)

    if judge_dir.exists():
        for path in sorted(judge_dir.glob("*.jsonl")):
            summary["judge_summaries"].append(summarize_judge_file(path))

    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(summary_out, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Saved aggregate summary to {summary_out}")


if __name__ == "__main__":
    main()
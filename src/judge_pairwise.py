import argparse
import json
import random
from pathlib import Path

from utils import ensure_dir, load_yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--eval_type", type=str, required=True, choices=["alpaca", "json"])
    parser.add_argument("--checkpoint_a", type=str, required=True, choices=["checkpoint0", "checkpoint1", "checkpoint2"])
    parser.add_argument("--checkpoint_b", type=str, required=True, choices=["checkpoint0", "checkpoint1", "checkpoint2"])
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def read_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_judge_prompt(example, response_a, response_b, eval_type):
    base = (
        "You are an impartial judge comparing two model responses to the same prompt.\n"
        "Score both responses from 1 to 5.\n"
        "Then choose one winner: A, B, or Tie.\n"
        "Return valid JSON only.\n\n"
    )

    if eval_type == "alpaca":
        criteria = (
            "Score on these dimensions:\n"
            "- instruction_following\n"
            "- correctness\n"
            "- clarity\n"
            "- completeness\n"
            "- hallucination_risk\n\n"
        )
    else:
        criteria = (
            "Score on these dimensions:\n"
            "- instruction_following\n"
            "- correctness\n"
            "- clarity\n"
            "- completeness\n"
            "- structured_output_validity\n"
            "- hallucination_risk\n\n"
        )

    prompt = (
        f"{base}"
        f"{criteria}"
        f"Instruction:\n{example.get('instruction', '')}\n\n"
        f"Input:\n{example.get('input', '')}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        "Return JSON with this structure:\n"
        "{\n"
        '  "response_a_scores": {...},\n'
        '  "response_b_scores": {...},\n'
        '  "winner": "A",\n'
        '  "justification": "..." \n'
        "}\n"
    )
    return prompt


def heuristic_score_response(example, response, eval_type):
    """
    Temporary local scoring function so the pipeline works end to end.
    Later this should be replaced by a strong judge-model API call.
    """
    instruction = (example.get("instruction") or "").lower()
    model_output = (response or "").strip()

    score = {
        "instruction_following": 3,
        "correctness": 3,
        "clarity": 3,
        "completeness": 3,
        "hallucination_risk": 3,
    }

    if eval_type == "json":
        score["structured_output_validity"] = 1
        try:
            parsed = json.loads(model_output)
            if isinstance(parsed, dict):
                score["structured_output_validity"] = 5
                score["instruction_following"] += 1
                score["correctness"] += 1
                score["clarity"] += 1
        except Exception:
            pass

    if model_output:
        score["completeness"] += 1
        score["clarity"] += 1

    if len(model_output.split()) > 8:
        score["instruction_following"] += 1

    if "json" in instruction:
        try:
            json.loads(model_output)
            score["correctness"] += 1
        except Exception:
            pass

    for key in list(score.keys()):
        score[key] = max(1, min(5, score[key]))

    return score


def total_score(score_dict):
    return sum(score_dict.values())


def judge_pair(example, response_a, response_b, eval_type):
    score_a = heuristic_score_response(example, response_a, eval_type)
    score_b = heuristic_score_response(example, response_b, eval_type)

    total_a = total_score(score_a)
    total_b = total_score(score_b)

    if total_a > total_b:
        winner = "A"
    elif total_b > total_a:
        winner = "B"
    else:
        winner = "Tie"

    justification = (
        f"Response A total score = {total_a}, Response B total score = {total_b}. "
        f"The higher-scoring response was selected based on the rubric."
    )

    return {
        "response_a_scores": score_a,
        "response_b_scores": score_b,
        "winner": winner,
        "justification": justification,
    }


def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    random.seed(args.seed)

    path_a = Path("outputs") / args.checkpoint_a / f"{args.eval_type}_outputs.jsonl"
    path_b = Path("outputs") / args.checkpoint_b / f"{args.eval_type}_outputs.jsonl"

    rows_a = read_jsonl(path_a)
    rows_b = read_jsonl(path_b)

    by_id_a = {row["id"]: row for row in rows_a}
    by_id_b = {row["id"]: row for row in rows_b}

    common_ids = sorted(set(by_id_a.keys()) & set(by_id_b.keys()))

    out_dir = Path("outputs/judge_scores")
    ensure_dir(out_dir)

    out_path = out_dir / f"{args.eval_type}_{args.checkpoint_a}_vs_{args.checkpoint_b}.jsonl"

    with open(out_path, "w", encoding="utf-8") as f:
        for prompt_id in common_ids:
            row_a = by_id_a[prompt_id]
            row_b = by_id_b[prompt_id]

            swap = random.choice([True, False])

            if swap:
                response_a = row_b["model_output"]
                response_b = row_a["model_output"]
                displayed_checkpoint_a = args.checkpoint_b
                displayed_checkpoint_b = args.checkpoint_a
            else:
                response_a = row_a["model_output"]
                response_b = row_b["model_output"]
                displayed_checkpoint_a = args.checkpoint_a
                displayed_checkpoint_b = args.checkpoint_b

            _judge_prompt = build_judge_prompt(row_a, response_a, response_b, args.eval_type)

            judged = judge_pair(row_a, response_a, response_b, args.eval_type)

            winner = judged["winner"]
            if swap:
                if winner == "A":
                    canonical_winner = args.checkpoint_b
                elif winner == "B":
                    canonical_winner = args.checkpoint_a
                else:
                    canonical_winner = "Tie"
            else:
                if winner == "A":
                    canonical_winner = args.checkpoint_a
                elif winner == "B":
                    canonical_winner = args.checkpoint_b
                else:
                    canonical_winner = "Tie"

            result = {
                "prompt_id": prompt_id,
                "eval_type": args.eval_type,
                "checkpoint_a": displayed_checkpoint_a,
                "checkpoint_b": displayed_checkpoint_b,
                "canonical_checkpoint_a": args.checkpoint_a,
                "canonical_checkpoint_b": args.checkpoint_b,
                "response_a_scores": judged["response_a_scores"],
                "response_b_scores": judged["response_b_scores"],
                "winner": judged["winner"],
                "canonical_winner": canonical_winner,
                "justification": judged["justification"],
                "swapped_order": swap,
            }

            f.write(json.dumps(result, ensure_ascii=False) + "\n")

    print(f"Saved judge results to {out_path}")


if __name__ == "__main__":
    main()
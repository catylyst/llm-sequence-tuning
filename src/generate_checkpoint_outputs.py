import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

from utils import load_yaml, ensure_dir


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    parser.add_argument("--checkpoint", type=str, required=True, choices=["checkpoint0", "checkpoint1", "checkpoint2"])
    parser.add_argument("--eval_type", type=str, required=True, choices=["alpaca", "json"])
    return parser.parse_args()


def build_prompt(example):
    instruction = (example.get("instruction") or "").strip()
    input_text = (example.get("input") or "").strip()

    if input_text:
        return (
            "### Instruction:\n"
            f"{instruction}\n\n"
            "### Input:\n"
            f"{input_text}\n\n"
            "### Response:\n"
        )
    return (
        "### Instruction:\n"
        f"{instruction}\n\n"
        "### Response:\n"
    )


def load_model_for_checkpoint(cfg, checkpoint_name):
    model_name = cfg["models"]["student_model"]

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["training"]["load_in_4bit"],
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg["training"]["bf16"] else torch.float16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    if checkpoint_name == "checkpoint0":
        return base_model, tokenizer

    if checkpoint_name == "checkpoint1":
        model = PeftModel.from_pretrained(base_model, cfg["stage1"]["output_dir"])
        return model, tokenizer

    if checkpoint_name == "checkpoint2":
        model = PeftModel.from_pretrained(base_model, cfg["stage2"]["output_dir"])
        return model, tokenizer

    raise ValueError(f"Unknown checkpoint: {checkpoint_name}")


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    eval_path = cfg["data"]["alpaca_eval_path"] if args.eval_type == "alpaca" else cfg["data"]["json_eval_path"]
    output_path = Path("outputs") / args.checkpoint / f"{args.eval_type}_outputs.jsonl"
    ensure_dir(output_path.parent)

    print(f"Loading eval set: {eval_path}")
    dataset = load_dataset("json", data_files=eval_path, split="train")

    print(f"Loading model for {args.checkpoint}")
    model, tokenizer = load_model_for_checkpoint(cfg, args.checkpoint)
    model.eval()

    rows = []
    for idx, ex in enumerate(dataset):
        prompt = build_prompt(ex)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=cfg["evaluation"]["max_new_tokens"],
                temperature=cfg["evaluation"]["temperature"],
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = decoded[len(prompt):].strip() if decoded.startswith(prompt) else decoded.strip()

        rows.append({
            "id": ex.get("id", f"{args.eval_type}_{idx:04d}"),
            "task_type": ex.get("task_type", args.eval_type),
            "instruction": ex.get("instruction", ""),
            "input": ex.get("input", ""),
            "reference_output": ex.get("output", ""),
            "model_output": generated_text,
            "checkpoint": args.checkpoint,
            "eval_type": args.eval_type,
        })

        if (idx + 1) % 10 == 0:
            print(f"Generated {idx + 1} / {len(dataset)}")

    with open(output_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Saved outputs to {output_path}")


if __name__ == "__main__":
    main()
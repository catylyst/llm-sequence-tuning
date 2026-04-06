import argparse
import os

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig
from trl import SFTTrainer

from utils import load_yaml, ensure_dir, format_example


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/config.yaml")
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_yaml(args.config)

    model_name = cfg["models"]["student_model"]
    train_path = cfg["data"]["alpaca_train_path"]
    output_dir = cfg["stage1"]["output_dir"]

    max_seq_length = cfg["training"]["max_seq_length"]
    learning_rate = cfg["stage1"]["learning_rate"]
    num_train_epochs = cfg["stage1"]["epochs"]
    per_device_train_batch_size = cfg["stage1"]["batch_size"]
    gradient_accumulation_steps = cfg["stage1"]["grad_accum"]

    lora_r = cfg["lora"]["r"]
    lora_alpha = cfg["lora"]["alpha"]
    lora_dropout = cfg["lora"]["dropout"]

    ensure_dir(output_dir)

    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading training data from: {train_path}")
    dataset = load_dataset("json", data_files=train_path, split="train")
    dataset = dataset.map(format_example)

    print("Preparing 4-bit quantization config...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["training"]["load_in_4bit"],
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if cfg["training"]["bf16"] else torch.float16,
    )

    print(f"Loading base model: {model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )

    model.config.use_cache = False

    if cfg["training"]["gradient_checkpointing"]:
        model.gradient_checkpointing_enable()

    peft_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
    )

    training_args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="no",
        report_to="none",
        bf16=cfg["training"]["bf16"],
        fp16=not cfg["training"]["bf16"],
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        max_grad_norm=0.3,
        seed=cfg["project"]["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        peft_config=peft_config,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )

    print("Starting Stage 1 training...")
    trainer.train()

    print(f"Saving Stage 1 adapter and tokenizer to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Stage 1 complete.")


if __name__ == "__main__":
    main()
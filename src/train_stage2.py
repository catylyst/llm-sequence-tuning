import argparse

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import PeftModel
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
    stage1_dir = cfg["stage1"]["output_dir"]
    train_path = cfg["data"]["json_train_path"]
    output_dir = cfg["stage2"]["output_dir"]

    max_seq_length = cfg["training"]["max_seq_length"]
    learning_rate = cfg["stage2"]["learning_rate"]
    num_train_epochs = cfg["stage2"]["epochs"]
    per_device_train_batch_size = cfg["stage2"]["batch_size"]
    gradient_accumulation_steps = cfg["stage2"]["grad_accum"]

    ensure_dir(output_dir)

    print(f"Loading tokenizer from base model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading JSON Stage 2 data from: {train_path}")
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
    base_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    base_model.config.use_cache = False

    if cfg["training"]["gradient_checkpointing"]:
        base_model.gradient_checkpointing_enable()

    print(f"Loading Stage 1 adapter from: {stage1_dir}")
    model = PeftModel.from_pretrained(base_model, stage1_dir, is_trainable=True)

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
        args=training_args,
        dataset_text_field="text",
        max_seq_length=max_seq_length,
        packing=False,
    )

    print("Starting Stage 2 training...")
    trainer.train()

    print(f"Saving Stage 2 adapter and tokenizer to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Stage 2 complete.")


if __name__ == "__main__":
    main()
# Assignment 3: Sequential Instruction Tuning of a Small LLM

This project implements a two-stage sequential instruction tuning pipeline for a small language model using QLoRA on UTSA HPC.

## Stages
1. Stage 1: Fine-tune on Alpaca-style instruction data
2. Stage 2: Continue fine-tuning on teacher-generated JSON structured output data

## Evaluation
The model is evaluated at:
- Checkpoint 0: untuned base model
- Checkpoint 1: after Alpaca tuning
- Checkpoint 2: after JSON tuning

Metrics include:
- Alpaca judge win rate
- ROUGE / BERTScore
- JSON validity
- Schema compliance
- Exact match
- Forgetting analysis

## Structure
- `src/` contains scripts
- `config/` contains configuration
- `prompts/` contains prompt templates
- `hpc/` contains Slurm scripts
- `outputs/` contains results
- `logs/` contains training/evaluation logs

## Reproduction
To be completed after pipeline implementation.
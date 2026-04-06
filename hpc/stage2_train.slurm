#!/bin/bash
#SBATCH --job-name=stage2_phi35
#SBATCH --output=logs/stage2/%x_%j.out
#SBATCH --error=logs/stage2/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_align

cd $HOME/llm-sequential-tuning

python src/train_stage2.py --config config/config.yaml
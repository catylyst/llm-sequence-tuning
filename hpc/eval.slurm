#!/bin/bash
#SBATCH --job-name=eval_phi35
#SBATCH --output=logs/eval/%x_%j.out
#SBATCH --error=logs/eval/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00

source ~/miniconda3/etc/profile.d/conda.sh
conda activate llm_align

cd $HOME/llm-sequential-tuning

python src/generate_checkpoint_outputs.py --checkpoint checkpoint0 --eval_type alpaca --config config/config.yaml
python src/generate_checkpoint_outputs.py --checkpoint checkpoint0 --eval_type json --config config/config.yaml

python src/generate_checkpoint_outputs.py --checkpoint checkpoint1 --eval_type alpaca --config config/config.yaml
python src/generate_checkpoint_outputs.py --checkpoint checkpoint1 --eval_type json --config config/config.yaml

python src/generate_checkpoint_outputs.py --checkpoint checkpoint2 --eval_type alpaca --config config/config.yaml
python src/generate_checkpoint_outputs.py --checkpoint checkpoint2 --eval_type json --config config/config.yaml

python src/evaluate_alpaca.py --checkpoint checkpoint0 --config config/config.yaml
python src/evaluate_alpaca.py --checkpoint checkpoint1 --config config/config.yaml
python src/evaluate_alpaca.py --checkpoint checkpoint2 --config config/config.yaml

python src/evaluate_json.py --checkpoint checkpoint0 --config config/config.yaml
python src/evaluate_json.py --checkpoint checkpoint1 --config config/config.yaml
python src/evaluate_json.py --checkpoint checkpoint2 --config config/config.yaml

python src/judge_pairwise.py --eval_type alpaca --checkpoint_a checkpoint0 --checkpoint_b checkpoint1 --config config/config.yaml
python src/judge_pairwise.py --eval_type alpaca --checkpoint_a checkpoint1 --checkpoint_b checkpoint2 --config config/config.yaml
python src/judge_pairwise.py --eval_type alpaca --checkpoint_a checkpoint0 --checkpoint_b checkpoint2 --config config/config.yaml

python src/judge_pairwise.py --eval_type json --checkpoint_a checkpoint0 --checkpoint_b checkpoint1 --config config/config.yaml
python src/judge_pairwise.py --eval_type json --checkpoint_a checkpoint1 --checkpoint_b checkpoint2 --config config/config.yaml
python src/judge_pairwise.py --eval_type json --checkpoint_a checkpoint0 --checkpoint_b checkpoint2 --config config/config.yaml

python src/aggregate_results.py
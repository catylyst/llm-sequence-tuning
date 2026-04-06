#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda create -n llm_align python=3.10 -y
conda activate llm_align

pip install --upgrade pip
pip install torch transformers datasets peft trl accelerate bitsandbytes sentencepiece pyyaml scikit-learn pandas numpy matplotlib evaluate rouge-score bert-score jsonschema tqdm
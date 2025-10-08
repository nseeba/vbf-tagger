#!/bin/bash

#SBATCH -p gpu
#SBATCH --gres gpu:rtx
#SBATCH --mem-per-gpu 40G
#SBATCH -o /home/norman/vbf-tagger/slurm-logs/slurm-%x-%j.out
#SBATCH -e /home/norman/vbf-tagger/slurm-logs/slurm-%x-%j.err

# env | grep CUDA
# nvidia-smi -L

./run.sh python3 vbf_tagger/scripts/train.py training.output_dir=$1 environment@host=manivald training.type=classification training.model_evaluation=$2
#!/usr/bin/env bash
#SBATCH --job-name=bm_traffic_H96_s2025
#SBATCH --output=logs/bm_traffic_H96_s2025_%j.out
#SBATCH --error=logs/bm_traffic_H96_s2025_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Environment setup
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"
mkdir -p logs results/cluster_results

echo "=========================================================="
echo " Starting Job: bm_traffic_H96_s2025 on $(hostname)"
echo " GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo " Date: $(date)"
echo "=========================================================="

# Run benchmark inside cluster virtualenv
.venv/bin/python3 run_bigmama_benchmark.py \
    --dataset traffic.csv \
    --pred_len 96 \
    --seed 2025 \
    --full_test \
    --output_dir results/cluster_results

echo "Job bm_traffic_H96_s2025 finished at $(date) with exit code $?"

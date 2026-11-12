#!/usr/bin/env bash
#SBATCH --job-name=srp26_array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=1-60%8

# Read parameters for this array task
PARAM_LINE=$(sed -n "${SLURM_ARRAY_TASK_ID}p" job_parameters.txt)
DATASET=$(echo $PARAM_LINE | cut -d' ' -f1)
PRED_LEN=$(echo $PARAM_LINE | cut -d' ' -f2)
SEED=$(echo $PARAM_LINE | cut -d' ' -f3)

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"
mkdir -p logs results/cluster_results

echo "Running task $SLURM_ARRAY_TASK_ID: Dataset=$DATASET Horizon=$PRED_LEN Seed=$SEED on $(hostname)"
.venv/bin/python3 run_bigmama_benchmark.py \
    --dataset "$DATASET" \
    --pred_len "$PRED_LEN" \
    --seed "$SEED" \
    --full_test \
    --output_dir results/cluster_results

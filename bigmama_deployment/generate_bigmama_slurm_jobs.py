"""
Bigmama SLURM Job Script Generator (ISMLL Cluster)
==================================================
Generates individual SLURM submission scripts and a unified SLURM task array
for running the multi-foundation & Fourier decoupling benchmark across Bigmama.
"""

import os
import stat

DATASETS = [
    "ETTh1.csv",
    "ETTh2.csv",
    "weather.csv",
    "traffic.csv",
    "electricity.csv"
]
HORIZONS = [96, 192, 336, 720]
SEEDS = [2024, 2025, 2026]


def generate_slurm_scripts():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    slurm_dir = os.path.join(base_dir, "slurm_scripts")
    logs_dir = os.path.join(base_dir, "logs")
    os.makedirs(slurm_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    job_entries = []
    script_paths = []

    for dataset in DATASETS:
        ds_clean = dataset.split('.')[0]
        for pred_len in HORIZONS:
            for seed in SEEDS:
                job_name = f"bm_{ds_clean}_H{pred_len}_s{seed}"
                sh_path = os.path.join(slurm_dir, f"run_{job_name}.sh")

                content = f"""#!/usr/bin/env bash
#SBATCH --job-name={job_name}
#SBATCH --output=logs/{job_name}_%j.out
#SBATCH --error=logs/{job_name}_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

# Environment setup
export HF_HOME="${{HF_HOME:-$HOME/.cache/huggingface}}"
mkdir -p "$HF_HOME"
mkdir -p logs results/cluster_results

echo "=========================================================="
echo " Starting Job: {job_name} on $(hostname)"
echo " GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo " Date: $(date)"
echo "=========================================================="

# Run benchmark inside cluster virtualenv
.venv/bin/python3 run_bigmama_benchmark.py \\
    --dataset {dataset} \\
    --pred_len {pred_len} \\
    --seed {seed} \\
    --full_test \\
    --output_dir results/cluster_results

echo "Job {job_name} finished at $(date) with exit code $?"
"""
                with open(sh_path, "w", newline="\n") as f:
                    f.write(content)
                script_paths.append(f"slurm_scripts/run_{job_name}.sh")
                job_entries.append((dataset, pred_len, seed))

    # Master submit script
    master_submit = os.path.join(base_dir, "submit_all_jobs.sh")
    with open(master_submit, "w", newline="\n") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Master SLURM submission script for Bigmama\n")
        f.write("mkdir -p logs results/cluster_results\n\n")
        for p in script_paths:
            f.write(f"sbatch {p}\n")
    # Make executable
    os.chmod(master_submit, os.stat(master_submit).st_mode | stat.S_IEXEC)

    # Master Slurm Array Job script (Cleaner & preferred by cluster admins)
    array_job_path = os.path.join(base_dir, "submit_array_job.sh")
    param_list_path = os.path.join(base_dir, "job_parameters.txt")
    with open(param_list_path, "w", newline="\n") as f:
        for ds, pl, sd in job_entries:
            f.write(f"{ds} {pl} {sd}\n")

    num_jobs = len(job_entries)
    with open(array_job_path, "w", newline="\n") as f:
        f.write(f"""#!/usr/bin/env bash
#SBATCH --job-name=srp26_array
#SBATCH --output=logs/array_%A_%a.out
#SBATCH --error=logs/array_%A_%a.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00
#SBATCH --array=1-{num_jobs}%8

# Read parameters for this array task
PARAM_LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" job_parameters.txt)
DATASET=$(echo $PARAM_LINE | cut -d' ' -f1)
PRED_LEN=$(echo $PARAM_LINE | cut -d' ' -f2)
SEED=$(echo $PARAM_LINE | cut -d' ' -f3)

export HF_HOME="${{HF_HOME:-$HOME/.cache/huggingface}}"
mkdir -p "$HF_HOME"
mkdir -p logs results/cluster_results

echo "Running task $SLURM_ARRAY_TASK_ID: Dataset=$DATASET Horizon=$PRED_LEN Seed=$SEED on $(hostname)"
.venv/bin/python3 run_bigmama_benchmark.py \\
    --dataset "$DATASET" \\
    --pred_len "$PRED_LEN" \\
    --seed "$SEED" \\
    --full_test \\
    --output_dir results/cluster_results
""")
    os.chmod(array_job_path, os.stat(array_job_path).st_mode | stat.S_IEXEC)

    print(f"Generated {len(script_paths)} SLURM job scripts in {slurm_dir}")
    print(f"Generated unified array job script: {array_job_path} (max 8 concurrent jobs)")
    print(f"Generated master submit script: {master_submit}")


if __name__ == "__main__":
    generate_slurm_scripts()


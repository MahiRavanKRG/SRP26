"""Generate and upload SLURM job scripts for the transform-ablation test suite."""

import os
import sys
import paramiko

DATASETS = [
    "ETTh1",
    "ETTh2",
    "ETTm1",
    "ETTm2",
    "weather",
    "electricity",
    "traffic",
]

def generate_script(ds: str) -> str:
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=ta_{ds}
#SBATCH --output=logs/ta_{ds}_%j.out
#SBATCH --error=logs/ta_{ds}_%j.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --time=04:00:00

mkdir -p logs results/transform_ablation results/transform_ablation/raw

echo "=========================================================="
echo " Starting Transform Ablation Job: ta_{ds} on $(hostname)"
echo " GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo " Date: $(date)"
echo "=========================================================="

.venv/bin/python3 experiment_transform_ablation.py \\
    --datasets {ds}.csv \\
    --horizons 96 192 336 720 \\
    --transforms identity dlinear rfft dct stft dwt_haar dwt_db4 \\
    --seeds 2024 2025 2026 \\
    --epochs 30 \\
    --patience 5 \\
    --save_dir results/transform_ablation

.venv/bin/python3 scripts/aggregate_transform_ablation.py \\
    --save_dir results/transform_ablation

echo "Job ta_{ds} finished at $(date) with exit code $?"
"""

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    local_slurm_dir = os.path.join(repo_root, "bigmama_deployment", "slurm_scripts")
    os.makedirs(local_slurm_dir, exist_ok=True)

    script_paths = []
    for ds in DATASETS:
        fname = f"run_ta_{ds}.sh"
        fpath = os.path.join(local_slurm_dir, fname)
        with open(fpath, "w", newline="\n") as f:
            f.write(generate_script(ds))
        script_paths.append((fname, fpath))
        print(f"Generated {fname}")

    # Also generate a master submit script
    master_submit = os.path.join(local_slurm_dir, "submit_all_ta_jobs.sh")
    with open(master_submit, "w", newline="\n") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("# Submit all transform ablation jobs to Slurm queue\n")
        for ds in DATASETS:
            f.write(f"sbatch slurm_scripts/run_ta_{ds}.sh\n")
    print("Generated submit_all_ta_jobs.sh")

    # Connect to Bigmama via Paramiko and upload
    env_file = os.path.join(repo_root, ".env")
    env_vars = {}
    if os.path.exists(env_file):
        with open(env_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    env_vars[k.strip()] = v.strip().strip("'\"")

    host = os.getenv("BIGMAMA_HOST") or env_vars.get("BIGMAMA_HOST", "")
    user = os.getenv("BIGMAMA_USER") or env_vars.get("BIGMAMA_USER", "")
    password = os.getenv("BIGMAMA_PASS") or env_vars.get("BIGMAMA_PASS")
    remote_dir = (os.getenv("BIGMAMA_REMOTE_DIR") or env_vars.get("BIGMAMA_REMOTE_DIR", "~/SRP26")).replace("~", f"/home/{user}")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=password, timeout=15)
    sftp = ssh.open_sftp()

    print("\nUploading SLURM scripts to Bigmama:")
    for fname, fpath in script_paths:
        rem_path = f"{remote_dir}/slurm_scripts/{fname}"
        print(f" -> {fname}")
        sftp.put(fpath, rem_path)
    sftp.put(master_submit, f"{remote_dir}/slurm_scripts/submit_all_ta_jobs.sh")
    print(" -> submit_all_ta_jobs.sh")

    ssh.exec_command(f"chmod +x {remote_dir}/slurm_scripts/run_ta_*.sh {remote_dir}/slurm_scripts/submit_all_ta_jobs.sh")
    sftp.close()
    ssh.close()
    print("All scripts uploaded and permissions set!")

if __name__ == "__main__":
    main()


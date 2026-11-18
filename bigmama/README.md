# Bigmama Cluster Integration & Verification

This directory manages synchronization, SLURM batch job execution, and result collection for experiments running on the **Bigmama** cluster at the University of Hildesheim (`master.ismll.de`).

---

## Directory Structure

- `slurm_scripts/`: Generated `#SBATCH` scripts for the cluster (`--partition=STUD`, `--gres=gpu:1`).
- `logs/`: SLURM stdout (`%j.out`) and stderr (`%j.err`) files fetched from Bigmama.
- `results/`: Processed JSON metrics, diagnostic logs, and benchmark summaries pulled back from Bigmama.
- `sync_to_bigmama.py`: Automated tool to sync code, models, and datasets from local `SRP26` to `~/SRP26/` on Bigmama.
- `fetch_results.py`: Tool to download experiment results and logs from `~/SRP26/results/` to `bigmama/results/`.
- `run_cluster_job.py`: Remote submission and monitoring helper using SSH/Paramiko.

---

## Prerequisites

1. **VPN**: Ensure you are connected to the University of Hildesheim VPN (eduVPN or Cisco AnyConnect).
2. **Credentials**: Ensure `.env` is configured in the root with:
   ```env
   BIGMAMA_HOST=master.ismll.de
   BIGMAMA_USER=chaturvedia
   BIGMAMA_PASS=<your_password>
   BIGMAMA_REMOTE_DIR=~/SRP26
   ```

---

## Workflow: Verifying Experiments on Bigmama

### 1. Sync Code to Bigmama
```powershell
uv run python bigmama/sync_to_bigmama.py
```

### 2. Submit Experiment Jobs on Bigmama
```powershell
uv run python bigmama/run_cluster_job.py --submit slurm_scripts/run_bm_ETTh1_H96_s2024.sh
```
Or check the active queue:
```powershell
uv run python bigmama/run_cluster_job.py --queue
```

### 3. Fetch Completed Results Back to Local Repo
```powershell
uv run python bigmama/fetch_results.py
```
Results will be saved in [bigmama/results/](file:///d:/uni/SRP26/bigmama/results/) for comparison against local runs!


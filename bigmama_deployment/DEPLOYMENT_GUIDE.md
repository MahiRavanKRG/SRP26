# Bigmama Cluster Deployment & Reproduction Guide
> **Institution:** University of Hildesheim | ISMLL (Information Systems & Machine Learning Lab)  
> **Project:** Ultra-lightweight MTS Forecasting Sandbox & Spectral Decomposition (SRP26)  
> **Supervisor:** Jung Min Choi | **Researcher:** Aaditya Chaturvedi  
> **Target System:** `master.ismll.de` (SLURM Partition: `STUD`, GPU Nodes: NVIDIA A100 / RTX 8000 / V100)

---

## 1. Prerequisites: VPN & Access
Bigmama's head node (`master.ismll.de`, `147.172.239.240`) is located behind the University of Hildesheim campus network firewall.
1. Connect to the **University of Hildesheim VPN** (via Cisco AnyConnect or eduVPN) using your university credentials (`<username>@uni-hildesheim.de`).
2. Test connection in your terminal:
   ```bash
   ssh <username>@master.ismll.de
   ```

---

## 2. One-Command Sync to Bigmama

From your local machine (PowerShell on Windows):
```powershell
.\bigmama_deployment\sync_to_bigmama.ps1 -Username "<your_username>"
```
Or manually using `scp`:
```bash
# Transfer the deployment directory
scp -r bigmama_deployment <username>@master.ismll.de:~/SRP26/

# Transfer benchmark datasets (if not already on the cluster)
scp -r data_sets/* <username>@master.ismll.de:~/SRP26/data/
```

---

## 3. Remote Setup on Bigmama

SSH into Bigmama:
```bash
ssh <username>@master.ismll.de
cd ~/SRP26
```

Run the automated environment setup script:
```bash
bash setup_bigmama_env.sh
```
*This installs `uv` (if missing), sets up the GPU virtual environment, redirects Hugging Face caches to `/scratch/$USER/hf_cache` to avoid quota limits, and verifies CUDA acceleration.*

---

## 4. Submitting Benchmark Experiments

You have two submission options:

### Option A: SLURM Array Job (Recommended & Cleanest)
Runs all 60 tasks (5 datasets $\times$ 4 horizons $\times$ 3 seeds) with a concurrency cap of 8 parallel GPU jobs so the cluster queue remains fair:
```bash
sbatch submit_array_job.sh
```

### Option B: Submit Individual Jobs
Submit all 60 individual jobs to the queue:
```bash
bash submit_all_jobs.sh
```

### Option C: Single Test Run on an Interactive GPU Node
To test a single run interactively before launching the whole batch:
```bash
srun --partition=STUD --gres=gpu:1 --mem=32G --time=01:00:00 --pty bash
uv run python run_bigmama_benchmark.py --dataset ETTh1.csv --pred_len 96 --seed 2024 --full_test
```

---

## 5. Monitoring & Job Control

Check your running and queued jobs:
```bash
squeue -u $USER
```

Inspect live job outputs:
```bash
tail -f logs/srp26_array_*.out
```

Cancel all your running jobs if needed:
```bash
scancel -u $USER
```

---

## 6. Aggregating Results & Pulling Back

Once your SLURM jobs complete:
1. Aggregate multi-seed Mean $\pm$ Std across random seeds `[2024, 2025, 2026]`:
   ```bash
   uv run python aggregate_cluster_results.py
   ```
   *This automatically generates `results/bigmama_master_summary.md` and `results/bigmama_master_summary.json`.*

2. From your local machine, pull the completed cluster results back:
   ```bash
   scp -r <username>@master.ismll.de:~/SRP26/results/cluster_results/ fresh_experiments/results/
   scp <username>@master.ismll.de:~/SRP26/results/bigmama_master_summary.* fresh_experiments/results/
   ```

---

## 7. What Running on Bigmama Solves vs. Local Laptop
| Dimension | Local Laptop (RTX 4050 6GB) | Bigmama Cluster (`STUD` GPU Node) |
| :--- | :--- | :--- |
| **Test Split Coverage** | 15 sampled evaluation windows | **100% Full Continuous Test Split** |
| **Forecast Horizons** | $H = 96$ only | **All 4 Horizons: $H \in \{96, 192, 336, 720\}$** |
| **Multi-Seed Stability** | Seed 2024 | **3 Seeds: $[2024, 2025, 2026]$ (Mean $\pm$ Std)** |
| **Model Capacity** | Tiny/Base (Memory bounded) | **Full 200M Models (TimesFM, Chronos-Bolt, CycleNet-Pro)** |
| **Execution Time** | Sequential (~20-40 mins) | **Parallel Slurm Array (8 concurrent A100/V100 GPUs)** |


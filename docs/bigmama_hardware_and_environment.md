# Bigmama Cluster Hardware & Execution Environment Specification

> **Host**: `master.ismll.de` | **Institution**: Information Systems and Machine Learning Lab (ISMLL), University of Hildesheim  
> **Verification Date**: November 17, 2026  
> **Source Verification**: Queried live via Slurm controller (`scontrol`, `sinfo`), kernel (`lscpu`, `/proc/version`), driver (`nvidia-smi`), and package manager (`uv pip list`).

---

## 1. Primary Execution Node: `gpu-211`

This is the primary compute node in the `STUD` partition where the Transform-Ablation benchmark suite and foundation model benchmarks run.

### 1.1 GPU Subsystem
- **GPU Model**: **4× NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition**
- **VRAM per GPU**: **97,887 MiB (~96 GB)** GDDR7 / ECC
- **Total Node VRAM**: ~384 GB GPU Memory
- **TDP / Power Cap**: 300 W per GPU (Idle: ~8 W, Temp: ~30°C)
- **NVIDIA Driver**: `595.45.04`
- **Supported CUDA Driver Version**: `CUDA 13.2`
- **Compute Mode**: Default (Multi-Process / Slurm shared)

### 1.2 CPU Subsystem
- **Processor**: **AMD EPYC 9275F 24-Core Processor** (Zen 5 Server Architecture)
- **Cores / Threads**: 24 physical cores / 24 threads (1 thread per core, SMT disabled)
- **Clock Speeds**: Base `1.50 GHz`, Max Boost `4.82 GHz`
- **Cache Hierarchy**:
  - L1d: 1.1 MiB (24 instances)
  - L1i: 768 KiB (24 instances)
  - L2: 24 MiB (24 instances)
  - L3: **256 MiB** (8 instances of 32 MiB)
- **Vector Extensions**: AVX-512 (F, DQ, CD, BW, VL, IFMA, VBMI, VBMI2, VNNI, BITALG, VPOPCNTDQ, BF16), AVX2, SHA-NI, AES.

### 1.3 Memory & Operating System
- **Total Host RAM**: **380,000 MB (~380 GB / 371 GiB)** DDR5 ECC
- **Operating System**: Ubuntu Linux `6.8.0-31-generic #31-Ubuntu SMP PREEMPT_DYNAMIC` (x86_64)
- **Slurm Version**: `25.11.7`
- **Partitions Assigned**: `ROOT`, `STUD`, `STUDL`, `NGPU`, `6000`

---

## 2. Secondary Compute Pool: `stud-000`

The legacy shared student node available in partition `STUD`.

| Component | Specification |
| :--- | :--- |
| **GPU Subsystem** | 10× NVIDIA GeForce RTX 2070 (8 GB VRAM each) |
| **CPU Subsystem** | 2 Sockets × 20 Cores × 2 Threads = **80 Threads** |
| **Total Host RAM** | **256,000 MB (~256 GB / 250 GiB)** |
| **Slurm Partitions** | `ROOT`, `STUD`, `STUDL`, `2070` |

---

## 3. Remote Python & Deep Learning Stack

All experiments run inside the project virtual environment on Bigmama (`~/SRP26/.venv`), provisioned and locked with `uv`:

### Core Runtime
- **Python**: `3.12.3` (`/usr/bin/python3`)
- **Package Manager**: `uv` (`~/.local/bin/uv`)

### Deep Learning & Acceleration
- **PyTorch**: `torch 2.11.0`
- **Triton**: `3.6.0`
- **Hugging Face Accelerate**: `1.15.0`
- **CUDA Runtime**: `13.0.2` (`nvidia-cuda-runtime 13.0.96`, `nvidia-cublas 13.1.0.3`)
- **cuDNN**: `nvidia-cudnn-cu13 9.19.0.56`

### Time Series Foundation Models
- **Chronos (Amazon)**: `chronos-forecasting 2.3.2`
- **TimesFM (Google)**: `timesfm 3.0.2`
- **Granite-TSFM (IBM)**: `granite-tsfm 0.3.12`
- **Transformers**: `5.22.2`
- **PEFT**: `0.21.2`
- **Datasets**: `5.1.0`

### Scientific & Wavelet Signal Processing
- **PyTorch Wavelet Toolbox**: `ptwt 1.0.1`
- **PyWavelets**: `pywavelets 1.10.0`
- **SciPy**: `1.18.1`
- **NumPy**: `2.5.4`
- **Scikit-learn**: `1.7.2`
- **Pandas**: `3.1.1`

---

## 4. Verification Commands

To re-verify any of these specs directly on the cluster from your local terminal:

```powershell
# 1. Query node state and memory from Slurm controller:
uv run python bigmama/run_cluster_job.py --exec "scontrol show node gpu-211"

# 2. Query live GPU specs from the node:
uv run python bigmama/run_cluster_job.py --exec "srun -p STUD -w gpu-211 --gres=gpu:1 nvidia-smi"

# 3. Query CPU specs from the node:
uv run python bigmama/run_cluster_job.py --exec "srun -p STUD -w gpu-211 lscpu"

# 4. List remote environment packages:
uv run python bigmama/run_cluster_job.py --exec "uv pip list"
```

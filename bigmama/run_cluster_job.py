"""
Remote Job Manager for Bigmama Slurm Cluster
===========================================
Submits, inspects, and monitors Slurm jobs on Bigmama directly from your terminal.

Usage:
    uv run python bigmama/run_cluster_job.py --queue
    uv run python bigmama/run_cluster_job.py --submit slurm_scripts/run_bm_ETTh1_H96_s2024.sh
    uv run python bigmama/run_cluster_job.py --exec "sinfo -p STUD"
"""

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bigmama_deployment.remote_exec import run_remote


def main():
    parser = argparse.ArgumentParser(description="Manage Slurm jobs on Bigmama cluster.")
    parser.add_argument("--queue", action="store_true", help="View your queued and running jobs.")
    parser.add_argument("--sinfo", action="store_true", help="View status of the student partitions.")
    parser.add_argument("--submit", type=str, default="", help="Relative path of script in ~/SRP26/ to submit with sbatch.")
    parser.add_argument("--exec", type=str, default="", help="Arbitrary command to execute in ~/SRP26 on Bigmama.")
    args = parser.parse_args()

    if args.queue:
        cmd = "squeue -u $USER"
    elif args.sinfo:
        cmd = "sinfo -p STUD,STUDL"
    elif args.submit:
        cmd = f"cd ~/SRP26 && sbatch {args.submit}"
    elif args.exec:
        cmd = f"cd ~/SRP26 && {args.exec}"
    else:
        parser.print_help()
        return

    run_remote(cmd)


if __name__ == "__main__":
    main()


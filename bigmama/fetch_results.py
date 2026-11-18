"""
Fetch Results Tool for Bigmama Cluster
=======================================
Downloads generated experiment metrics (JSON/CSV) and logs from
`~/SRP26/results/` and `~/SRP26/logs/` on Bigmama to local `bigmama/results/` and `bigmama/logs/`.
"""

import os
import sys
from pathlib import Path
import paramiko

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from bigmama_deployment.remote_exec import resolve_ssh_credentials

LOCAL_RESULTS_DIR = REPO_ROOT / "bigmama" / "results"
LOCAL_LOGS_DIR = REPO_ROOT / "bigmama" / "logs"


def fetch_remote_directory(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> int:
    local_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        entries = sftp.listdir_attr(remote_dir)
    except Exception as e:
        print(f"  [Notice] Could not list {remote_dir}: {e}")
        return 0

    for entry in entries:
        rem_path = f"{remote_dir}/{entry.filename}"
        loc_path = local_dir / entry.filename
        # Check if regular file
        if not (entry.st_mode & 0o040000):  # not a directory
            if not loc_path.exists() or loc_path.stat().st_size != entry.st_size:
                print(f"  Downloading: {entry.filename} ({entry.st_size / 1024:.1f} KB)")
                sftp.get(rem_path, str(loc_path))
                count += 1
    return count


def main():
    host, user, password = resolve_ssh_credentials()
    print(f"Connecting to {user}@{host} ...")

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=host, username=user, password=password, timeout=15)
    sftp = ssh.open_sftp()

    print(f"\n--- Fetching Results from ~/SRP26/results/ to {LOCAL_RESULTS_DIR} ---")
    n_res = fetch_remote_directory(sftp, "SRP26/results/cluster_results", LOCAL_RESULTS_DIR)

    local_ta_dir = REPO_ROOT / "results" / "transform_ablation"
    print(f"\n--- Fetching Transform Ablation from ~/SRP26/results/transform_ablation to {local_ta_dir} ---")
    n_ta = fetch_remote_directory(sftp, "SRP26/results/transform_ablation", local_ta_dir)

    print(f"\n--- Fetching Logs from ~/SRP26/logs/ to {LOCAL_LOGS_DIR} ---")
    n_logs = fetch_remote_directory(sftp, "SRP26/logs", LOCAL_LOGS_DIR)

    sftp.close()
    ssh.close()
    print(f"\nDone! Downloaded {n_res} cluster files, {n_ta} transform ablation files, and {n_logs} log files.")


if __name__ == "__main__":
    main()


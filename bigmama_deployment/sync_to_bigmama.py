"""
Automated Bigmama Cluster Sync & Deployment Tool (Paramiko SFTP)
================================================================
Transfers SRP26 deployment files, SLURM scripts, and benchmark datasets
directly to master.ismll.de without external SCP dependencies.
"""

import os
import sys
import glob
import time
import paramiko

def progress_callback(transferred, total, fname):
    percent = (transferred / total) * 100 if total > 0 else 0
    sys.stdout.write(f"\r  Uploading {fname}: {transferred / (1024*1024):.2f}MB / {total / (1024*1024):.2f}MB ({percent:.1f}%)")
    sys.stdout.flush()

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(base_dir)

    # Load credentials from environment or .env (never hardcoded)
    env_file = os.path.join(project_dir, ".env")
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
    if not host or not user:
        raise RuntimeError(
            "Missing cluster configuration. Copy .env.example to .env and set "
            "BIGMAMA_HOST and BIGMAMA_USER (plus BIGMAMA_PASS if not using SSH keys)."
        )
    remote_dir = (os.getenv("BIGMAMA_REMOTE_DIR") or env_vars.get("BIGMAMA_REMOTE_DIR", "~/SRP26")).replace(
        "~", f"/home/{user}"
    )

    print("=" * 80)
    print(f" AUTOMATED BIGMAMA DEPLOYMENT SYNC")
    print(f" Target: {user}@{host}:{remote_dir}")
    print("=" * 80)

    print(f"\n1. Connecting via SSH to {host} ...")
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(host, username=user, password=password, timeout=15)
        print("   [OK] SSH Authentication successful!")
    except Exception as e:
        print(f"   [ERROR] Connection failed: {e}")
        print("   Please verify eduVPN is connected.")
        return

    # Create remote directories
    dirs_to_make = [
        remote_dir,
        f"{remote_dir}/data",
        f"{remote_dir}/results/cluster_results",
        f"{remote_dir}/logs",
        f"{remote_dir}/slurm_scripts",
    ]
    mkdir_cmd = f"mkdir -p {' '.join(dirs_to_make)}"
    ssh.exec_command(mkdir_cmd)
    time.sleep(1)

    sftp = ssh.open_sftp()

    # 2. Upload deployment scripts
    print("\n2. Uploading deployment scripts and SLURM jobs...")
    deploy_files = [
        "run_bigmama_benchmark.py",
        "generate_bigmama_slurm_jobs.py",
        "setup_bigmama_env.sh",
        "submit_all_jobs.sh",
        "submit_array_job.sh",
        "job_parameters.txt",
        "aggregate_cluster_results.py",
        "DEPLOYMENT_GUIDE.md",
    ]
    for df in deploy_files:
        local_path = os.path.join(base_dir, df)
        if os.path.exists(local_path):
            rem_path = f"{remote_dir}/{df}"
            print(f"   -> {df}")
            sftp.put(local_path, rem_path)

    # Upload slurm_scripts
    local_slurm_dir = os.path.join(base_dir, "slurm_scripts")
    if os.path.exists(local_slurm_dir):
        sh_files = glob.glob(os.path.join(local_slurm_dir, "*.sh"))
        print(f"   -> Uploading {len(sh_files)} SLURM job scripts to slurm_scripts/ ...")
        for sh in sh_files:
            fname = os.path.basename(sh)
            sftp.put(sh, f"{remote_dir}/slurm_scripts/{fname}")

    # Set executable permissions
    ssh.exec_command(f"chmod +x {remote_dir}/*.sh {remote_dir}/slurm_scripts/*.sh")

    # 3. Upload datasets
    print("\n3. Uploading benchmark datasets to remote data/ ...")
    datasets = ["ETTh1.csv", "ETTh2.csv", "weather.csv", "traffic.csv", "electricity.csv"]
    for ds in datasets:
        local_ds = os.path.join(project_dir, "data_sets", ds)
        if os.path.exists(local_ds):
            rem_ds = f"{remote_dir}/data/{ds}"
            # Check if file exists remotely and size matches
            try:
                rem_stat = sftp.stat(rem_ds)
                if rem_stat.st_size == os.path.getsize(local_ds):
                    print(f"   [SKIP] {ds} already exists on remote with identical size ({rem_stat.st_size / (1024*1024):.2f}MB).")
                    continue
            except IOError:
                pass

            file_size_mb = os.path.getsize(local_ds) / (1024 * 1024)
            print(f"   -> Uploading {ds} ({file_size_mb:.2f} MB) ...")
            cb = lambda trans, tot, name=ds: progress_callback(trans, tot, name)
            sftp.put(local_ds, rem_ds, callback=cb)
            print(" [Done]")

    sftp.close()
    ssh.close()

    print("\n" + "=" * 80)
    print(" DEPLOYMENT SYNC COMPLETED SUCCESSFULLY!")
    print(f" Next steps on Bigmama:")
    print(f"   1. ssh {user}@{host}")
    print(f"   2. cd ~/SRP26")
    print(f"   3. bash setup_bigmama_env.sh")
    print(f"   4. sbatch submit_array_job.sh")
    print("=" * 80)

if __name__ == "__main__":
    main()

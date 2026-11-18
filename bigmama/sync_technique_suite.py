"""Sync the new forecasting techniques test suite and datasets to Bigmama cluster."""

import os
import sys
import glob
import time
import paramiko

def progress_callback(transferred, total, fname):
    pct = (transferred / total) * 100 if total > 0 else 0
    sys.stdout.write(f"\r  Uploading {fname}: {transferred / (1024*1024):.2f}MB / {total / (1024*1024):.2f}MB ({pct:.1f}%)")
    sys.stdout.flush()

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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

    print("=" * 80)
    print(f" SYNCING TECHNIQUE TEST SUITE TO BIGMAMA")
    print(f" Target: {user}@{host}:{remote_dir}")
    print("=" * 80)

    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=password, timeout=15)
    print("SSH connected.")

    # Create directories
    remote_dirs = [
        remote_dir,
        f"{remote_dir}/srp26",
        f"{remote_dir}/scripts",
        f"{remote_dir}/data",
        f"{remote_dir}/results/transform_ablation",
        f"{remote_dir}/results/transform_ablation/raw",
        f"{remote_dir}/slurm_scripts",
        f"{remote_dir}/logs",
    ]
    ssh.exec_command(f"mkdir -p {' '.join(remote_dirs)}")
    time.sleep(1)

    sftp = ssh.open_sftp()

    # 1. Upload root scripts
    root_files = [
        "experiment_transform_ablation.py",
        "metrics.py",
        "diagnostics.py",
    ]
    for rf in root_files:
        loc = os.path.join(repo_root, rf)
        if os.path.exists(loc):
            rem = f"{remote_dir}/{rf}"
            print(f" -> {rf}")
            sftp.put(loc, rem)

    # 2. Upload srp26 package
    print("\nUploading srp26 package:")
    srp26_files = glob.glob(os.path.join(repo_root, "srp26", "*.py"))
    for sf in srp26_files:
        fname = os.path.basename(sf)
        rem = f"{remote_dir}/srp26/{fname}"
        print(f" -> srp26/{fname}")
        sftp.put(sf, rem)

    # 3. Upload scripts/
    print("\nUploading scripts/ :")
    script_files = ["aggregate_transform_ablation.py", "run_dsof_sd.py"]
    for sc in script_files:
        loc = os.path.join(repo_root, "scripts", sc)
        if os.path.exists(loc):
            rem = f"{remote_dir}/scripts/{sc}"
            print(f" -> scripts/{sc}")
            sftp.put(loc, rem)

    # 4. Upload datasets: ETTm1, ETTm2 (and check others)
    print("\nChecking datasets in data/ :")
    for ds_name in ["ETTm1.csv", "ETTm2.csv"]:
        loc_ds = os.path.join(repo_root, "data_sets", ds_name)
        if os.path.exists(loc_ds):
            rem_ds = f"{remote_dir}/data/{ds_name}"
            try:
                stat = sftp.stat(rem_ds)
                if stat.st_size == os.path.getsize(loc_ds):
                    print(f" [SKIP] {ds_name} already exists ({stat.st_size / (1024*1024):.1f}MB)")
                    continue
            except IOError:
                pass
            print(f" -> Uploading {ds_name} ...")
            cb = lambda trans, tot, n=ds_name: progress_callback(trans, tot, n)
            sftp.put(loc_ds, rem_ds, callback=cb)
            print(" [Done]")

    sftp.close()
    ssh.close()
    print("\nSync completed successfully!")

if __name__ == "__main__":
    main()


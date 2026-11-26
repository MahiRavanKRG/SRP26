"""v2 Bigmama tooling: ship an exact git commit, build the locked env, submit, fetch.

    uv run python bigmama/v2_cluster.py sync      # upload `git archive HEAD` + datasets to ~/SRP26_v2
    uv run python bigmama/v2_cluster.py setup     # Python 3.13 + `uv sync --frozen` on the cluster
    uv run python bigmama/v2_cluster.py submit --jobs configs/v2/jobs_lrsel.jsonl --phase-name lrsel --shards 16
    uv run python bigmama/v2_cluster.py status
    uv run python bigmama/v2_cluster.py fetch     # results/v2/bigmama -> local results/v2/bigmama
    uv run python bigmama/v2_cluster.py exec "<command>"

Credentials come from the git-ignored .env via bigmama_deployment.remote_exec.
Refuses to sync from a dirty tree, so every cluster result maps to one commit.
"""

import argparse
import io
import posixpath
import stat
import subprocess
import sys
import tarfile
from pathlib import Path

import paramiko

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from bigmama_deployment.remote_exec import resolve_ssh_credentials  # noqa: E402

REMOTE_DIR = "SRP26_v2"
OLD_DATA = "SRP26/data"
DATA_FILES = ["ETTh1.csv", "ETTh2.csv", "ETTm1.csv", "ETTm2.csv", "weather.csv",
              "electricity.csv", "traffic.csv", "solar_AL.txt"]
UV = "$HOME/.local/bin/uv"


def connect() -> paramiko.SSHClient:
    host, user, password = resolve_ssh_credentials()
    ssh = paramiko.SSHClient()
    ssh.load_system_host_keys()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, username=user, password=password, timeout=20)
    return ssh


def sh(ssh: paramiko.SSHClient, cmd: str, check: bool = True) -> str:
    _, out, err = ssh.exec_command(f"bash -lc {cmd!r}")
    code = out.channel.recv_exit_status()
    o, e = out.read().decode(), err.read().decode()
    if o:
        print(o, end="")
    if e:
        print(e, end="", file=sys.stderr)
    if check and code != 0:
        raise SystemExit(f"remote command failed ({code}): {cmd}")
    return o


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO).decode().strip()


def cmd_sync(ssh: paramiko.SSHClient) -> None:
    if git("status", "--porcelain", "--", "srp26", "models", "configs", "pyproject.toml", "uv.lock", "data_sets"):
        raise SystemExit("Refusing to sync: commit code/config changes first (results must map to one commit).")
    commit = git("rev-parse", "HEAD")
    archive = subprocess.check_output(["git", "archive", "--format=tar", "HEAD"], cwd=REPO)
    home = sh(ssh, "echo $HOME").strip()
    root = f"{home}/{REMOTE_DIR}"
    sh(ssh, f"mkdir -p {root}/data_sets {root}/logs {root}/results/v2")
    sftp = ssh.open_sftp()
    sftp.putfo(io.BytesIO(archive), f"{root}/code.tar")
    sh(ssh, f"cd {root} && tar -xf code.tar && rm code.tar && echo {commit} > COMMIT")
    for name in DATA_FILES:  # reuse the verified copies already on the cluster where possible
        remote = f"{root}/data_sets/{name}"
        old = f"{home}/{OLD_DATA}/{name}"
        if sh(ssh, f"test -e {remote} && echo yes || true").strip() == "yes":
            continue
        if sh(ssh, f"test -e {old} && echo yes || true").strip() == "yes":
            sh(ssh, f"ln -s {old} {remote}")
        else:
            print(f"uploading {name} ...")
            sftp.put(str(REPO / "data_sets" / name), remote)
    sftp.close()
    sh(ssh, f"cd {root}/data_sets && sha256sum -c CHECKSUMS.sha256")
    print(f"synced commit {commit} to ~/{REMOTE_DIR}")


def cmd_setup_ttm(ssh: paramiko.SSHClient) -> None:
    """Separate locked env for TTM only (envs/ttm): torch 2.11 + granite-tsfm 0.3.12."""
    sh(ssh, f"cd ~/{REMOTE_DIR}/envs/ttm && {UV} sync --frozen --python 3.13")
    sh(ssh, f"cd ~/{REMOTE_DIR} && envs/ttm/.venv/bin/python -c "
            "\"import torch, tsfm_public; print(torch.__version__, torch.version.cuda)\"")


def cmd_setup(ssh: paramiko.SSHClient) -> None:
    sh(ssh, f"{UV} python install 3.13")
    sh(ssh, f"cd ~/{REMOTE_DIR} && {UV} sync --frozen --all-extras --python 3.13")
    sh(ssh, f"cd ~/{REMOTE_DIR} && .venv/bin/python -c "
            "\"import sys, torch; print(sys.version.split()[0], torch.__version__, torch.version.cuda, torch.cuda.get_arch_list())\"")


SLURM = """#!/usr/bin/env bash
#SBATCH --job-name=v2_{name}
#SBATCH --output=logs/v2_{name}_%A_%a.out
#SBATCH --error=logs/v2_{name}_%A_%a.err
#SBATCH --partition=STUD
#SBATCH --gres=gpu:{gpu}:1
#SBATCH --mem={mem}
#SBATCH --cpus-per-task=4
#SBATCH --time={time}
#SBATCH --array=0-{last}%{concurrent}

cd $HOME/{root}
export SRP26_GIT_COMMIT=$(cat COMMIT)
export SRP26_GIT_DIRTY=0
export SRP26_HOST_LABEL=bigmama
export SRP26_DATA_DIR=$HOME/{root}/data_sets
export CUBLAS_WORKSPACE_CONFIG=:4096:8
echo "host=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader -i $CUDA_VISIBLE_DEVICES) commit=$SRP26_GIT_COMMIT"
{python} -m {module} --set bigmama --jobs {jobs} --shard $SLURM_ARRAY_TASK_ID --n-shards {shards}
"""


def cmd_submit(ssh: paramiko.SSHClient, jobs: str, name: str, shards: int, concurrent: int,
               time: str, mem: str, python: str = ".venv/bin/python", module: str = "srp26.bench.run", gpu: str = "6000") -> None:
    script = SLURM.format(name=name, mem=mem, time=time, last=shards - 1, concurrent=concurrent,
                          root=REMOTE_DIR, jobs=jobs, shards=shards, python=python, module=module, gpu=gpu)
    sftp = ssh.open_sftp()
    path = f"{sh(ssh, 'echo $HOME').strip()}/{REMOTE_DIR}/slurm_v2_{name}.sh"
    with sftp.open(path, "w") as f:
        f.write(script)
    sftp.close()
    sh(ssh, f"cd ~/{REMOTE_DIR} && sbatch slurm_v2_{name}.sh")


def cmd_fetch(ssh: paramiko.SSHClient) -> None:
    sftp = ssh.open_sftp()
    home = sh(ssh, "echo $HOME").strip()
    remote_root = f"{home}/{REMOTE_DIR}/results/v2/bigmama"
    local_root = REPO / "results" / "v2" / "bigmama"
    new = 0

    def walk(rdir: str, ldir: Path) -> None:
        nonlocal new
        try:
            entries = sftp.listdir_attr(rdir)
        except FileNotFoundError:
            return
        for e in entries:
            rpath = posixpath.join(rdir, e.filename)
            lpath = ldir / e.filename
            if stat.S_ISDIR(e.st_mode):
                walk(rpath, lpath)
            elif not e.filename.endswith(".tmp") and not lpath.exists():
                lpath.parent.mkdir(parents=True, exist_ok=True)
                sftp.get(rpath, str(lpath))
                new += 1

    walk(remote_root, local_root)
    sftp.close()
    print(f"fetched {new} new files into {local_root.relative_to(REPO)}")


def main() -> None:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("sync")
    sub.add_parser("setup")
    sub.add_parser("setup-ttm")
    s = sub.add_parser("submit")
    s.add_argument("--jobs", required=True)
    s.add_argument("--phase-name", required=True)
    s.add_argument("--shards", type=int, default=16)
    s.add_argument("--concurrent", type=int, default=8)
    s.add_argument("--time", default="24:00:00")
    s.add_argument("--mem", default="32G")
    s.add_argument("--python", default=".venv/bin/python", help="e.g. envs/ttm/.venv/bin/python for TTM jobs")
    s.add_argument("--module", default="srp26.bench.run", help="e.g. srp26.bench.foundation for TTM jobs")
    s.add_argument("--gpu", default="6000", choices=["6000", "2070"], help="GPU type to pin the array to")
    sub.add_parser("status")
    sub.add_parser("fetch")
    e = sub.add_parser("exec")
    e.add_argument("command")
    a = p.parse_args()

    ssh = connect()
    try:
        if a.cmd == "sync":
            cmd_sync(ssh)
        elif a.cmd == "setup":
            cmd_setup(ssh)
        elif a.cmd == "setup-ttm":
            cmd_setup_ttm(ssh)
        elif a.cmd == "submit":
            cmd_submit(ssh, a.jobs, a.phase_name, a.shards, a.concurrent, a.time, a.mem, a.python, a.module, a.gpu)
        elif a.cmd == "status":
            sh(ssh, "squeue -u $USER -o '%.12i %.20j %.8T %.10M %.6D %R'", check=False)
        elif a.cmd == "fetch":
            cmd_fetch(ssh)
        else:
            sh(ssh, f"cd ~/{REMOTE_DIR} && {a.command}", check=False)
    finally:
        ssh.close()


if __name__ == "__main__":
    main()

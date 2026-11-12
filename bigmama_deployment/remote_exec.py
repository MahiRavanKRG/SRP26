"""Remote SSH command executor for the Bigmama Slurm cluster."""

import getpass
import os
import sys
from typing import Dict, Optional, Tuple

import paramiko


def load_env_config(env_path: Optional[str] = None) -> Dict[str, str]:
    """Load key=value pairs from a local ``.env`` file when present."""
    path = env_path or os.path.join(os.path.dirname(__file__), "..", ".env")
    env_vars: Dict[str, str] = {}
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    env_vars[key.strip()] = val.strip().strip("'\"")
    return env_vars


def resolve_ssh_credentials(interactive: bool = False) -> Tuple[str, str, Optional[str]]:
    """
    Resolve ``(host, user, password)`` from environment variables or ``.env``.

    Raises ``RuntimeError`` when ``BIGMAMA_HOST`` or ``BIGMAMA_USER`` is missing.
    """
    env_vars = load_env_config()
    host = os.getenv("BIGMAMA_HOST") or env_vars.get("BIGMAMA_HOST", "")
    user = os.getenv("BIGMAMA_USER") or env_vars.get("BIGMAMA_USER", "")
    password = os.getenv("BIGMAMA_PASS") or env_vars.get("BIGMAMA_PASS")

    if not host or not user:
        raise RuntimeError(
            "Missing cluster configuration. Copy .env.example to .env and set "
            "BIGMAMA_HOST and BIGMAMA_USER (plus BIGMAMA_PASS if not using SSH keys)."
        )
    if not password and interactive:
        password = getpass.getpass(f"SSH password for {user}@{host} (leave empty for SSH key): ") or None
    return host, user, password


def run_remote(command: str, interactive: bool = False) -> Tuple[str, str]:
    """Execute ``command`` on the configured Slurm login node via Paramiko."""
    host, user, password = resolve_ssh_credentials(interactive=interactive)
    print(f"Connecting to {user}@{host}...", flush=True)
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(host, username=user, password=password, timeout=15)
        print(f"Executing:\n{command}\n" + "-" * 60, flush=True)
        _, stdout, stderr = ssh.exec_command(f"bash -l -c {repr(command)}")
        out = stdout.read().decode("utf-8")
        err = stderr.read().decode("utf-8")
        if out:
            print("=== STDOUT ===", flush=True)
            print(out, flush=True)
        if err:
            print("=== STDERR ===", flush=True)
            print(err, flush=True)
        return out, err
    finally:
        ssh.close()


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--file":
        with open(sys.argv[2], "r", encoding="utf-8") as handle:
            cmd = handle.read()
    elif len(sys.argv) > 1:
        cmd = " ".join(sys.argv[1:])
    else:
        cmd = "hostname && nvidia-smi"
    run_remote(cmd, interactive=sys.stdin.isatty())

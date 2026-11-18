#!/usr/bin/env python3
import sys
try:
    from huggingface_hub import snapshot_download
    print("Downloading google/timesfm-2.5-200m-pytorch to ~/.cache/huggingface ...")
    path = snapshot_download("google/timesfm-2.5-200m-pytorch")
    print(f"Success! Model cached at: {path}")
except Exception as e:
    print(f"Error: {e}", file=sys.stderr)
    sys.exit(1)


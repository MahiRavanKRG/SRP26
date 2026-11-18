#!/usr/bin/env python3
import sys
from huggingface_hub import snapshot_download

MODELS = [
    "google/timesfm-2.5-200m-pytorch",
    "amazon/chronos-bolt-base",
    "ibm-granite/granite-timeseries-ttm-r1"
]

print("=== Pre-caching all benchmark models on Bigmama master ===")
for model in MODELS:
    print(f"\nDownloading: {model} ...")
    try:
        path = snapshot_download(model)
        print(f"  [OK] Cached at: {path}")
    except Exception as e:
        print(f"  [FAIL] Error caching {model}: {e}", file=sys.stderr)
        sys.exit(1)

print("\n=== All models successfully cached in ~/.cache/huggingface ===")


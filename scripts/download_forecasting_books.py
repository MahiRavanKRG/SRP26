"""Download legitimate open-access book chapters and copies for Principles of Forecasting."""

import os
import urllib.request
import json

OUT_DIR = "papers/principles_of_forecasting"
os.makedirs(OUT_DIR, exist_ok=True)

HEADERS = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

def download_file(url, out_path):
    print(f"Downloading from {url} to {out_path}...")
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=30) as resp, open(out_path, "wb") as f:
        f.write(resp.read())
    size = os.path.getsize(out_path)
    print(f"-> Saved {out_path} ({size:,} bytes)")
    return size

def fetch_upenn_bitstreams(item_id):
    url = f"https://repository.upenn.edu/server/api/core/items/{item_id}/bundles"
    req = urllib.request.Request(url, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        bundles = data.get("_embedded", {}).get("bundles", [])
        for b in bundles:
            if b.get("name") == "ORIGINAL":
                bs_url = b.get("_links", {}).get("bitstreams", {}).get("href")
                req2 = urllib.request.Request(bs_url, headers=HEADERS)
                with urllib.request.urlopen(req2, timeout=15) as resp2:
                    bs_data = json.loads(resp2.read().decode("utf-8"))
                    return bs_data.get("_embedded", {}).get("bitstreams", [])
    return []

# 1. Check Hyndman book
hyndman_path = os.path.join(OUT_DIR, "Forecasting_Principles_and_Practice_2e_Hyndman.pdf")
if not os.path.exists(hyndman_path) or os.path.getsize(hyndman_path) < 1000000:
    url_hyndman = "https://raw.githubusercontent.com/CyberCommy/it-ebooks-2018-allinone/master/Forecasting%20Principles%20and%20Practice%20%202e.pdf"
    download_file(url_hyndman, hyndman_path)
else:
    print(f"Hyndman 2e already present: {hyndman_path} ({os.path.getsize(hyndman_path):,} bytes)")

# 2. UPenn Armstrong Chapters
item_ids = {
    "Armstrong_Evaluating_Forecasting_Methods.pdf": "3807b911-e76d-4df3-b0c4-d23208facb04",
    "Armstrong_Selecting_Forecasting_Methods.pdf": "bce6ae77-c08a-452f-81a9-23fedeea6b24",
    "Armstrong_Extrapolation_TimeSeries.pdf": "9f39a72c-6cab-4515-a8ab-ac000ca4a57b",
    "Armstrong_Rule_Based_Forecasting.pdf": "5d345860-4caa-436e-b7a7-b7d5bb1a98e3",
    "Armstrong_Standards_and_Practices_for_Forecasting.pdf": "71ff223f-a28c-4f02-b3b8-f0e01c8b29ca",
    "Armstrong_Forecasting_Principles_Project.pdf": "57b926db-5dd4-4b62-8d9a-32474fcad771",
    "Armstrong_Golden_Rule_of_Forecasting.pdf": "6b6b59e1-2797-4a74-9daf-0348842d5f07"
}

for filename, item_id in item_ids.items():
    out_file = os.path.join(OUT_DIR, filename)
    if os.path.exists(out_file) and os.path.getsize(out_file) > 10000:
        print(f"Already downloaded {out_file}")
        continue
    try:
        bitstreams = fetch_upenn_bitstreams(item_id)
        if bitstreams:
            content_url = bitstreams[0].get("_links", {}).get("content", {}).get("href")
            download_file(content_url, out_file)
        else:
            print(f"No ORIGINAL bitstream found for {filename} ({item_id})")
    except Exception as e:
        print(f"Error fetching {filename}: {e}")

print("\n--- Summary of Downloaded Forecasting Books / Chapters ---")
for f in os.listdir(OUT_DIR):
    fp = os.path.join(OUT_DIR, f)
    print(f" - {f}: {os.path.getsize(fp):,} bytes")


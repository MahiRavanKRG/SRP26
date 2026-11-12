import os
import requests
import pandas as pd

def download_ett_data():
    url = "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv"
    save_path = "ETTh1.csv"
    
    if not os.path.exists(save_path):
        print(f"Downloading {save_path}...")
        response = requests.get(url)
        with open(save_path, "wb") as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print(f"{save_path} already exists.")

    df = pd.read_csv(save_path)
    print(f"Dataset Shape: {df.shape}")
    print(df.head())

if __name__ == "__main__":
    download_ett_data()

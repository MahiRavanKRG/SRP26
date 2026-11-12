import os
import subprocess
import sys
import json
import argparse
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser(description="Run Full Baseline Benchmarks")
    parser.add_argument('--dataset', type=str, default='ETTh1.csv')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seq_len', type=int, default=336)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--models', nargs='+', default=['dlinear', 'freq', 'wavelet'])
    parser.add_argument('--horizons', nargs='+', type=int, default=[96, 192, 336, 720])
    return parser.parse_args()

def run_all():
    args = parse_args()
    models = args.models
    horizons = args.horizons
    dataset = args.dataset
    seq_len = args.seq_len
    epochs = max(3, args.epochs)
    batch_size = args.batch_size
    lr = args.lr
    
    total_runs = len(models) * len(horizons)
    run_idx = 1
    
    print(f"\n==================================================================")
    print(f"=== Starting Autonomous Baseline Benchmark Execution ({total_runs} runs) ===")
    print(f"Models: {models} | Horizons: {horizons}")
    print(f"Dataset: {dataset} | Seq Len: {seq_len} | Epochs: {epochs}")
    print(f"==================================================================\n")
    
    for model in models:
        for h in horizons:
            print(f"\n[{run_idx}/{total_runs}] Running {model} with Horizon H={h} (epochs={epochs})...")
            cmd = [
                sys.executable, 'experiment.py',
                '--dataset', dataset,
                '--model', model,
                '--seq_len', str(seq_len),
                '--pred_len', str(h),
                '--epochs', str(epochs),
                '--batch_size', str(batch_size),
                '--lr', str(lr)
            ]
            res = subprocess.run(cmd, capture_output=True, text=True)
            if res.returncode != 0:
                print(f"Error running {model} H={h}:")
                print(res.stderr)
            else:
                # Print summary output lines
                out_lines = [l for l in res.stdout.strip().split('\n') if 'Results saved to' in l or 'Loss:' in l]
                print(f"Completed {model} H={h} successfully.")
                if out_lines:
                    print("   " + "\n   ".join(out_lines[-2:]))
            run_idx += 1
            
    print("\n=== All runs completed. Compiling summary report... ===")
    compile_summary(models, horizons, dataset)

def compile_summary(models, horizons, dataset):
    prefix = os.path.basename(dataset).split('.')[0]
    records = []
    
    for model in models:
        for h in horizons:
            filepath = f"results/{model}_{prefix}_H{h}.json"
            if os.path.exists(filepath):
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    records.append(data)
                    
    if not records:
        print("No results found.")
        return
        
    df = pd.DataFrame(records)
    csv_path = "results/spectral_benchmark_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"Summary CSV saved to {csv_path}")
    
    # Generate Markdown Report
    md_report = "# Spectral Evaluation & Band-Wise Error Decomposition Summary\n\n"
    md_report += f"**Dataset:** `{dataset}` | **Sequence Length:** 336 | **Horizons:** {horizons}\n\n"
    
    md_report += "## 1. Overall Forecast Performance (MSE / MAE)\n\n"
    pivot_mse = df.pivot(index='pred_len', columns='model', values='mse')
    pivot_mae = df.pivot(index='pred_len', columns='model', values='mae')
    
    md_report += "### Mean Squared Error (MSE)\n\n"
    md_report += pivot_mse.to_markdown() + "\n\n"
    
    md_report += "### Mean Absolute Error (MAE)\n\n"
    md_report += pivot_mae.to_markdown() + "\n\n"
    
    md_report += "## 2. Band-Wise Error Decomposition (Parseval-Exact FFT)\n\n"
    
    for band in ['low', 'mid', 'high']:
        md_report += f"### Band: {band.upper()} Frequency\n\n"
        sub_df = df.pivot(index='pred_len', columns='model', values=f'band_mse_{band}')
        md_report += f"**Absolute MSE ({band}):**\n\n" + sub_df.to_markdown() + "\n\n"
        
        if f'band_fvu_{band}' in df.columns:
            fvu_df = df.pivot(index='pred_len', columns='model', values=f'band_fvu_{band}')
            md_report += f"**Fraction of Variance Unexplained (FVU - {band}):**\n\n" + fvu_df.to_markdown() + "\n\n"
            
        if f'energy_ratio_{band}' in df.columns:
            er_df = df.pivot(index='pred_len', columns='model', values=f'energy_ratio_{band}')
            md_report += f"**Energy Ratio ({band}):**\n\n" + er_df.to_markdown() + "\n\n"
            
    md_report += "## 3. Spectral Distortion & Over-Smoothing (LSD & Coherence)\n\n"
    if 'lsd' in df.columns:
        lsd_df = df.pivot(index='pred_len', columns='model', values='lsd')
        md_report += "**Log-Spectral Distance (LSD in dB - Lower is better):**\n\n" + lsd_df.to_markdown() + "\n\n"
        
    if 'mean_coherence' in df.columns:
        coh_df = df.pivot(index='pred_len', columns='model', values='mean_coherence')
        md_report += "**Mean Spectral Coherence:**\n\n" + coh_df.to_markdown() + "\n\n"
        
    md_report += "## 4. Failure Window Diagnostics (Sliding Window Anomalies)\n\n"
    if 'failure_windows_flagged' in df.columns:
        fail_df = df.pivot(index='pred_len', columns='model', values='failure_windows_flagged')
        md_report += "**Flagged Failure Windows:**\n\n" + fail_df.to_markdown() + "\n\n"
        
    report_file = "results/spectral_benchmark_summary.md"
    with open(report_file, 'w') as f:
        f.write(md_report)
    print(f"Markdown report saved to {report_file}")

if __name__ == '__main__':
    run_all()

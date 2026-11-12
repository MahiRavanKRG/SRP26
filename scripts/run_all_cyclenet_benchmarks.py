import os
import sys
import subprocess
import time
import pandas as pd
import json

def run_benchmarks():
    datasets = ['ETTh1.csv', 'ETTh2.csv', 'ETTm1.csv', 'ETTm2.csv', 'weather.csv']
    horizons = [96, 192, 336, 720]
    model_type = 'linear'
    seq_len = 336
    epochs = 25
    patience = 5
    batch_size = 32
    lr = 1e-3
    seed = 2024
    
    total_tasks = len(datasets) * len(horizons)
    print(f"\n==================================================================")
    print(f"CycleNet-Pro Autonomous Benchmark Suite: {total_tasks} Tasks")
    print(f"Datasets: {datasets}")
    print(f"Horizons: {horizons}")
    print(f"Lookback L: {seq_len} | Epochs: {epochs} (Patience: {patience})")
    print(f"==================================================================\n")
    
    start_all = time.time()
    task_idx = 1
    
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) if os.path.basename(os.path.dirname(os.path.abspath(__file__))) == 'scripts' else os.path.abspath('.')
    exp_script = os.path.join(root_dir, 'experiment_cyclenet.py')
    
    for dataset in datasets:
        for h in horizons:
            d_name = dataset.split('.')[0]
            print(f"[{task_idx}/{total_tasks}] Launching CycleNet ({model_type}) on {d_name} (H={h})...")
            cmd = [
                sys.executable, exp_script,
                '--dataset', dataset,
                '--model_type', model_type,
                '--seq_len', str(seq_len),
                '--pred_len', str(h),
                '--epochs', str(epochs),
                '--patience', str(patience),
                '--batch_size', str(batch_size),
                '--lr', str(lr),
                '--seed', str(seed)
            ]
            
            t0 = time.time()
            res = subprocess.run(cmd, cwd=root_dir, capture_output=True, text=True)
            elapsed = time.time() - t0
            
            if res.returncode != 0:
                print(f"FAILED {d_name} H={h} (in {elapsed:.1f}s):")
                print(res.stderr)
            else:
                # Extract summary line from stdout
                lines = res.stdout.strip().split('\n')
                result_lines = [l for l in lines if 'MSE:' in l or 'Mid-Band MSE:' in l]
                print(f"COMPLETED {d_name} H={h} in {elapsed:.1f}s")
                for rl in result_lines:
                    print(f"   -> {rl}")
                    
            task_idx += 1
            
    total_elapsed = time.time() - start_all
    print(f"\nAll {total_tasks} tasks completed in {total_elapsed/60:.2f} minutes.")
    
    # Generate Formatted Markdown Report
    summary_csv = "results/cyclenet/cyclenet_benchmark_summary.csv"
    if os.path.exists(summary_csv):
        df = pd.read_csv(summary_csv)
        report_md = "# CycleNet-Pro Standardized Benchmark Report\n\n"
        report_md += f"**Lookback Window:** {seq_len} | **Optimizer:** Adam (lr={lr}) | **Patience:** {patience} | **Seed:** {seed}\n\n"
        
        report_md += "## 1. Overall Predictive Accuracy (MSE & MAE)\n\n"
        p_mse = df.pivot(index='dataset', columns='pred_len', values='mse')
        p_mae = df.pivot(index='dataset', columns='pred_len', values='mae')
        
        report_md += "### Mean Squared Error (MSE)\n\n" + p_mse.to_markdown() + "\n\n"
        report_md += "### Mean Absolute Error (MAE)\n\n" + p_mae.to_markdown() + "\n\n"
        
        report_md += "## 2. Exact Parseval Subband Decompositions\n\n"
        p_mid_mse = df.pivot(index='dataset', columns='pred_len', values='band_mse_mid')
        p_mid_fvu = df.pivot(index='dataset', columns='pred_len', values='band_fvu_mid')
        p_high_fvu = df.pivot(index='dataset', columns='pred_len', values='band_fvu_high')
        
        report_md += "### Mid-Band MSE (Periodic Harmonic Error)\n\n" + p_mid_mse.to_markdown() + "\n\n"
        report_md += "### Mid-Band FVU (Fraction of Variance Unexplained - lower is better)\n\n" + p_mid_fvu.to_markdown() + "\n\n"
        report_md += "### High-Band FVU (High-Frequency Noise Tracking)\n\n" + p_high_fvu.to_markdown() + "\n\n"
        
        report_md += "## 3. Spectral Distortion & Failure Window Tracking\n\n"
        p_lsd = df.pivot(index='dataset', columns='pred_len', values='lsd')
        p_fail = df.pivot(index='dataset', columns='pred_len', values='failure_windows_flagged')
        
        report_md += "### Log-Spectral Distance (LSD in dB)\n\n" + p_lsd.to_markdown() + "\n\n"
        report_md += "### Flagged Failure Windows\n\n" + p_fail.to_markdown() + "\n\n"
        
        md_path = "results/cyclenet/cyclenet_benchmark_summary.md"
        with open(md_path, 'w') as f:
            f.write(report_md)
        print(f"Markdown report compiled to: {md_path}")

if __name__ == '__main__':
    run_benchmarks()

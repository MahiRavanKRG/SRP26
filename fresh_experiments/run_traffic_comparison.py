"""
Traffic Benchmark Comparison Runner (CycleNet vs. TwoStageFourierPatchAttentionNet)
===================================================================================
Executes sequential evaluation on the high-dimensional traffic.csv benchmark (862 channels):
- Standardized Chronological Split (70% Train / 10% Val / 20% Test)
- Lookback L=336, Horizon H=96, Batch Size=16
- Precision: 16-mixed (AMP)
- Models:
  1. CycleNet (Primary Proposed Periodic Queue Model, W=24)
  2. TwoStageFourierPatchAttentionNet (Decoupled Fourier Harmonic Base + PatchTST Attention)
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from train_lightning import run_single_experiment


def parse_args():
    parser = argparse.ArgumentParser(description="Traffic Benchmark: CycleNet vs. Fourier-PatchAttention")
    parser.add_argument("--models", nargs="+", default=["cyclenet", "patch_attention"],
                        help="List of models to benchmark on traffic")
    parser.add_argument("--dataset", type=str, default="traffic.csv", help="Traffic dataset filename")
    parser.add_argument("--seq_len", type=int, default=336, help="Lookback window length")
    parser.add_argument("--pred_len", type=int, default=96, help="Forecast horizon length")
    parser.add_argument("--period_w", type=int, default=24, help="Periodic queue length W for CycleNet")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (16 for high-channel traffic)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--max_epochs", type=int, default=30, help="Maximum epochs")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--precision", type=str, default="16-mixed", help="Precision: 16-mixed or 32")
    parser.add_argument("--wandb_project", type=str, default="SRP26-LTSF-Sandbox", help="W&B Project")
    parser.add_argument("--seed", type=int, default=2024, help="Random seed")
    return parser.parse_args()


def format_markdown_table(results_list):
    md = "# Traffic Benchmark Results: CycleNet vs. Fourier-PatchAttention\n\n"
    md += f"**Execution Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  \n"
    md += f"**Dataset:** Traffic (`traffic.csv`, 862 variables)  \n"
    md += f"**Lookback ($L$):** 336 | **Horizon ($H$):** 96 | **Batch Size:** 16 | **Precision:** 16-mixed (AMP)  \n"
    md += "**Protocol:** Standard 70/10/20 Continuous Split (Train-only standardization, lookback prefix preserved)  \n\n"

    md += "| Model | Parameters | Training Time | Test MSE | Test MAE | Mid-Band Error | Mid-Band FVU | Log-Spectral Dist (LSD) |\n"
    md += "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n"

    for r in results_list:
        md += f"| **{r['model']}** | {r['params']:,} | {r['train_time_sec']}s | **{r['test_mse']:.4f}** | {r['test_mae']:.4f} | {r['mse_mid']:.5f} | {r['fvu_mid']:.4f} | {r['log_spectral_distance']:.4f} |\n"

    md += "\n\n### Analytical Takeaways & Comparative Insights\n"
    if len(results_list) >= 2:
        c_res = next((r for r in results_list if 'CycleNet' in r['model']), None)
        f_res = next((r for r in results_list if 'Fourier' in r['model'] or 'Patch' in r['model']), None)
        if c_res and f_res:
            mse_diff = (f_res['test_mse'] - c_res['test_mse']) / c_res['test_mse'] * 100
            time_diff = f_res['train_time_sec'] / max(1e-3, c_res['train_time_sec'])
            md += f"1. **Accuracy Comparison**: CycleNet Test MSE = {c_res['test_mse']:.4f} vs. Fourier-PatchAttention Test MSE = {f_res['test_mse']:.4f} ({abs(mse_diff):.2f}% {'lower' if mse_diff > 0 else 'higher'} for CycleNet).\n"
            md += f"2. **Computational Footprint**: CycleNet utilizes {c_res['params']:,} parameters vs. Fourier-PatchAttention's {f_res['params']:,} parameters ({f_res['params']/c_res['params']:.1f}x larger).\n"
            md += f"3. **Training Efficiency**: CycleNet trained in {c_res['train_time_sec']}s vs. Fourier-PatchAttention's {f_res['train_time_sec']}s ({time_diff:.1f}x runtime ratio).\n"
            md += f"4. **Spectral Fidelity**: Mid-Band FVU is {c_res['fvu_mid']:.4f} for CycleNet vs. {f_res['fvu_mid']:.4f} for Fourier-PatchAttention.\n"

    return md


def main():
    args = parse_args()
    results_dir = os.path.join(current_dir, "results")
    os.makedirs(results_dir, exist_ok=True)

    print("=" * 80)
    print(f"LAUNCHING TRAFFIC BENCHMARK COMPARISON ({len(args.models)} Models)")
    print(f"Dataset:   {args.dataset}")
    print(f"Models:    {args.models}")
    print(f"Config:    L={args.seq_len}, H={args.pred_len}, Batch Size={args.batch_size}, Precision={args.precision}, Seed={args.seed}")
    print("=" * 80)

    results_list = []
    total_start = time.time()

    for idx, model_name in enumerate(args.models, 1):
        print(f"\n[{idx}/{len(args.models)}] Running Model: {model_name} on {args.dataset}...")
        try:
            res = run_single_experiment(
                model=model_name,
                dataset=args.dataset,
                seq_len=args.seq_len,
                pred_len=args.pred_len,
                period_w=args.period_w,
                batch_size=args.batch_size,
                lr=args.lr,
                max_epochs=args.max_epochs,
                patience=args.patience,
                precision=args.precision,
                wandb_project=args.wandb_project,
                seed=args.seed
            )
            results_list.append(res)
            print(f"[{idx}/{len(args.models)}] Completed {model_name}: Test MSE = {res['test_mse']:.4f}, Test MAE = {res['test_mae']:.4f}")
        except Exception as e:
            print(f"ERROR executing {model_name}: {e}")
            import traceback
            traceback.print_exc()

    total_time = round(time.time() - total_start, 2)
    print("\n" + "=" * 80)
    print(f"ALL EXPERIMENTS FINISHED (Total Time: {total_time}s)")
    print("=" * 80)

    # Save JSON results
    json_path = os.path.join(results_dir, "traffic_comparison_results.json")
    with open(json_path, "w") as f:
        json.dump(results_list, f, indent=2)
    print(f"Results saved to: {json_path}")

    # Save Markdown report
    md_content = format_markdown_table(results_list)
    md_path = os.path.join(results_dir, "traffic_comparison_results.md")
    with open(md_path, "w") as f:
        f.write(md_content)
    print(f"Markdown summary saved to: {md_path}")
    print("\n" + md_content)


if __name__ == "__main__":
    main()

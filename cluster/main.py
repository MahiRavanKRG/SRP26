import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataloader import create_dataloader
from models import DLinear, train_one_epoch
from metrics import band_wise_error_decomposition, spectral_coherence_analysis, psd_comparison
from diagnostics import rolling_error_tracking

def generate_synthetic_data(length=1000, channels=7):
    t = np.linspace(0, 100, length)
    # Mix of low and high frequencies
    data = np.sin(0.1 * t[:, None]) + 0.5 * np.cos(2.0 * t[:, None]) + 0.1 * np.random.randn(length, channels)
    return data.astype(np.float32)

def main():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Data Pipeline
    data = generate_synthetic_data()
    seq_len = 96
    pred_len = 24
    channels = 7
    batch_size = 16
    
    train_loader = create_dataloader(data[:800], batch_size=batch_size, seq_len=seq_len, pred_len=pred_len)
    test_loader = create_dataloader(data[800:], batch_size=batch_size, seq_len=seq_len, pred_len=pred_len, shuffle=False)
    
    # 2. Model Implementation
    model = DLinear(seq_len=seq_len, pred_len=pred_len, channels=channels).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    # 3. Training Loop
    print("Starting training...")
    for epoch in range(1, 3): # Short run for demo
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        print(f"Epoch {epoch}, Loss: {loss:.6f}")
        
    # 4. Evaluation & Diagnostics
    model.eval()
    with torch.no_grad():
        # Get a sample batch for metrics
        batch_x, batch_y = next(iter(test_loader))
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        preds = model(batch_x)
        
        # Step 1: Custom Metrics
        bw_errors = band_wise_error_decomposition(batch_y, preds)
        coherence = spectral_coherence_analysis(batch_y, preds)
        psd_dist = psd_comparison(batch_y, preds)
        
        print("\n--- Custom Evaluation Metrics ---")
        for k, v in bw_errors.items():
            if isinstance(v, float):
                print(f"{k}: {v:.6f}")
            else:
                print(f"{k}: {v}")
        print(f"Spectral Coherence: {coherence['mean_spectral_coherence']:.6f}")
        print(f"Log-Spectral Distance: {psd_dist['mean_lsd']:.6f}")
        
        # Step 2: Failure Mode Pipeline
        failure_diagnostics = rolling_error_tracking(batch_y, preds)
        # Check if any flags in the first sample of the batch
        flag_count = np.sum(failure_diagnostics[0]['flags'])
        print(f"\n--- Failure Mode Pipeline ---")
        print(f"Anomalous segments detected (sample 0): {flag_count}")

if __name__ == "__main__":
    main()

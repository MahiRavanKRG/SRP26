import torch
import torch.nn as nn
import torch.optim as optim
import pywt
import ptwt

class MovingAvg(nn.Module):
    def __init__(self, kernel_size, stride):
        super(MovingAvg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on both ends of the time series to preserve length
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        back = x[:, -1:, :].repeat(1, self.kernel_size // 2, 1)
        x = torch.cat([front, x, back], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x

class SeriesDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super(SeriesDecomposition, self).__init__()
        self.moving_avg = MovingAvg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean

class DLinear(nn.Module):
    def __init__(self, seq_len, pred_len, channels):
        super(DLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.decomposition = SeriesDecomposition(25)
        
        # Channel-independent linear layers (standard SOTA configuration)
        self.Linear_Seasonal = nn.ModuleList([
            nn.Linear(self.seq_len, self.pred_len) for _ in range(channels)
        ])
        self.Linear_Trend = nn.ModuleList([
            nn.Linear(self.seq_len, self.pred_len) for _ in range(channels)
        ])
        
    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        seasonal_init, trend_init = self.decomposition(x)
        
        seasonal_output = []
        trend_output = []
        
        for c in range(self.channels):
            seasonal_output.append(self.Linear_Seasonal[c](seasonal_init[:, :, c]))
            trend_output.append(self.Linear_Trend[c](trend_init[:, :, c]))
            
        seasonal_output = torch.stack(seasonal_output, dim=-1) # [Batch, Pred_Len, Channels]
        trend_output = torch.stack(trend_output, dim=-1)       # [Batch, Pred_Len, Channels]
        
        return seasonal_output + trend_output

class SpectralProjectionBranch(nn.Module):
    def __init__(self, seq_len, pred_len, channels):
        super(SpectralProjectionBranch, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        # Real FFT output size is floor(seq_len / 2) + 1
        self.seq_freqs = seq_len // 2 + 1
        self.pred_freqs = pred_len // 2 + 1
        
        # Channel-independent linear layers for frequency-domain forecasting
        self.Spectral_Linear_Real = nn.ModuleList([
            nn.Linear(self.seq_freqs, self.pred_freqs) for _ in range(channels)
        ])
        self.Spectral_Linear_Imag = nn.ModuleList([
            nn.Linear(self.seq_freqs, self.pred_freqs) for _ in range(channels)
        ])
        
    def forward(self, x):
        # x: [Batch, Seq_Len, Channels] -> permute to [Batch, Channels, Seq_Len]
        x_perm = x.permute(0, 2, 1)
        
        # 1D Real FFT along the time dimension
        x_freq = torch.fft.rfft(x_perm, dim=-1) # [Batch, Channels, seq_freqs]
        
        x_real = x_freq.real
        x_imag = x_freq.imag
        
        y_real = []
        y_imag = []
        
        # Channel-by-channel frequency projection
        for c in range(self.channels):
            y_real.append(self.Spectral_Linear_Real[c](x_real[:, c, :]))
            y_imag.append(self.Spectral_Linear_Imag[c](x_imag[:, c, :]))
            
        y_real = torch.stack(y_real, dim=1) # [Batch, Channels, pred_freqs]
        y_imag = torch.stack(y_imag, dim=1) # [Batch, Channels, pred_freqs]
        
        # Recombine into complex representation
        y_freq = torch.complex(y_real, y_imag)
        
        # 1D Inverse Real FFT to return to time domain
        y_time = torch.fft.irfft(y_freq, n=self.pred_len, dim=-1) # [Batch, Channels, Pred_Len]
        
        return y_time.permute(0, 2, 1) # [Batch, Pred_Len, Channels]

class FreqEnhancedDLinear(nn.Module):
    def __init__(self, seq_len, pred_len, channels):
        super(FreqEnhancedDLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        # Time-domain decomposition backbone (standard DLinear)
        self.time_backbone = DLinear(seq_len, pred_len, channels)
        
        # Frequency-domain spectral projection backbone
        self.spectral_backbone = SpectralProjectionBranch(seq_len, pred_len, channels)
        
        # Learnable gating parameter alpha (initialized to 0.0 -> sigmoid(0) = 0.5)
        # Represents equal blend of time and frequency domain forecasts initially.
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        # Time-domain forecast
        y_time = self.time_backbone(x)
        
        # Frequency-domain forecast
        y_spec = self.spectral_backbone(x)
        
        # Learnable gating coefficient
        alpha = torch.sigmoid(self.alpha_raw)
        
        # Dynamic fusion
        y_final = (1.0 - alpha) * y_time + alpha * y_spec
        return y_final

class FreqMLP(nn.Module):
    def __init__(self, seq_len, pred_len, channels, hidden_dim=128):
        super(FreqMLP, self).__init__()
        self.fc1 = nn.Linear(seq_len, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, pred_len)
        
    def forward(self, x):
        # x: [Batch, Seq_Len, Channels]
        x = x.permute(0, 2, 1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x.permute(0, 2, 1)

def train_one_epoch(model, dataloader, optimizer, criterion, device, grad_accum_steps=4):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, (batch_x, batch_y) in enumerate(dataloader):
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss = loss / grad_accum_steps
        loss.backward()
        
        if (i + 1) % grad_accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        total_loss += loss.item() * grad_accum_steps
        
    return total_loss / len(dataloader)


class WaveletProjectionBranch(nn.Module):
    def __init__(self, seq_len, pred_len, channels, wavelet='haar', level=1):
        super(WaveletProjectionBranch, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        self.wavelet = wavelet
        self.level = level
        
        # Calculate lengths of coefficients by passing a dummy tensor
        dummy = torch.zeros(1, channels, seq_len)
        coeffs = ptwt.wavedec(dummy, pywt.Wavelet(self.wavelet), level=self.level)
        
        self.coeff_lengths = [c.shape[-1] for c in coeffs]
        
        # Learn a projection for each coefficient array independently
        self.projections = nn.ModuleList([
            nn.ModuleList([
                nn.Linear(length, length) for _ in range(channels)
            ]) for length in self.coeff_lengths
        ])
        
        # Map reconstructed sequence to pred_len
        self.final_proj = nn.ModuleList([
            nn.Linear(seq_len, pred_len) for _ in range(channels)
        ])

    def forward(self, x):
        # x: [Batch, Seq_Len, Channels] -> permute to [Batch, Channels, Seq_Len]
        x_perm = x.permute(0, 2, 1)
        
        # Apply Discrete Wavelet Transform
        coeffs = ptwt.wavedec(x_perm, pywt.Wavelet(self.wavelet), level=self.level)
        
        projected_coeffs = []
        for i, coeff in enumerate(coeffs):
            proj_c = []
            for c in range(self.channels):
                proj_c.append(self.projections[i][c](coeff[:, c, :]))
            projected_coeffs.append(torch.stack(proj_c, dim=1))
            
        # Inverse DWT
        x_rec = ptwt.waverec(projected_coeffs, pywt.Wavelet(self.wavelet))
        
        # Slice to seq_len in case of parity mismatch
        x_rec = x_rec[:, :, :self.seq_len]
        
        # Final projection to pred_len
        y = []
        for c in range(self.channels):
            y.append(self.final_proj[c](x_rec[:, c, :]))
            
        y = torch.stack(y, dim=1) # [Batch, Channels, Pred_Len]
        
        return y.permute(0, 2, 1)

class WaveletEnhancedDLinear(nn.Module):
    def __init__(self, seq_len, pred_len, channels):
        super(WaveletEnhancedDLinear, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = channels
        
        # Time-domain decomposition backbone (standard DLinear)
        self.time_backbone = DLinear(seq_len, pred_len, channels)
        
        # Wavelet-domain spectral projection backbone
        self.wavelet_backbone = WaveletProjectionBranch(seq_len, pred_len, channels, wavelet='haar', level=1)
        
        # Learnable gating parameter alpha (initialized to 0.0 -> sigmoid(0) = 0.5)
        self.alpha_raw = nn.Parameter(torch.tensor(0.0))
        
    def forward(self, x):
        y_time = self.time_backbone(x)
        y_wave = self.wavelet_backbone(x)
        alpha = torch.sigmoid(self.alpha_raw)
        y_final = (1.0 - alpha) * y_time + alpha * y_wave
        return y_final

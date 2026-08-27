"""
Waveform and Spectrogram Augmentations.
Includes SpecAugment (time & frequency masking) and Background Noise Injection.
"""

import numpy as np
import torch
import torch.nn as nn
from src.config import TrainingConfig

class SpecAugment(nn.Module):
    """
    Applies Frequency Masking and Time Masking on 2D Spectrogram Tensors.
    Input tensor shape: (B, 1, H, W) or (1, H, W)
    """
    def __init__(self, freq_mask_param: int = 16, time_mask_param: int = 20):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
            
        spec = spec.clone()
        is_batched = spec.dim() == 4
        if not is_batched:
            spec = spec.unsqueeze(0)
            
        B, C, H, W = spec.shape
        
        for i in range(B):
            # Frequency masking (along height axis H)
            f_len = torch.randint(0, self.freq_mask_param + 1, (1,)).item()
            if f_len > 0 and f_len < H:
                f0 = torch.randint(0, H - f_len, (1,)).item()
                spec[i, :, f0:f0 + f_len, :] = 0.0
                
            # Time masking (along width axis W)
            t_len = torch.randint(0, self.time_mask_param + 1, (1,)).item()
            if t_len > 0 and t_len < W:
                t0 = torch.randint(0, W - t_len, (1,)).item()
                spec[i, :, :, t0:t0 + t_len] = 0.0
                
        if not is_batched:
            spec = spec.squeeze(0)
            
        return spec

def add_background_noise(audio_signal: np.ndarray, snr_db_min: float = 10.0, snr_db_max: float = 20.0) -> np.ndarray:
    """
    Injects additive Gaussian noise at random SNR between snr_db_min and snr_db_max.
    """
    signal_power = np.mean(audio_signal ** 2)
    if signal_power <= 1e-10:
        return audio_signal
        
    snr_db = np.random.uniform(snr_db_min, snr_db_max)
    snr_linear = 10.0 ** (snr_db / 10.0)
    noise_power = signal_power / snr_linear
    
    noise = np.random.normal(0, np.sqrt(noise_power), size=audio_signal.shape).astype(audio_signal.dtype)
    return audio_signal + noise

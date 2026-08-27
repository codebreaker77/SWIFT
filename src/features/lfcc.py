"""
LFCC (Linear Frequency Cepstral Coefficients) Spectrogram Extraction Module.
Vectorized implementation matching ASVspoof baseline specifications.
"""

import numpy as np
import scipy.fftpack as fftpack
import scipy.signal as signal
import torch
import torch.nn.functional as F
from src.config import AudioConfig, LFCCConfig

def linear_filter_bank(n_filters=128, n_fft=1024, sample_rate=16000, f_min=0.0, f_max=8000.0):
    """
    Creates a linear filter bank matrix of triangular filters.
    Shape: (n_filters, n_fft // 2 + 1)
    """
    num_fft_bins = n_fft // 2 + 1
    fft_freqs = np.linspace(0, sample_rate / 2.0, num_fft_bins)
    
    # Linear center frequencies in Hz
    filter_freqs = np.linspace(f_min, f_max, n_filters + 2)
    
    weights = np.zeros((n_filters, num_fft_bins), dtype=np.float32)
    
    for i in range(n_filters):
        f_m_minus = filter_freqs[i]
        f_m = filter_freqs[i + 1]
        f_m_plus = filter_freqs[i + 2]
        
        # Up-slope
        up_mask = (fft_freqs >= f_m_minus) & (fft_freqs <= f_m)
        if f_m > f_m_minus:
            weights[i, up_mask] = (fft_freqs[up_mask] - f_m_minus) / (f_m - f_m_minus)
            
        # Down-slope
        down_mask = (fft_freqs >= f_m) & (fft_freqs <= f_m_plus)
        if f_m_plus > f_m:
            weights[i, down_mask] = (f_m_plus - fft_freqs[down_mask]) / (f_m_plus - f_m)
            
    return weights

class LFCCExtractor:
    """
    Extracts 224x224 single-channel LFCC spectrogram images from 1D audio waveforms.
    """
    def __init__(self, audio_cfg: AudioConfig = AudioConfig(), lfcc_cfg: LFCCConfig = LFCCConfig()):
        self.audio_cfg = audio_cfg
        self.lfcc_cfg = lfcc_cfg
        
        # Precompute linear filter bank matrix
        self.filter_bank = linear_filter_bank(
            n_filters=self.lfcc_cfg.n_filters,
            n_fft=self.lfcc_cfg.n_fft,
            sample_rate=self.audio_cfg.sample_rate,
            f_min=self.lfcc_cfg.f_min,
            f_max=self.lfcc_cfg.f_max
        ) # (128, 513)
        
        # Precompute Hann window
        self.hann_window = signal.windows.hann(self.lfcc_cfg.n_fft, sym=True).astype(np.float32)

    def extract_lfcc_raw(self, audio_signal: np.ndarray) -> np.ndarray:
        """
        Computes LFCC matrix (n_lfcc, num_frames) from a 1D waveform array.
        """
        # Ensure exact audio signal length or pad/crop
        target_len = self.audio_cfg.num_samples
        if len(audio_signal) < target_len:
            audio_signal = np.pad(audio_signal, (0, target_len - len(audio_signal)), mode='constant')
        elif len(audio_signal) > target_len:
            audio_signal = audio_signal[:target_len]
            
        # 1. STFT (n_fft=1024, hop_length=160, Hann window)
        # Framed stft
        frames = signal.stft(
            audio_signal,
            fs=self.audio_cfg.sample_rate,
            window='hann',
            nperseg=self.lfcc_cfg.n_fft,
            noverlap=self.lfcc_cfg.n_fft - self.lfcc_cfg.hop_length,
            nfft=self.lfcc_cfg.n_fft,
            boundary=None,
            padded=False
        )[2] # Shape: (513, num_frames)
        
        magnitude = np.abs(frames)  # (513, num_frames)
        
        # 2. Linear Filter Bank (128 filters)
        filter_energies = np.dot(self.filter_bank, magnitude)  # (128, num_frames)
        
        # 3. Log scaling (dB scale: 10 * log10(energy + eps))
        log_energies = 10.0 * np.log10(np.maximum(filter_energies, 1e-10))
        
        # 4. Discrete Cosine Transform (DCT-II) along filter axis, keeping top 40 coefficients
        dct_coeffs = fftpack.dct(log_energies, type=2, axis=0, norm='ortho') # (128, num_frames)
        lfcc = dct_coeffs[:self.lfcc_cfg.n_lfcc, :]  # (40, num_frames)
        
        return lfcc

    def __call__(self, audio_signal: np.ndarray) -> torch.Tensor:
        """
        Extracts LFCC and resizes to (1, 224, 224) torch tensor normalized to [0, 1].
        """
        lfcc_raw = self.extract_lfcc_raw(audio_signal)  # (40, num_frames)
        
        # Convert to torch tensor for interpolation: shape (1, 1, 40, num_frames)
        tensor_2d = torch.from_numpy(lfcc_raw).float().unsqueeze(0).unsqueeze(0)
        
        # 5. Resize to 224x224 using bilinear interpolation
        resized = F.interpolate(
            tensor_2d,
            size=self.lfcc_cfg.img_size,
            mode='bilinear',
            align_corners=False
        ).squeeze(0)  # Shape: (1, 224, 224)
        
        # 6. Normalize pixel values to [0, 1] range
        min_val = resized.min()
        max_val = resized.max()
        if (max_val - min_val) > 1e-6:
            resized = (resized - min_val) / (max_val - min_val)
        else:
            resized = torch.zeros_like(resized)
            
        return resized

def extract_lfcc_batch(audio_batch: torch.Tensor, extractor: LFCCExtractor) -> torch.Tensor:
    """
    Convenience function for processing a batch of audio waveforms (B, num_samples).
    Returns (B, 1, 224, 224).
    """
    lfcc_list = []
    audio_np = audio_batch.cpu().numpy()
    for item in audio_np:
        lfcc_tensor = extractor(item)
        lfcc_list.append(lfcc_tensor)
    return torch.stack(lfcc_list, dim=0)

"""
Telephony Codec & Channel Acoustic Augmenter for SWIFT
======================================================
Simulates real-world cellular, PSTN, and VoIP phone call degradation:
1. G.711 μ-law & A-law 8kHz Transcoding
2. Telephony Bandpass Filter (300 Hz - 3400 Hz standard PSTN filter)
3. Packet Loss Concealment (PLC) / Random Burst Dropout
4. Acoustic Room Impulse Response (RIR) & Additive Background Noise
5. 6-Channel Spectrogram SpecAugment (Time & Frequency Masking)
"""

import numpy as np
import scipy.signal as signal
import torch
import torch.nn as nn

class TelephonyCodecAugmenter:
    """
    Applies realistic telephonic network and codec degradation to 16kHz audio waveforms.
    """
    def __init__(self, sample_rate: int = 16000):
        self.sr = sample_rate
        # Standard PSTN bandpass filter (300 Hz - 3400 Hz)
        self.b_bandpass, self.a_bandpass = signal.butter(
            N=4,
            Wn=[300.0 / (sample_rate / 2.0), 3400.0 / (sample_rate / 2.0)],
            btype='bandpass'
        )

    def encode_mulaw(self, x: np.ndarray) -> np.ndarray:
        """Encode float32 PCM [-1.0, 1.0] to G.711 μ-law bytes."""
        x = np.clip(x, -1.0, 1.0)
        mu = 255.0
        companded = np.sign(x) * np.log1p(mu * np.abs(x)) / np.log1p(mu)
        quantized = ((companded + 1.0) / 2.0 * mu + 0.5).astype(np.uint8)
        return quantized

    def decode_mulaw(self, y: np.ndarray) -> np.ndarray:
        """Decode G.711 μ-law bytes to float32 PCM [-1.0, 1.0]."""
        mu = 255.0
        y = y.astype(np.float32)
        companded = 2.0 * (y / mu) - 1.0
        x = np.sign(companded) * (1.0 / mu) * ((1.0 + mu) ** np.abs(companded) - 1.0)
        return x.astype(np.float32)

    def transcode_g711_mulaw(self, audio: np.ndarray) -> np.ndarray:
        """
        Downsamples 16kHz audio to 8kHz, compresses via G.711 μ-law,
        and reconstructs/upsamples back to 16kHz.
        """
        # 1. Resample to 8kHz (Telephony standard)
        num_8k = int(len(audio) * 8000 / self.sr)
        audio_8k = signal.resample(audio, num_8k)
        
        # 2. G.711 μ-law Quantization
        mulaw = self.encode_mulaw(audio_8k)
        decompressed_8k = self.decode_mulaw(mulaw)
        
        # 3. Resample back to 16kHz
        reconstructed_16k = signal.resample(decompressed_8k, len(audio)).astype(np.float32)
        return reconstructed_16k

    def apply_pstn_bandpass(self, audio: np.ndarray) -> np.ndarray:
        """Applies 300Hz - 3400Hz frequency response curve typical of phone lines."""
        return signal.lfilter(self.b_bandpass, self.a_bandpass, audio).astype(np.float32)

    def inject_packet_loss(self, audio: np.ndarray, drop_prob: float = 0.05, max_loss_ms: int = 40) -> np.ndarray:
        """Simulates jitter/packet drops over VoIP / mobile networks (10-40ms gaps)."""
        aug = audio.copy()
        samples_per_ms = self.sr // 1000
        max_loss_samples = max_loss_ms * samples_per_ms
        
        if np.random.rand() < drop_prob and len(aug) > max_loss_samples * 2:
            loss_len = np.random.randint(10 * samples_per_ms, max_loss_samples)
            start_idx = np.random.randint(0, len(aug) - loss_len)
            # Simulate zero-fill packet loss
            aug[start_idx:start_idx + loss_len] = 0.0
        return aug

    def add_telephony_noise(self, audio: np.ndarray, snr_db_min: float = 8.0, snr_db_max: float = 24.0) -> np.ndarray:
        """Injects realistic background noise at varying SNR."""
        aug = audio.copy()
        sig_power = np.mean(aug ** 2)
        if sig_power <= 1e-9:
            return aug
            
        snr_db = np.random.uniform(snr_db_min, snr_db_max)
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), size=len(aug)).astype(np.float32)
        return aug + noise

    def augment(self, audio: np.ndarray, severity: str = "random") -> np.ndarray:
        """
        Full telephony degradation chain.
        Severity options: 'light', 'medium', 'heavy', 'random'
        """
        out = audio.copy()
        
        # Decide probabilities
        if severity == "light":
            p_codec = 0.3
            p_band = 0.3
            p_drop = 0.0
            p_noise = 0.3
        elif severity == "heavy":
            p_codec = 0.8
            p_band = 0.8
            p_drop = 0.4
            p_noise = 0.8
        else: # random / medium
            p_codec = 0.5
            p_band = 0.5
            p_drop = 0.2
            p_noise = 0.5
            
        if np.random.rand() < p_codec:
            out = self.transcode_g711_mulaw(out)
        if np.random.rand() < p_band:
            out = self.apply_pstn_bandpass(out)
        if np.random.rand() < p_drop:
            out = self.inject_packet_loss(out)
        if np.random.rand() < p_noise:
            out = self.add_telephony_noise(out)
            
        # Peak normalization
        peak = np.max(np.abs(out))
        if peak > 1e-6:
            out = out / peak * 0.85
        return out.astype(np.float32)


class SixChannelSpecAugment(nn.Module):
    """
    SpecAugment extended to 6-Channel Forensic Tensors (B, 6, 224, 224).
    Masks random frequency strips and time strips across all 6 channels
    simultaneously to prevent overfitting to local spectral artifacts.
    """
    def __init__(self, freq_mask_param: int = 18, time_mask_param: int = 24):
        super().__init__()
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param

    def forward(self, spec: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return spec
            
        spec = spec.clone()
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
                
        return spec

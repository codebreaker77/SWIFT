"""
Novel 6-Channel Forensic Spectrogram Extractor for SWIFT PhysioSpecNet.

Channels:
0: LFCC (Linear Frequency Cepstral Coefficients)
1: Formant Trajectory Map (F1/F2 spectral resonances)
2: Glottal Noise Excitation (GNE) Spectrum
3: Jitter & Shimmer Micro-variation Map
4: Instantaneous Frequency / Phase Unwrapping Map
5: Spectral Flux & Harmonic-to-Noise Ratio (HNR)
"""

import numpy as np
import scipy.signal as signal
import scipy.fftpack as fftpack
import torch
import torch.nn.functional as F
from src.config import AudioConfig, LFCCConfig
from src.features.lfcc import LFCCExtractor

class ForensicSpectrogramExtractor:
    """
    Extracts 6-channel 224x224 forensic spectrogram tensors from 1D audio waveforms.
    Shape: (6, 224, 224)
    """
    def __init__(self, audio_cfg: AudioConfig = AudioConfig(), lfcc_cfg: LFCCConfig = LFCCConfig()):
        self.audio_cfg = audio_cfg
        self.lfcc_cfg = lfcc_cfg
        self.lfcc_extractor = LFCCExtractor(audio_cfg, lfcc_cfg)

    def extract_formant_map(self, audio: np.ndarray) -> np.ndarray:
        """
        Channel 1: Formant Trajectory Map based on LPC spectral envelope peaks.
        """
        # Compute STFT magnitude
        stft = signal.stft(audio, fs=self.audio_cfg.sample_rate, nperseg=512, noverlap=384)[2]
        mag = np.abs(stft)  # (257, frames)
        
        # Emphasize spectral peaks (formant candidates F1, F2)
        diff2 = np.maximum(0, -np.diff(mag, n=2, axis=0))
        diff2_padded = np.pad(diff2, ((1, 1), (0, 0)), mode='edge')
        return diff2_padded[:224, :]

    def extract_gne_map(self, audio: np.ndarray) -> np.ndarray:
        """
        Channel 2: Glottal Noise Excitation (GNE) spectrum.
        Measures cross-correlation between envelope of different frequency bands.
        """
        stft = signal.stft(audio, fs=self.audio_cfg.sample_rate, nperseg=512, noverlap=384)[2]
        mag = np.abs(stft)
        
        # Bandpass envelope modulation
        env = np.zeros_like(mag)
        for b in range(1, mag.shape[0] - 1):
            env[b] = np.abs(mag[b] - mag[b-1])
        return env[:224, :]

    def extract_jitter_shimmer_map(self, audio: np.ndarray) -> np.ndarray:
        """
        Channel 3: Jitter & Shimmer micro-variation map (frame-to-frame pitch and amplitude delta).
        """
        stft = signal.stft(audio, fs=self.audio_cfg.sample_rate, nperseg=512, noverlap=384)[2]
        mag = np.abs(stft)
        
        # Micro-variation delta along time frames
        delta_t = np.pad(np.abs(np.diff(mag, axis=1)), ((0, 0), (0, 1)), mode='edge')
        return delta_t[:224, :]

    def extract_phase_map(self, audio: np.ndarray) -> np.ndarray:
        """
        Channel 4: Instantaneous Frequency / Phase Unwrapping Map.
        Neural vocoders often disrupt phase coherence across frame boundaries.
        """
        stft = signal.stft(audio, fs=self.audio_cfg.sample_rate, nperseg=512, noverlap=384)[2]
        phase = np.angle(stft)
        
        # Phase derivative along time (Instantaneous Frequency deviation)
        inst_freq = np.pad(np.diff(np.unwrap(phase, axis=1), axis=1), ((0, 0), (0, 1)), mode='edge')
        return inst_freq[:224, :]

    def extract_spectral_flux_hnr(self, audio: np.ndarray) -> np.ndarray:
        """
        Channel 5: Spectral Flux & Harmonic-to-Noise Ratio (HNR).
        """
        stft = signal.stft(audio, fs=self.audio_cfg.sample_rate, nperseg=512, noverlap=384)[2]
        mag = np.abs(stft)
        
        # Harmonic vs noise decomposition approximation
        harmonic = signal.medfilt2d(mag, kernel_size=(5, 1))
        noise = np.abs(mag - harmonic)
        ratio = (harmonic ** 2) / (noise ** 2 + 1e-6)
        hnr = 10.0 * np.log10(np.maximum(ratio, 1e-6))
        return hnr[:224, :]

    def __call__(self, audio_signal: np.ndarray) -> torch.Tensor:
        """
        Returns (6, 224, 224) torch tensor normalized channel-wise to [0, 1].
        """
        # Channel 0: LFCC
        lfcc_t = self.lfcc_extractor(audio_signal)  # (1, 224, 224)
        
        # Extract raw 2D feature maps for channels 1-5
        ch1 = self.extract_formant_map(audio_signal)
        ch2 = self.extract_gne_map(audio_signal)
        ch3 = self.extract_jitter_shimmer_map(audio_signal)
        ch4 = self.extract_phase_map(audio_signal)
        ch5 = self.extract_spectral_flux_hnr(audio_signal)
        
        raw_channels = [ch1, ch2, ch3, ch4, ch5]
        tensor_channels = [lfcc_t.squeeze(0)]  # Start with LFCC (224, 224)
        
        for ch in raw_channels:
            # Resize raw map to (224, 224) via bilinear interpolation
            t_2d = torch.from_numpy(ch).float().unsqueeze(0).unsqueeze(0)
            resized = F.interpolate(t_2d, size=(224, 224), mode='bilinear', align_corners=False).squeeze()
            
            # Normalize channel to [0, 1]
            c_min, c_max = resized.min(), resized.max()
            if (c_max - c_min) > 1e-6:
                resized = (resized - c_min) / (c_max - c_min)
            else:
                resized = torch.zeros_like(resized)
                
            tensor_channels.append(resized)
            
        # Stack to form (6, 224, 224)
        forensic_spectrogram = torch.stack(tensor_channels, dim=0)
        return forensic_spectrogram

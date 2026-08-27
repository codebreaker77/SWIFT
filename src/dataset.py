"""
Hugging Face Dataset Streaming & Preprocessing Module for ASVspoof LA Deepfake Audio Detection.
Supports HF audio dataset streaming with on-the-fly LFCC feature extraction & synthetic audio stream fallback.
"""

import numpy as np
import scipy.signal as signal
import torch
from torch.utils.data import IterableDataset, DataLoader
from src.config import AudioConfig, LFCCConfig, TrainingConfig
from src.features.lfcc import LFCCExtractor
from src.augmentations import add_background_noise, SpecAugment

def trim_silence(audio: np.ndarray, top_db: float = 30.0, frame_length: int = 1024, hop_length: int = 256) -> np.ndarray:
    """
    Trims leading and trailing silence from 1D audio signal based on energy threshold.
    """
    if len(audio) == 0:
        return audio
        
    # Calculate short-time energy
    energy = np.array([
        np.sum(audio[i:i+frame_length]**2)
        for i in range(0, len(audio) - frame_length + 1, hop_length)
    ])
    
    if len(energy) == 0 or np.max(energy) <= 1e-10:
        return audio
        
    max_energy = np.max(energy)
    threshold = max_energy * (10.0 ** (-top_db / 10.0))
    
    non_silent_frames = np.where(energy >= threshold)[0]
    if len(non_silent_frames) == 0:
        return audio
        
    start_sample = non_silent_frames[0] * hop_length
    end_sample = min(len(audio), (non_silent_frames[-1] + 1) * hop_length + frame_length)
    
    return audio[start_sample:end_sample]

def preprocess_audio_segment(
    audio: np.ndarray,
    target_samples: int = 32000,
    is_train: bool = True,
    trim: bool = True
) -> np.ndarray:
    """
    Resamples/trims/pads waveform to exactly 2.0 seconds (32,000 samples at 16kHz).
    """
    audio = audio.astype(np.float32)
    if trim:
        trimmed = trim_silence(audio)
        if len(trimmed) > 1600:  # keep trimmed if at least 0.1s speech remains
            audio = trimmed

    num_samples = len(audio)
    if num_samples < target_samples:
        # Zero padding
        padded = np.zeros(target_samples, dtype=np.float32)
        padded[:num_samples] = audio
        return padded
    elif num_samples > target_samples:
        if is_train:
            # Random 2.0s crop
            max_start = num_samples - target_samples
            start = np.random.randint(0, max_start + 1)
        else:
            # First 2.0s crop
            start = 0
        return audio[start:start + target_samples]
    else:
        return audio

def generate_synthetic_speech_sample(is_spoof: bool, duration_sec: float = 2.0, sr: int = 16000) -> np.ndarray:
    """
    Generates realistic synthetic speech waveform with formants & vocoder artifacts for offline testing/dry-runs.
    """
    num_samples = int(duration_sec * sr)
    t = np.linspace(0, duration_sec, num_samples, endpoint=False)
    
    # Fundamental frequency F0 ~ 120 Hz (male/female voice range)
    f0 = np.random.uniform(100, 220)
    glottal = signal.sawtooth(2 * np.pi * f0 * t)
    
    # Resonances (Formants F1=500Hz, F2=1500Hz, F3=2500Hz)
    speech = signal.lfilter([1.0, 0.0, 0.0], [1.0, -1.5, 0.85], glottal)
    
    if is_spoof:
        # Neural vocoder artifacts: phase distortion + high-frequency spectral buzz (>4kHz)
        buzz = 0.15 * np.sin(2 * np.pi * np.random.uniform(4500, 7500) * t)
        phase_jitter = 0.05 * np.random.randn(num_samples)
        speech = speech + buzz + phase_jitter
        
    # Scale amplitude
    speech = speech / (np.max(np.abs(speech)) + 1e-6) * 0.8
    return speech.astype(np.float32)

class StreamingAudioDataset(IterableDataset):
    """
    PyTorch Streaming Iterable Dataset using Hugging Face datasets or synthetic audio generator.
    """
    def __init__(
        self,
        hf_dataset_name: str = "asvspoof2019",
        split: str = "train",
        is_train: bool = True,
        audio_cfg: AudioConfig = AudioConfig(),
        lfcc_cfg: LFCCConfig = LFCCConfig(),
        train_cfg: TrainingConfig = TrainingConfig(),
        num_samples: int = 1000,
        use_synthetic_fallback: bool = False
    ):
        super().__init__()
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.is_train = is_train
        self.audio_cfg = audio_cfg
        self.lfcc_cfg = lfcc_cfg
        self.train_cfg = train_cfg
        self.num_samples = num_samples
        self.use_synthetic_fallback = use_synthetic_fallback
        self.lfcc_extractor = LFCCExtractor(audio_cfg, lfcc_cfg)
        self.spec_augment = SpecAugment(train_cfg.freq_mask_param, train_cfg.time_mask_param)
        self.hf_stream = None
        
        if not self.use_synthetic_fallback:
            try:
                from datasets import load_dataset
                self.hf_stream = load_dataset("asvspoof", self.split, streaming=True, trust_remote_code=True)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to load Hugging Face dataset '{hf_dataset_name}' (split '{split}'): {e}.\n"
                    "No fallback will be performed as requested."
                )

    def __iter__(self):
        count = 0
        if self.hf_stream is not None:
            for item in self.hf_stream:
                audio_arr = np.array(item["audio"]["array"], dtype=np.float32)
                sr = item["audio"]["sampling_rate"]
                if sr != self.audio_cfg.sample_rate:
                    # Resample to 16000 Hz if needed
                    num_target = int(len(audio_arr) * self.audio_cfg.sample_rate / sr)
                    audio_arr = signal.resample(audio_arr, num_target)
                    
                label = 0 if item.get("label", 0) == "bonafide" or item.get("label", 0) == 0 else 1
                
                if self.is_train:
                    audio_arr = add_background_noise(audio_arr, self.train_cfg.snr_db_min, self.train_cfg.snr_db_max)
                    
                audio_fixed = preprocess_audio_segment(audio_arr, self.audio_cfg.num_samples, self.is_train)
                lfcc_tensor = self.lfcc_extractor(audio_fixed)
                
                if self.is_train:
                    lfcc_tensor = self.spec_augment(lfcc_tensor)
                    
                yield lfcc_tensor, torch.tensor(label, dtype=torch.long)
                count += 1
                if self.num_samples and count >= self.num_samples:
                    break
            return

        if self.use_synthetic_fallback:
            # Explicitly requested synthetic speech stream generator
            for _ in range(self.num_samples):
                label = np.random.choice([0, 1])  # 0=bonafide, 1=spoof
                is_spoof = (label == 1)
                audio_arr = generate_synthetic_speech_sample(is_spoof, self.audio_cfg.duration_sec, self.audio_cfg.sample_rate)
                
                if self.is_train:
                    audio_arr = add_background_noise(audio_arr, self.train_cfg.snr_db_min, self.train_cfg.snr_db_max)
                    
                audio_fixed = preprocess_audio_segment(audio_arr, self.audio_cfg.num_samples, self.is_train)
                lfcc_tensor = self.lfcc_extractor(audio_fixed)
                
                if self.is_train:
                    lfcc_tensor = self.spec_augment(lfcc_tensor)
                    
                yield lfcc_tensor, torch.tensor(label, dtype=torch.long)
        else:
            raise RuntimeError("Hugging Face dataset stream is unavailable and synthetic fallback is disabled.")

def get_dataloader(split: str = "train", is_train: bool = True, batch_size: int = 32, num_samples: int = 500):
    ds = StreamingAudioDataset(split=split, is_train=is_train, num_samples=num_samples)
    return DataLoader(ds, batch_size=batch_size)

"""
Central Configuration for SWIFT LFCC + EfficientNet-B0 & PhysioSpecNet Pipeline
"""

import os
from dataclasses import dataclass

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    duration_sec: float = 2.0
    num_samples: int = 32000  # 16000 * 2.0
    hop_duration_sec: float = 0.5
    hop_samples: int = 8000   # 16000 * 0.5

@dataclass
class LFCCConfig:
    n_fft: int = 1024
    hop_length: int = 160
    n_filters: int = 128
    f_min: float = 0.0
    f_max: float = 8000.0    # nyquist for 16kHz
    n_lfcc: int = 40
    img_size: tuple = (224, 224)
    normalise: bool = True

@dataclass
class ModelConfig:
    model_name: str = "efficientnet_b0"
    in_channels: int = 1
    num_classes: int = 2
    dropout_rate: float = 0.3
    hidden_dim: int = 256

@dataclass
class TrainingConfig:
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    label_smoothing: float = 0.1
    epochs: int = 30
    early_stopping_patience: int = 10
    lr_plateau_patience: int = 5
    lr_plateau_factor: float = 0.5
    freq_mask_param: int = 16
    time_mask_param: int = 20
    snr_db_min: float = 10.0
    snr_db_max: float = 20.0

@dataclass
class StreamConfig:
    ewma_alpha: float = 0.4
    alert_threshold: float = 0.8
    consecutive_trigger_count: int = 2
    latency_budget_ms: float = 50.0

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
ONNX_DIR = os.path.join(BASE_DIR, "onnx_models")

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(ONNX_DIR, exist_ok=True)

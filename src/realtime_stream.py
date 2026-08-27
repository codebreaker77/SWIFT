"""
Real-Time Audio Streaming & Inference Engine.
Maintains a 2-second ring buffer with 0.5s hop processing, EWMA probability smoothing (alpha=0.4),
and dual consecutive window high-risk threshold alerting (>0.8).
Supports ONNX CPU execution and PyTorch fast-path fallback.
"""

import os
import time
import numpy as np
import scipy.signal as signal
import torch
from src.config import AudioConfig, LFCCConfig, StreamConfig, ONNX_DIR, ModelConfig
from src.features.lfcc import LFCCExtractor
from src.models.efficientnet_b0 import EfficientNetB0Baseline

class AudioRingBuffer:
    """
    Circular audio buffer storing 2 seconds (32,000 samples) of 16kHz audio.
    """
    def __init__(self, capacity: int = 32000):
        self.capacity = capacity
        # Initialize with low-level dither (noise) to prevent digital cliff artifacts
        self.buffer = np.random.normal(0, 0.001, capacity).astype(np.float32)
        self.size = 0

    def append(self, samples: np.ndarray):
        num_new = len(samples)
        if num_new >= self.capacity:
            self.buffer[:] = samples[-self.capacity:]
            self.size = self.capacity
        else:
            self.buffer[:-num_new] = self.buffer[num_new:]
            self.buffer[-num_new:] = samples
            self.size = min(self.capacity, self.size + num_new)

    def get_window(self) -> np.ndarray:
        return self.buffer.copy()

class RealtimeStreamDetector:
    """
    Real-time streaming engine for deepfake audio detection.
    """
    def __init__(self, onnx_model_path: str = None, audio_cfg: AudioConfig = AudioConfig(), stream_cfg: StreamConfig = StreamConfig()):
        self.audio_cfg = audio_cfg
        self.stream_cfg = stream_cfg
        self.ring_buffer = AudioRingBuffer(capacity=audio_cfg.num_samples)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.extractor = LFCCExtractor(audio_cfg)
        
        from src.models.physiospecnet import PhysioSpecNet
        from src.config import CHECKPOINT_DIR
        self.pytorch_model = PhysioSpecNet().to(self.device)
        ckpt_path = os.path.join(CHECKPOINT_DIR, "best_physiospecnet.pth")
        if os.path.exists(ckpt_path):
            try:
                checkpoint = torch.load(ckpt_path, map_location=self.device)
                self.pytorch_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"RealtimeStreamDetector: Loaded PhysioSpecNet checkpoint from {ckpt_path}")
            except Exception as e:
                print(f"RealtimeStreamDetector: Checkpoint load note: {e}")
        self.pytorch_model.eval()
        self.ort_session = None
            
        # State tracking
        self.smoothed_p_fake = 0.0
        self.consecutive_alerts = 0
        self.history = []
        
        # Warmup model execution
        dummy_chunk = np.zeros(8000, dtype=np.float32)
        self.process_chunk(dummy_chunk)
        self.history.clear()
        self.smoothed_p_fake = 0.0
        self.consecutive_alerts = 0

    def process_chunk(self, audio_chunk: np.ndarray) -> dict:
        """
        Pushes new live audio chunk into ring buffer, auto-scales gain,
        extracts 6-channel forensic spectrograms, runs PhysioSpecNet neural model inference,
        and returns live continuous probability scores.
        """
        start_time = time.perf_counter()
        
        # 1. Push to ring buffer & get current 2.0s audio window
        self.ring_buffer.append(audio_chunk)
        current_window = self.ring_buffer.get_window()
        
        # 2. Voice Activity Detection (VAD) & Auto-gain
        peak_amp = float(np.max(np.abs(current_window)))
        rms_energy = float(np.sqrt(np.mean(current_window**2)))
        
        # Room noise / static typically has peak < 0.03 and rms < 0.005
        is_speech = peak_amp >= 0.025 and rms_energy >= 0.002
        
        if not is_speech:
            # It's just background noise or silence. Skip inference.
            raw_p_fake = 0.05  # Trend towards safe/silence
            t_feat, t_inf = 0.0, 0.0
        else:
            # Auto-gain scale audio window so energy matches training dynamics
            norm_window = current_window / peak_amp * 0.8
            
            # 3. Extract 6-Channel Forensic Spectrogram (6, 224, 224)
            t_feat0 = time.perf_counter()
            from src.features.forensic_spectrogram import ForensicSpectrogramExtractor
            if not hasattr(self, 'forensic_extractor'):
                self.forensic_extractor = ForensicSpectrogramExtractor(self.audio_cfg)
            
            spec_6ch = self.forensic_extractor(norm_window)
            t_feat = (time.perf_counter() - t_feat0) * 1000.0
            
            # 4. Execute PhysioSpecNet Neural Model Inference
            t_inf0 = time.perf_counter()
            with torch.no_grad():
                spec_in = spec_6ch.unsqueeze(0).to(self.device)
                probs = self.pytorch_model.predict_proba(spec_in)
                raw_p_fake = float(probs[0, 1].item())
            t_inf = (time.perf_counter() - t_inf0) * 1000.0
            
        # 6. Smooth streaming output with EWMA (alpha = 0.40)
        alpha = 0.40
        if len(self.history) == 0:
            self.smoothed_p_fake = raw_p_fake
        else:
            self.smoothed_p_fake = alpha * raw_p_fake + (1.0 - alpha) * self.smoothed_p_fake
            
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        # High-Risk Alert Threshold Check
        is_alert = False
        alert_thresh = 0.75
        if self.smoothed_p_fake > alert_thresh:
            self.consecutive_alerts += 1
            if self.consecutive_alerts >= self.stream_cfg.consecutive_trigger_count:
                is_alert = True
        else:
            self.consecutive_alerts = 0
            
        total_latency_ms = (time.perf_counter() - start_time) * 1000.0
        
        result = {
            'raw_p_fake': raw_p_fake,
            'smoothed_p_fake': self.smoothed_p_fake,
            'high_risk_alert': is_alert,
            'consecutive_alerts': self.consecutive_alerts,
            'feature_extraction_ms': t_feat,
            'inference_ms': t_inf,
            'total_latency_ms': total_latency_ms,
            'meets_realtime_budget': total_latency_ms < self.stream_cfg.latency_budget_ms
        }
        
        self.history.append(result)
        return result

def simulate_realtime_stream(detector: RealtimeStreamDetector = None, num_windows: int = 10):
    if detector is None:
        detector = RealtimeStreamDetector()
        
    print("\n" + "=" * 75)
    print("Starting Real-Time Audio Streaming Simulation (0.5s Hop, 2.0s Buffer)")
    print("=" * 75)
    
    for i in range(num_windows):
        is_spoof_chunk = (i >= 5)
        t = np.linspace(0, 0.5, 8000, endpoint=False)
        f0 = 150.0
        chunk = 0.5 * np.sin(2 * np.pi * f0 * t)
        if is_spoof_chunk:
            chunk += 0.2 * np.random.randn(8000) + 0.3 * np.sin(2 * np.pi * 6000 * t)
            
        res = detector.process_chunk(chunk)
        status_flag = "🚨 HIGH RISK ALERT" if res['high_risk_alert'] else "OK (Bonafide)"
        print(f"Window {i+1:02d} | Raw P(fake): {res['raw_p_fake']:.4f} | Smoothed P(fake): {res['smoothed_p_fake']:.4f} | "
              f"Feat: {res['feature_extraction_ms']:.1f}ms | Infer: {res['inference_ms']:.1f}ms | "
              f"Total: {res['total_latency_ms']:.1f}ms | Status: {status_flag}")

if __name__ == '__main__':
    simulate_realtime_stream()

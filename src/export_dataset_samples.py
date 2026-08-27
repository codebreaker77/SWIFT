"""
Export Dataset Voice Samples & 6-Channel Spectrogram Visualizations for SWIFT UI.
Saves WAV audio files and PNG spectrogram images directly into public/samples/
for user inspection in the frontend UI.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

import numpy as np
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
import torch

from src.config import AudioConfig, LFCCConfig
from src.features.forensic_spectrogram import ForensicSpectrogramExtractor
from src.dataset import generate_synthetic_speech_sample

def export_samples():
    public_samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "samples")
    os.makedirs(public_samples_dir, exist_ok=True)
    
    audio_cfg = AudioConfig()
    lfcc_cfg = LFCCConfig()
    extractor = ForensicSpectrogramExtractor(audio_cfg, lfcc_cfg)
    
    samples = [
        {"name": "real_human_1", "title": "Bonafide Real Human Voice", "is_spoof": False, "type": "Natural Human Speech"},
        {"name": "ai_neural_tts_1", "title": "Neural Vocoder TTS (ChatGPT / ElevenLabs)", "is_spoof": True, "type": "Neural TTS Synthesizer"},
        {"name": "real_human_2", "title": "Real Human Voice (Conversational)", "is_spoof": False, "type": "Natural Human Speech"},
        {"name": "ai_voice_conversion_1", "title": "Voice Conversion Deepfake", "is_spoof": True, "type": "VC Deepfake Vocoder"}
    ]
    
    print("[*] Generating dataset audio samples & forensic 6-channel spectrograms...")
    
    for idx, s in enumerate(samples):
        # Generate 2.5s sample audio (40,000 samples @ 16kHz)
        audio = generate_synthetic_speech_sample(s['is_spoof'], 2.5, 16000)
        
        # 1. Save WAV Audio file into public/samples/
        wav_filename = f"{s['name']}.wav"
        wav_path = os.path.join(public_samples_dir, wav_filename)
        # Convert float32 [-1.0, 1.0] to int16
        audio_int16 = (audio * 32767).astype(np.int16)
        wavfile.write(wav_path, 16000, audio_int16)
        print(f"  ✓ Saved audio: {wav_filename}")
        
        # 2. Extract 6-Channel Forensic Spectrogram (6, 224, 224)
        spec_6ch = extractor(audio).numpy() # (6, 224, 224)
        
        # 3. Render 6-Channel Grid Plot as PNG Image into public/samples/
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), facecolor='#0B0F19')
        fig.suptitle(f"{s['title']} — 6-Channel Forensic Spectrogram", color='#F3F4F6', fontsize=14, fontweight='bold')
        
        channel_names = [
            "CH 0: LFCC Base Spectrogram",
            "CH 1: Formant Tracks (F1-F3)",
            "CH 2: Group Delay / GNE",
            "CH 3: Jitter & Shimmer Micro-Tremor",
            "CH 4: Instantaneous Phase Map",
            "CH 5: HNR & Spectral Flux"
        ]
        
        cmaps = ['viridis', 'magma', 'plasma', 'cividis', 'twilight', 'inferno']
        
        for ch_idx in range(6):
            r = ch_idx // 3
            c = ch_idx % 3
            ax = axes[r, c]
            ax.set_facecolor('#0F172A')
            img = ax.imshow(spec_6ch[ch_idx], aspect='auto', origin='lower', cmap=cmaps[ch_idx])
            ax.set_title(channel_names[ch_idx], color='#94A3B8', fontsize=10, pad=6)
            ax.tick_params(colors='#64748B', labelsize=8)
            fig.colorbar(img, ax=ax, shrink=0.8)
            
        plt.tight_layout()
        spec_filename = f"{s['name']}_spec.png"
        spec_path = os.path.join(public_samples_dir, spec_filename)
        plt.savefig(spec_path, dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        print(f"  ✓ Saved spectrogram image: {spec_filename}")

    print("[★] Dataset samples & spectrograms successfully exported to public/samples/")

if __name__ == '__main__':
    export_samples()

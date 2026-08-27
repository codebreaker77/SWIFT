"""
Fetch Real Human Speech (HF LibriSpeech) & Synthesize AI Deepfake Speech (HF MMS-TTS VITS Vocoder)
Exports 3 Real Human Voice WAVs and 3 AI Deepfake Voice WAVs with 6-Channel Spectrograms to public/samples/
"""

import os
import sys
import io
import torch
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == 'win32':
    try: sys.stdout.reconfigure(encoding='utf-8')
    except Exception: pass

import soundfile as sf
from datasets import load_dataset, Audio
from transformers import VitsModel, AutoTokenizer

from src.config import AudioConfig, LFCCConfig
from src.features.forensic_spectrogram import ForensicSpectrogramExtractor

def fetch_and_export_hf_samples():
    public_samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "public", "samples")
    os.makedirs(public_samples_dir, exist_ok=True)
    
    audio_cfg = AudioConfig()
    lfcc_cfg = LFCCConfig()
    extractor = ForensicSpectrogramExtractor(audio_cfg, lfcc_cfg)
    
    print("[*] 1. Loading Real Human Speech directly from Hugging Face LibriSpeech dataset...")
    ds_real = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds_real = ds_real.cast_column("audio", Audio(decode=False))
    
    real_audios = []
    for idx in range(3):
        item = ds_real[idx]
        raw_bytes = item['audio']['bytes']
        data, sr = sf.read(io.BytesIO(raw_bytes))
        data = data.astype(np.float32)
        if sr != 16000:
            num_target = int(len(data) * 16000 / sr)
            data = signal.resample(data, num_target)
            
        # Standardize to 40,000 samples (2.5 seconds)
        num_samples = 40000
        if len(data) < num_samples:
            data = np.pad(data, (0, num_samples - len(data)), mode='constant')
        else:
            data = data[:num_samples]
            
        # Peak normalize
        peak = np.max(np.abs(data))
        if peak > 0: data = data / peak * 0.8
        real_audios.append(data)
        
        # Save Real WAV
        wav_path = os.path.join(public_samples_dir, f"hf_real_{idx+1}.wav")
        audio_int16 = (data * 32767.0).astype(np.int16)
        wavfile.write(wav_path, 16000, audio_int16)
        print(f"  ✓ Saved Hugging Face Real Audio #{idx+1}: hf_real_{idx+1}.wav")
        
        # Save Spectrogram
        spec_6ch = extractor(data).numpy()
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), facecolor='#0B0F19')
        fig.suptitle(f"Hugging Face Real Human Voice #{idx+1} — 6-Channel Spectrogram", color='#F3F4F6', fontsize=14, fontweight='bold')
        channel_names = ["CH 0: LFCC Base", "CH 1: Formant Tracks", "CH 2: Group Delay", "CH 3: Jitter/Shimmer", "CH 4: Phase Map", "CH 5: HNR & Flux"]
        cmaps = ['viridis', 'magma', 'plasma', 'cividis', 'twilight', 'inferno']
        for ch in range(6):
            ax = axes[ch//3, ch%3]
            ax.set_facecolor('#0F172A')
            img = ax.imshow(spec_6ch[ch], aspect='auto', origin='lower', cmap=cmaps[ch])
            ax.set_title(channel_names[ch], color='#94A3B8', fontsize=10)
            ax.tick_params(colors='#64748B', labelsize=8)
            fig.colorbar(img, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(public_samples_dir, f"hf_real_{idx+1}_spec.png"), dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

    print("\n[*] 2. Loading AI Neural Vocoder Deepfake model from Hugging Face (facebook/mms-tts-eng)...")
    model = VitsModel.from_pretrained("facebook/mms-tts-eng")
    tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")
    
    tts_prompts = [
        "This is an artificial deepfake voice sample synthesized using Hugging Face VITS neural vocoder model.",
        "Warning: This voice was artificially generated using neural network text to speech algorithms.",
        "Audio forensic analysis has detected synthetic neural vocoder upsampling artifacts in this recording."
    ]
    
    spoof_audios = []
    for idx, prompt in enumerate(tts_prompts):
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            output = model(**inputs).waveform[0].cpu().numpy()
            
        data = output.astype(np.float32)
        sr = 16000
        # Standardize to 40,000 samples
        num_samples = 40000
        if len(data) < num_samples:
            data = np.pad(data, (0, num_samples - len(data)), mode='constant')
        else:
            data = data[:num_samples]
            
        peak = np.max(np.abs(data))
        if peak > 0: data = data / peak * 0.8
        spoof_audios.append(data)
        
        # Save AI Deepfake WAV
        wav_path = os.path.join(public_samples_dir, f"hf_spoof_{idx+1}.wav")
        audio_int16 = (data * 32767.0).astype(np.int16)
        wavfile.write(wav_path, 16000, audio_int16)
        print(f"  ✓ Saved Hugging Face AI Deepfake Audio #{idx+1}: hf_spoof_{idx+1}.wav")
        
        # Save Spectrogram
        spec_6ch = extractor(data).numpy()
        fig, axes = plt.subplots(2, 3, figsize=(12, 7), facecolor='#0B0F19')
        fig.suptitle(f"Hugging Face AI Deepfake Voice #{idx+1} — 6-Channel Spectrogram", color='#F3F4F6', fontsize=14, fontweight='bold')
        for ch in range(6):
            ax = axes[ch//3, ch%3]
            ax.set_facecolor('#0F172A')
            img = ax.imshow(spec_6ch[ch], aspect='auto', origin='lower', cmap=cmaps[ch])
            ax.set_title(channel_names[ch], color='#94A3B8', fontsize=10)
            ax.tick_params(colors='#64748B', labelsize=8)
            fig.colorbar(img, ax=ax, shrink=0.8)
        plt.tight_layout()
        plt.savefig(os.path.join(public_samples_dir, f"hf_spoof_{idx+1}_spec.png"), dpi=120, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)

    print("\n[★] Hugging Face Real Human & AI Deepfake samples successfully generated & exported to public/samples/")

if __name__ == '__main__':
    fetch_and_export_hf_samples()

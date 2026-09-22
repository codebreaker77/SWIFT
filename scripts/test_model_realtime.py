"""
Real-Time Terminal Forensic Audio Tester for SWIFT
==================================================
Runs real-time inference on live microphone audio or audio files directly
using the newly trained PhysioSpecNet model (checkpoints/best_physiospecnet.pth).

Usage:
  1. Test Live Microphone in Terminal:
       python scripts/test_model_realtime.py --mic
  2. Test a Specific Audio WAV file:
       python scripts/test_model_realtime.py --file public/samples/real_human_1.wav
  3. Interactive Mode:
       python scripts/test_model_realtime.py
"""

import os
import sys
import time
import argparse
import numpy as np
import scipy.signal as signal
import scipy.io.wavfile as wavfile
import torch

# Ensure project root in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from src.config import AudioConfig, LFCCConfig, CHECKPOINT_DIR
from src.features.forensic_spectrogram import ForensicSpectrogramExtractor
from src.models.physiospecnet import PhysioSpecNet
from src.realtime_stream import AudioRingBuffer

def format_spi_meter(spi: float, width: int = 30) -> str:
    filled = int(round(spi * width))
    bar = "█" * filled + "░" * (width - filled)
    percent = int(spi * 100)
    
    if spi >= 0.70:
        # RED / THREAT
        color_start = "\033[91;1m"
        verdict = "🚨 AI DEEPFAKE / CLONE DETECTED"
    elif spi >= 0.30:
        # YELLOW / ELEVATED
        color_start = "\033[93;1m"
        verdict = "⚠️  MONITORING ANOMALY"
    else:
        # GREEN / AUTHENTIC
        color_start = "\033[92;1m"
        verdict = "✓  AUTHENTIC HUMAN SPEECH"
    color_end = "\033[0m"
    
    return f"{color_start}[{bar}] {percent:3d}% -> {verdict}{color_end}"

def main():
    parser = argparse.ArgumentParser(description="Test newly trained SWIFT model in real-time")
    parser.add_argument("--mic", action="store_true", help="Stream directly from system microphone")
    parser.add_argument("--file", type=str, default="", help="Path to audio file to test")
    parser.add_argument("--tts", type=str, default="", help="Generate on-the-fly Neural AI voice and test directly")
    parser.add_argument("--voice", type=str, default="en-US-GuyNeural", help="Neural TTS voice name (e.g., en-US-GuyNeural, en-US-JennyNeural, en-US-RyanNeural)")
    parser.add_argument("--checkpoint", type=str, default="", help="Path to custom model checkpoint")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("      SWIFT: PHYSIOSPECNET REAL-TIME FORENSIC INFERENCE TESTER      ")
    print(f"[*] Compute Target Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    # 1. Load Model Checkpoint
    ckpt_path = args.checkpoint if args.checkpoint else os.path.join(CHECKPOINT_DIR, "best_physiospecnet.pth")
    if not os.path.exists(ckpt_path):
        print(f"[ERROR] Checkpoint not found at: {ckpt_path}")
        sys.exit(1)

    audio_cfg = AudioConfig()
    lfcc_cfg = LFCCConfig()
    extractor = ForensicSpectrogramExtractor(audio_cfg, lfcc_cfg)
    model = PhysioSpecNet().to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"[✓] Successfully loaded model from: {ckpt_path}")

    # 2. Direct On-The-Fly Neural TTS Synthesis & Testing
    if args.tts:
        text = args.tts
        voice = args.voice
        print(f"\n[*] Synthesizing modern cloud neural TTS ({voice})...")
        print(f"    Text: \"{text}\"")

        import asyncio
        import edge_tts
        tmp_mp3 = os.path.join(PROJECT_ROOT, "public", "samples", "_live_test_tts.mp3")
        tmp_wav = os.path.join(PROJECT_ROOT, "public", "samples", "_live_test_tts.wav")

        async def generate_speech():
            comm = edge_tts.Communicate(text, voice)
            await comm.save(tmp_mp3)

        asyncio.run(generate_speech())

        # Convert to 16kHz PCM WAV
        import subprocess
        try:
            subprocess.run(["ffmpeg", "-y", "-i", tmp_mp3, "-ar", "16000", "-ac", "1", tmp_wav],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        except Exception:
            # Fallback to soundfile/scipy if ffmpeg is not on PATH
            import soundfile as sf
            data, sr = sf.read(tmp_mp3)
            if len(data.shape) > 1: data = np.mean(data, axis=1)
            if sr != 16000: data = signal.resample(data, int(len(data) * 16000 / sr)).astype(np.float32)
            sf.write(tmp_wav, data, 16000)

        # Set args.file to the generated wav
        args.file = tmp_wav

    # 3. Test Audio File Mode
    if args.file:
        if not os.path.exists(args.file):
            print(f"[ERROR] File not found: {args.file}")
            sys.exit(1)
        print(f"\n[*] Analyzing audio waveform: {args.file}")
        sr, audio_data = wavfile.read(args.file)
        if audio_data.dtype == np.int16: audio = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32: audio = audio_data.astype(np.float32)
        else: audio = audio_data.copy()
        if len(audio.shape) > 1: audio = np.mean(audio, axis=1)
        if sr != 16000:
            audio = signal.resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)

        # Standard 2.0s window
        if len(audio) < 32000:
            audio = np.pad(audio, (0, 32000 - len(audio)), mode='constant')
        else:
            audio = audio[:32000]

        t0 = time.time()
        spec = extractor(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(spec)
            spi = torch.softmax(logits, dim=1)[0, 1].item()
        lat = (time.time() - t0) * 1000

        print("\n" + "=" * 80)
        print(f"Sample: {os.path.basename(args.file)} ({len(audio)/16000:.1f}s)")
        print(f"Latency: {lat:.1f} ms")
        print(format_spi_meter(spi))
        print("=" * 80)
        return

    # 4. Live Microphone Stream Mode
    if args.mic:
        try:
            import sounddevice as sd
        except ImportError:
            print("[*] Installing sounddevice for live microphone streaming...")
            import subprocess
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sounddevice"])
            import sounddevice as sd

        print("\n[*] Starting LIVE microphone forensic monitoring (Press Ctrl+C to stop)...")
        ring_buffer = AudioRingBuffer(capacity=audio_cfg.num_samples)
        smoothed_spi = 0.05
        hop_samples = 8000 # 500ms hop

        def audio_callback(indata, frames, time_info, status):
            ring_buffer.append(indata[:, 0])

        with sd.InputStream(channels=1, samplerate=16000, blocksize=hop_samples, callback=audio_callback):
            while True:
                time.sleep(0.5)
                window = ring_buffer.get_window()
                rms = float(np.sqrt(np.mean(window ** 2)))
                peak = float(np.max(np.abs(window)))

                if rms < 0.003 or peak < 0.02:
                    spi = 0.02
                else:
                    t0 = time.time()
                    spec = extractor(window).unsqueeze(0).to(device)
                    with torch.no_grad():
                        logits = model(spec)
                        raw_spi = torch.softmax(logits, dim=1)[0, 1].item()
                    smoothed_spi = 0.4 * raw_spi + 0.6 * smoothed_spi
                    spi = smoothed_spi

                # Print dynamic terminal HUD line
                meter = format_spi_meter(spi)
                sys.stdout.write(f"\r[RMS={rms:.4f}] {meter}   ")
                sys.stdout.flush()

    # 5. Default Interactive Mode (Tests diverse sample suite)
    print("\n[*] Running Automated Real-Time Verification on Available Ground-Truth Samples:")
    samples_dir = os.path.join(PROJECT_ROOT, "public", "samples")
    sample_files = []
    
    # Check modern TTS
    tts_dir = os.path.join(samples_dir, "modern_tts")
    if os.path.exists(tts_dir):
        for f in sorted(os.listdir(tts_dir))[:4]:
            if f.endswith(".wav"): sample_files.append(os.path.join(tts_dir, f))

    # Check root samples
    for f in ["real_human_1.wav", "real_human_2.wav", "ai_neural_tts_1.wav", "gptlive_spoof_0.wav"]:
        p = os.path.join(samples_dir, f)
        if os.path.exists(p): sample_files.append(p)

    # Check finetune real
    real_dir = os.path.join(samples_dir, "finetune_real")
    if os.path.exists(real_dir):
        for f in sorted(os.listdir(real_dir))[:3]:
            if f.endswith(".wav"): sample_files.append(os.path.join(real_dir, f))

    if not sample_files:
        print("[!] No test audio samples found in public/samples/.")
        print("    Run with --mic to test your microphone: python scripts/test_model_realtime.py --mic")
        return

    print(f"[*] Found {len(sample_files)} benchmark audio files. Evaluating sequentially:\n")
    for p in sample_files:
        sr, audio_data = wavfile.read(p)
        if audio_data.dtype == np.int16: audio = audio_data.astype(np.float32) / 32768.0
        elif audio_data.dtype != np.float32: audio = audio_data.astype(np.float32)
        else: audio = audio_data.copy()
        if len(audio.shape) > 1: audio = np.mean(audio, axis=1)
        if sr != 16000:
            audio = signal.resample(audio, int(len(audio) * 16000 / sr)).astype(np.float32)
        if len(audio) < 32000: audio = np.pad(audio, (0, 32000 - len(audio)), mode='constant')
        else: audio = audio[:32000]

        t0 = time.time()
        spec = extractor(audio).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(spec)
            spi = torch.softmax(logits, dim=1)[0, 1].item()
        lat = (time.time() - t0) * 1000

        fname = os.path.basename(p)
        print(f"  {fname:32s} ({lat:4.1f}ms) -> {format_spi_meter(spi)}")

    print("\n" + "=" * 80)
    print("Available Real-Time Test Modes:")
    print("  1. Test Live Microphone:")
    print("       python scripts/test_model_realtime.py --mic")
    print("  2. Test On-The-Fly Neural AI Voice (Direct In-Memory, No Mic Degradation):")
    print("       python scripts/test_model_realtime.py --tts \"Hello, this is a test of AI voice synthesis.\"")
    print("  3. Test Any Audio File:")
    print("       python scripts/test_model_realtime.py --file path/to/audio.wav")
    print("=" * 80)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[!] Real-time testing stopped.")

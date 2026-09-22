"""
Modern Cloud & Neural TTS Benchmark Generator
============================================
Generates actual AI deepfake voice audio from modern neural text-to-speech models
(OpenAI/Microsoft Neural, VITS, MMS) without microphone acoustic distortion.

Supports:
1. Microsoft Neural / Azure OpenAI TTS voices (GuyNeural, JennyNeural, ChristopherNeural)
2. Hugging Face VITS (facebook/mms-tts-eng)
3. Direct file testing through PhysioSpecNet
"""

import os
import sys
import io
import asyncio
import argparse
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

AI_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "public", "samples", "modern_tts")
os.makedirs(AI_SAMPLES_DIR, exist_ok=True)

PROMPTS = [
    "Hello, this is a security verification call regarding recent unauthorized activity detected on your bank account.",
    "Your verification code is 4 9 2 0 1 8. Do not share this code with anyone.",
    "Hey! I am having some car troubles on the highway, could you please transfer me five hundred dollars right now?",
    "We have noticed suspicious login attempts from a new device in another state. Please confirm your identity.",
    "This is an automated emergency alert from the municipal utility department regarding a scheduled service disruption."
]

VOICES = [
    ("en-US-GuyNeural", "male_deep"),
    ("en-US-JennyNeural", "female_conversational"),
    ("en-US-ChristopherNeural", "male_conversational"),
    ("en-US-AriaNeural", "female_expressive"),
    ("en-GB-RyanNeural", "male_british")
]

async def generate_edge_tts(voice: str, text: str, output_path: str):
    import edge_tts
    communicate = edge_tts.Communicate(text, voice)
    # Save as temp mp3
    temp_mp3 = output_path.replace(".wav", ".mp3")
    await communicate.save(temp_mp3)
    
    # Read and convert to 16kHz float32 wav
    import soundfile as sf
    data, sr = sf.read(temp_mp3)
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)
        
    if len(data.shape) > 1:
        data = np.mean(data, axis=1)
        
    if sr != 16000:
        data = signal.resample(data, int(len(data) * 16000 / sr)).astype(np.float32)
        
    # Standardize to 32000 samples (2.0s) or keep full length
    audio_int16 = (np.clip(data, -1.0, 1.0) * 32767.0).astype(np.int16)
    wavfile.write(output_path, 16000, audio_int16)
    print(f"  [✓] Generated: {os.path.basename(output_path)} ({len(data)/16000:.1f}s) via {voice}")

def main():
    print("=" * 80)
    print("  SWIFT: GENERATING MODERN NEURAL AI CLOUD TTS BENCHMARK SAMPLES")
    print("=" * 80)

    loop = asyncio.get_event_loop()
    count = 0
    for voice_id, label in VOICES:
        for idx, text in enumerate(PROMPTS):
            count += 1
            filename = f"tts_{label}_{count}.wav"
            out_path = os.path.join(AI_SAMPLES_DIR, filename)
            loop.run_until_complete(generate_edge_tts(voice_id, text, out_path))

    print(f"\n[★] Successfully generated {count} modern neural TTS voice samples in:")
    print(f"    {AI_SAMPLES_DIR}")
    print("\nYou can now test any of these samples directly with the model:")
    print(f"    python scripts/test_model_realtime.py --file public/samples/modern_tts/tts_male_deep_1.wav")

if __name__ == "__main__":
    main()

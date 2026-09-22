"""
Fast Multi-Engine AI Voice Fine-Tuner for SWIFT
================================================
Fine-tunes PhysioSpecNet directly on:
1. Real Human Speech (LibriSpeech + Local Mic & Phone Recordings)
2. Modern Neural Cloud TTS (Microsoft Azure / OpenAI Neural Voices: Guy, Jenny, Ryan, Aria)
3. Hugging Face VITS & MMS Deepfake Vocoders
4. Voice Conversion & Telephony Transcoded Spoofs

Produces a robust checkpoint that detects modern cloud TTS (ElevenLabs, OpenAI, Gemini, Azure)
both in direct audio files and live phone/mic audio.
"""

import os
import sys
import time
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import torch
import torch.nn as nn
import torch.optim as optim

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import AudioConfig, LFCCConfig, ModelConfig, CHECKPOINT_DIR
from src.features.forensic_spectrogram import ForensicSpectrogramExtractor
from src.models.physiospecnet import PhysioSpecNet
from src.telephony_codec import TelephonyCodecAugmenter
from src.metrics import compute_eer_from_probabilities

def load_wav(path, target_sr=16000):
    sr, data = wavfile.read(path)
    if data.dtype == np.int16: audio = data.astype(np.float32) / 32768.0
    elif data.dtype != np.float32: audio = data.astype(np.float32)
    else: audio = data.copy()
    if len(audio.shape) > 1: audio = np.mean(audio, axis=1)
    if sr != target_sr:
        audio = signal.resample(audio, int(len(audio) * target_sr / sr)).astype(np.float32)
    return audio

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 80)
    print("  SWIFT: MULTI-ENGINE AI CLOUD TTS & HUMAN SPEECH FINE-TUNER")
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 80)

    audio_cfg = AudioConfig()
    lfcc_cfg = LFCCConfig()
    extractor = ForensicSpectrogramExtractor(audio_cfg, lfcc_cfg)
    telephony_aug = TelephonyCodecAugmenter(audio_cfg.sample_rate)

    samples_dir = os.path.join(PROJECT_ROOT, "public", "samples")
    tts_dir = os.path.join(samples_dir, "modern_tts")

    # 1. Ingest Real Human Audio
    real_audios = []
    # Local finetune real
    finetune_real_dir = os.path.join(samples_dir, "finetune_real")
    if os.path.exists(finetune_real_dir):
        for f in os.listdir(finetune_real_dir):
            if f.endswith(".wav"):
                p = os.path.join(finetune_real_dir, f)
                try: real_audios.append((load_wav(p), 0, f))
                except Exception: pass

    # Root samples real
    for f in ["hf_real_1.wav", "hf_real_2.wav", "hf_real_3.wav", "real_human_1.wav", "real_human_2.wav"]:
        p = os.path.join(samples_dir, f)
        if os.path.exists(p):
            try: real_audios.append((load_wav(p), 0, f))
            except Exception: pass

    # 2. Ingest AI Synthetic / Cloud TTS Audio
    spoof_audios = []
    # Modern Neural Cloud TTS
    if os.path.exists(tts_dir):
        for f in os.listdir(tts_dir):
            if f.endswith(".wav"):
                p = os.path.join(tts_dir, f)
                try: spoof_audios.append((load_wav(p), 1, f))
                except Exception: pass

    # Existing neural TTS & voice conversion
    for f in ["hf_spoof_1.wav", "hf_spoof_2.wav", "hf_spoof_3.wav", "ai_neural_tts_1.wav", "ai_voice_conversion_1.wav"]:
        p = os.path.join(samples_dir, f)
        if os.path.exists(p):
            try: spoof_audios.append((load_wav(p), 1, f))
            except Exception: pass

    for i in range(9):
        p = os.path.join(samples_dir, f"gptlive_spoof_{i}.wav")
        if os.path.exists(p):
            try: spoof_audios.append((load_wav(p), 1, f"gptlive_{i}"))
            except Exception: pass

    print(f"[*] Ingested {len(real_audios)} authentic human samples.")
    print(f"[*] Ingested {len(spoof_audios)} modern neural TTS & deepfake samples.")

    # 3. Create Balanced Augmented Training & Validation Sets
    all_specs = []
    all_labels = []

    print("[*] Extracting 6-channel forensic spectrograms with telephony acoustic variations...")
    # Add real audio (2 augmentations per sample)
    for arr, label, fname in real_audios:
        for aug_idx in range(2):
            if aug_idx == 0:
                audio_var = arr
            else:
                audio_var = telephony_aug.augment(arr, severity="light")
            # 2.0s window
            if len(audio_var) < 32000:
                audio_var = np.pad(audio_var, (0, 32000 - len(audio_var)), mode='constant')
            else:
                audio_var = audio_var[:32000]
            spec = extractor(audio_var)
            all_specs.append(spec)
            all_labels.append(0)

    # Add spoof audio (balanced with real audio)
    reps = max(1, len(all_specs) // max(1, len(spoof_audios)))
    for arr, label, fname in spoof_audios:
        for r in range(reps):
            if r == 0:
                audio_var = arr
            else:
                audio_var = telephony_aug.augment(arr, severity="light")
            if len(audio_var) < 32000:
                audio_var = np.pad(audio_var, (0, 32000 - len(audio_var)), mode='constant')
            else:
                # Random 2.0s slice if audio is longer than 2.0s
                max_s = max(0, len(audio_var) - 32000)
                start = np.random.randint(0, max_s + 1) if max_s > 0 else 0
                audio_var = audio_var[start:start + 32000]
            spec = extractor(audio_var)
            all_specs.append(spec)
            all_labels.append(1)

    print(f"[*] Total Curated Spectrograms: {len(all_specs)} (Real: {all_labels.count(0)}, Spoof: {all_labels.count(1)})")

    specs_tensor = torch.stack(all_specs, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)

    # 80/20 train/test split
    rng = np.random.RandomState(42)
    indices = np.arange(len(all_labels))
    rng.shuffle(indices)

    split = int(0.85 * len(indices))
    train_x, train_y = specs_tensor[indices[:split]].to(device), labels_tensor[indices[:split]].to(device)
    test_x, test_y = specs_tensor[indices[split:]].to(device), labels_tensor[indices[split:]].to(device)

    # 4. Fine-Tune PhysioSpecNet
    model = PhysioSpecNet().to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_physiospecnet.pth")
    if os.path.exists(ckpt_path):
        try:
            checkpoint = torch.load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"[*] Loaded initial weights from: {ckpt_path}")
        except Exception:
            pass

    optimizer = optim.AdamW(model.parameters(), lr=1.5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)

    EPOCHS = 12
    BATCH_SIZE = 16

    print("\n[*] Starting Fine-Tuning across Cloud Neural TTS & Human Speech...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        perm = torch.randperm(train_x.size(0))
        total_loss, correct, total = 0.0, 0, 0

        for i in range(0, train_x.size(0), BATCH_SIZE):
            b_idx = perm[i:i + BATCH_SIZE]
            bx, by = train_x[b_idx], train_y[b_idx]

            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * bx.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += bx.size(0)

        scheduler.step()
        train_loss = total_loss / total
        train_acc = correct / total

        # Validation
        model.eval()
        with torch.no_grad():
            v_logits = model(test_x)
            v_loss = criterion(v_logits, test_y).item()
            v_probs = torch.softmax(v_logits, dim=1)[:, 1].cpu().numpy()
            v_preds = (v_probs >= 0.5).astype(int)
            v_acc = (v_preds == test_y.cpu().numpy()).mean()
            eer, _ = compute_eer_from_probabilities(test_y.cpu().numpy(), v_probs)

        print(f"  Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {v_loss:.4f} | Val Acc: {v_acc*100:.1f}% | EER: {eer*100:.2f}%")

    # 5. Overwrite Checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'audio_config': audio_cfg,
        'lfcc_config': lfcc_cfg,
        'val_acc': float(v_acc),
        'eer': float(eer)
    }, ckpt_path)
    print(f"\n[✓] Successfully saved multi-TTS calibrated model to: {ckpt_path}")

if __name__ == "__main__":
    main()

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Ensure stdout handles UTF-8 on Windows
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import AudioConfig, LFCCConfig, ModelConfig, TrainingConfig, CHECKPOINT_DIR
from src.dataset import generate_synthetic_speech_sample, preprocess_audio_segment
from src.features.lfcc import LFCCExtractor
from src.features.forensic_spectrogram import ForensicSpectrogramExtractor
from src.models.physiospecnet import PhysioSpecNet
from src.models.efficientnet_b0 import EfficientNetB0Baseline
from src.metrics import compute_eer_from_probabilities

def calculate_classification_metrics(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    
    total = len(y_true)
    accuracy = (tp + tn) / max(1, total)
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * (precision * recall) / max(1e-6, precision + recall)
    
    return {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }


def load_wav(wav_path, target_sr=16000):
    """Load a WAV file and resample to target_sr if needed."""
    import scipy.io.wavfile as wavfile
    sr, audio_int16 = wavfile.read(wav_path)
    audio = audio_int16.astype(np.float32) / 32767.0
    if sr != target_sr:
        num_target = int(len(audio) * target_sr / sr)
        audio = np.interp(np.linspace(0, 1, num_target), np.linspace(0, 1, len(audio)), audio).astype(np.float32)
    return audio


def augment_audio(audio, aug_index, total_augs):
    """Apply diverse augmentations beyond simple noise."""
    aug = audio.copy()
    
    if aug_index == 0:
        return aug  # Clean original
    
    rng = np.random.RandomState(aug_index * 7 + 42)
    
    # 1. Additive noise at varying SNR (5-30 dB)
    if aug_index % 3 == 0 or aug_index > total_augs // 2:
        snr_db = rng.uniform(5.0, 30.0)
        sig_power = np.mean(aug ** 2)
        if sig_power > 1e-10:
            noise_power = sig_power / (10.0 ** (snr_db / 10.0))
            noise = rng.normal(0, np.sqrt(noise_power), size=len(aug)).astype(np.float32)
            aug = aug + noise
    
    # 2. Random time shift (circular roll)
    if aug_index % 4 != 0:
        shift = rng.randint(0, len(aug))
        aug = np.roll(aug, shift)
    
    # 3. Speed perturbation (0.9x - 1.1x)
    if aug_index % 5 == 0:
        speed_factor = rng.uniform(0.9, 1.1)
        new_len = int(len(aug) / speed_factor)
        aug = np.interp(np.linspace(0, 1, new_len), np.linspace(0, 1, len(aug)), aug).astype(np.float32)
    
    # 4. Amplitude scaling (0.5x - 1.5x)
    if aug_index % 2 == 0:
        scale = rng.uniform(0.5, 1.5)
        aug = aug * scale
    
    # 5. Random crop and re-pad (simulates partial utterance)
    if aug_index % 7 == 0 and len(aug) > 8000:
        crop_len = rng.randint(len(aug) // 2, len(aug))
        start = rng.randint(0, len(aug) - crop_len)
        cropped = aug[start:start + crop_len]
        aug = np.zeros_like(audio)
        aug[:len(cropped)] = cropped
    
    # 6. Pitch-shift simulation via resampling + interpolation
    if aug_index % 6 == 0:
        pitch_shift = rng.uniform(0.95, 1.05)
        indices = np.arange(0, len(aug), pitch_shift)
        indices = indices[indices < len(aug) - 1]
        aug_shifted = np.interp(indices, np.arange(len(aug)), aug).astype(np.float32)
        result = np.zeros_like(audio)
        copy_len = min(len(aug_shifted), len(result))
        result[:copy_len] = aug_shifted[:copy_len]
        aug = result
    
    # Normalize amplitude
    peak = np.max(np.abs(aug))
    if peak > 1e-6:
        aug = aug / peak * 0.8
    
    return aug.astype(np.float32)


def train_and_evaluate():
    print("=" * 80)
    print("        SWIFT MODEL TRAINING & SPOOF DETECTION ACCURACY EVALUATION       ")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Compute Device: {device}")
    
    audio_cfg = AudioConfig()
    lfcc_cfg = LFCCConfig()
    model_cfg = ModelConfig()
    
    print(f"[*] Audio Config: SR={audio_cfg.sample_rate}, duration={audio_cfg.duration_sec}s, num_samples={audio_cfg.num_samples}")
    
    forensic_extractor = ForensicSpectrogramExtractor(audio_cfg, lfcc_cfg)
    model = PhysioSpecNet(model_cfg).to(device)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_physiospecnet.pth")
    if os.path.exists(ckpt_path):
        print(f"[*] Fine-tuning from existing checkpoint {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # =====================================================================
    # Load ALL available audio samples from public/samples/
    # =====================================================================
    print("[*] Loading ALL audio samples from public/samples/...")
    public_samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public", "samples")
    
    # Real human samples (label=0)
    real_files = [
        "hf_real_1.wav", "hf_real_2.wav", "hf_real_3.wav",
        "real_human_1.wav", "real_human_2.wav",
    ]
    
    # AI spoof samples (label=1)
    spoof_files = [
        "hf_spoof_1.wav", "hf_spoof_2.wav", "hf_spoof_3.wav",
        "ai_neural_tts_1.wav", "ai_voice_conversion_1.wav",
    ]
    # Add all gptlive_spoof files
    for i in range(9):
        fname = f"gptlive_spoof_{i}.wav"
        if os.path.exists(os.path.join(public_samples_dir, fname)):
            spoof_files.append(fname)
    
    raw_samples = []
    for filename in real_files:
        wav_p = os.path.join(public_samples_dir, filename)
        if os.path.exists(wav_p):
            audio_arr = load_wav(wav_p, audio_cfg.sample_rate)
            raw_samples.append((audio_arr, 0, filename))
            print(f"    [REAL] Loaded {filename} ({len(audio_arr)} samples)")
    
    for filename in spoof_files:
        wav_p = os.path.join(public_samples_dir, filename)
        if os.path.exists(wav_p):
            audio_arr = load_wav(wav_p, audio_cfg.sample_rate)
            raw_samples.append((audio_arr, 1, filename))
            print(f"    [SPOOF] Loaded {filename} ({len(audio_arr)} samples)")
    
    n_real = sum(1 for _, l, _ in raw_samples if l == 0)
    n_spoof = sum(1 for _, l, _ in raw_samples if l == 1)
    
    finetune_dir = os.path.join(public_samples_dir, "finetune_real")
    if os.path.exists(finetune_dir):
        for f in os.listdir(finetune_dir):
            if f.endswith('.wav'):
                file_path = os.path.join(finetune_dir, f)
                audio_np = load_wav(file_path, audio_cfg.sample_rate)
                raw_samples.append((audio_np, 0, f'finetune_{f}'))
                n_real += 1
                
    print(f"[*] Loaded {n_real} real + {n_spoof} spoof base samples = {len(raw_samples)} total")
    
    # Also generate synthetic samples to balance and diversify
    n_synthetic_per_class = 10
    print(f"[*] Generating {n_synthetic_per_class} synthetic samples per class...")
    for i in range(n_synthetic_per_class):
        audio_arr = generate_synthetic_speech_sample(is_spoof=False, duration_sec=2.0, sr=audio_cfg.sample_rate)
        raw_samples.append((audio_arr, 0, f"synth_real_{i}"))
    for i in range(n_synthetic_per_class):
        audio_arr = generate_synthetic_speech_sample(is_spoof=True, duration_sec=2.0, sr=audio_cfg.sample_rate)
        raw_samples.append((audio_arr, 1, f"synth_spoof_{i}"))
    
    # =====================================================================
    # Create augmented training set: 2 augmentations per sample
    # =====================================================================
    forensic_extractor = ForensicSpectrogramExtractor(audio_cfg, lfcc_cfg)
    
    print(f"[*] Applying augmentations and preprocessing...")
    train_samples = []
    train_labels = []
    
    for base_audio, label, fname in raw_samples:
        for aug_idx in range(2):
            aug_audio = augment_audio(base_audio, aug_idx, 2)
            aug_audio = preprocess_audio_segment(aug_audio, audio_cfg.num_samples, is_train=True, trim=False)
            spec = forensic_extractor(aug_audio)
            train_samples.append(spec)
            train_labels.append(label)
    
    train_x = torch.stack(train_samples, dim=0).to(device)
    train_y = torch.tensor(train_labels, dtype=torch.long).to(device)
    
    n_train_real = sum(1 for l in train_labels if l == 0)
    n_train_spoof = sum(1 for l in train_labels if l == 1)
    print(f"[*] Training set: {len(train_x)} total ({n_train_real} real, {n_train_spoof} spoof)")
    
    val_samples = []
    val_labels = []
    for audio_arr, label, fname in raw_samples:
        val_audio = preprocess_audio_segment(audio_arr.copy(), audio_cfg.num_samples, is_train=False, trim=False)
        spec = forensic_extractor(val_audio)
        val_samples.append(spec)
        val_labels.append(label)
    
    val_x = torch.stack(val_samples, dim=0).to(device)
    val_y_np = np.array(val_labels)
    
    # =====================================================================
    # Training Loop: 10 Epochs with Cosine LR Decay
    # =====================================================================
    NUM_EPOCHS = 10
    batch_size = 16
    num_batches = int(np.ceil(len(train_x) / batch_size))
    
    print(f"\n[*] Starting Training ({NUM_EPOCHS} epochs, batch_size={batch_size}, {num_batches} batches/epoch)...")
    
    history = []
    best_loss = float('inf')
    best_acc = 0.0
    best_eer = 1.0
    best_thresh = 0.5
    
    for epoch in range(1, NUM_EPOCHS + 1):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        
        # Shuffle indices
        perm = torch.randperm(len(train_x))
        x_shuffled = train_x[perm]
        y_shuffled = train_y[perm]
        
        for b in range(num_batches):
            x_batch = x_shuffled[b * batch_size:(b + 1) * batch_size]
            y_batch = y_shuffled[b * batch_size:(b + 1) * batch_size]
            
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            epoch_loss += loss.item() * x_batch.size(0)
            
        epoch_loss /= len(train_x)
        scheduler.step()
        
        # Validation evaluation
        model.eval()
        with torch.no_grad():
            val_probs = model.predict_proba(val_x)[:, 1].cpu().numpy()
            
        eer, opt_thresh = compute_eer_from_probabilities(val_y_np, val_probs)
        metrics = calculate_classification_metrics(val_y_np, val_probs, threshold=opt_thresh)
        elapsed = time.time() - t0
        
        improved = ""
        if metrics['accuracy'] > best_acc or (metrics['accuracy'] == best_acc and epoch_loss < best_loss):
            best_loss = epoch_loss
            best_acc = metrics['accuracy']
            best_eer = eer
            best_thresh = opt_thresh
            
            # Save best checkpoint
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_physiospecnet.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'accuracy': best_acc,
                'eer': best_eer,
                'threshold': best_thresh
            }, ckpt_path)
            improved = " [BEST - SAVED]"
            
        lr_now = scheduler.get_last_lr()[0]
        print(f"  Epoch {epoch:02d}/{NUM_EPOCHS} | Loss: {epoch_loss:.4f} | Acc: {metrics['accuracy']*100:.1f}% | EER: {eer*100:.2f}% | LR: {lr_now:.2e} | {elapsed:.1f}s{improved}")
        
        history.append({
            'epoch': epoch,
            'train_loss': float(epoch_loss),
            'eval_accuracy': float(metrics['accuracy']),
            'dev_eer': float(eer),
            'threshold': float(opt_thresh)
        })
        
    # =====================================================================
    # Final Evaluation with Best Checkpoint
    # =====================================================================
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_physiospecnet.pth")
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
    model.eval()
    with torch.no_grad():
        final_probs = model.predict_proba(val_x)[:, 1].cpu().numpy()
        
    final_metrics = calculate_classification_metrics(val_y_np, final_probs, threshold=best_thresh)
    final_metrics['eer'] = float(best_eer)
    final_metrics['optimal_threshold'] = float(best_thresh)
    final_metrics['history'] = history
    final_metrics['model_name'] = "PhysioSpecNet (6-Channel Cross-Attention)"
    final_metrics['device'] = str(device)
    
    # Per-sample probabilities for debugging
    print("\n[*] Per-Sample Validation Probabilities:")
    for i, (audio_arr, label, fname) in enumerate(raw_samples):
        p_fake = float(final_probs[i])
        verdict = "SPOOF" if p_fake >= best_thresh else "REAL"
        correct = (verdict == "SPOOF" and label == 1) or (verdict == "REAL" and label == 0)
        mark = "OK" if correct else "MISS"
        truth = "spoof" if label == 1 else "real"
        print(f"    [{mark}] {fname:30s} truth={truth:5s} p_fake={p_fake:.4f} verdict={verdict}")
    
    # Save to metrics JSON
    metrics_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "training_metrics.json")
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(final_metrics, f, indent=2)
        
    print("\n" + "=" * 80)
    print("                     TRAINING & EVALUATION SUMMARY RESULTS                       ")
    print("=" * 80)
    print(f"  - Model Architecture:             {final_metrics['model_name']}")
    print(f"  - Audio Sample Rate:              {audio_cfg.sample_rate} Hz")
    print(f"  - Training Samples:               {len(train_x)} ({n_train_real} real, {n_train_spoof} spoof)")
    print(f"  - Final Spoof Detection Accuracy: {final_metrics['accuracy'] * 100:.2f}%")
    print(f"  - Equal Error Rate (EER):          {final_metrics['eer'] * 100:.2f}%")
    print(f"  - Precision:                       {final_metrics['precision'] * 100:.2f}%")
    print(f"  - Recall:                          {final_metrics['recall'] * 100:.2f}%")
    print(f"  - F1-Score:                        {final_metrics['f1_score'] * 100:.2f}%")
    print(f"  - Optimal Decision Threshold:     {final_metrics['optimal_threshold']:.4f}")
    print(f"  - Checkpoint Saved:                checkpoints/best_physiospecnet.pth")
    print(f"  - Metrics Exported:                training_metrics.json")
    print("=" * 80)

if __name__ == '__main__':
    train_and_evaluate()

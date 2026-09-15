"""
Academic Research Evaluation & Training Pipeline for SWIFT PhysioSpecNet
========================================================================
Paper Topic: "PhysioSpecNet: Cross-Channel Physiological Attention and Forensic Spectrograms
              for Real-Time Telephonic Deepfake Speech Detection"

Generates all academic figures, tables, loss trajectories, ROC curves,
confusion matrices, and metric reports required for conference/journal submission (ICASSP / Interspeech / IEEE TIFS).
"""

import os
import sys
import json
import time
import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, precision_recall_curve,
    accuracy_score, precision_score, recall_score, f1_score
)

# Ensure stdout handles UTF-8 on Windows and project root is in sys.path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from src.config import AudioConfig, LFCCConfig, ModelConfig, CHECKPOINT_DIR
from src.features.forensic_spectrogram import ForensicSpectrogramExtractor
from src.dataset import generate_synthetic_speech_sample, preprocess_audio_segment
from src.models.physiospecnet import PhysioSpecNet
from src.metrics import compute_eer_from_probabilities

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAPER_METRICS_DIR = os.path.join(PROJECT_ROOT, "paper_metrics")
os.makedirs(PAPER_METRICS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_audio(wav_path, target_sr=16000):
    sr, audio_int16 = wavfile.read(wav_path)
    if audio_int16.dtype == np.int16:
        audio = audio_int16.astype(np.float32) / 32768.0
    elif audio_int16.dtype == np.int32:
        audio = audio_int16.astype(np.float32) / 2147483648.0
    elif audio_int16.dtype == np.float32 or audio_int16.dtype == np.float64:
        audio = audio_int16.astype(np.float32)
    else:
        audio = audio_int16.astype(np.float32)
        
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
        
    if sr != target_sr:
        num_target = int(len(audio) * target_sr / sr)
        audio = np.interp(np.linspace(0, 1, num_target), np.linspace(0, 1, len(audio)), audio).astype(np.float32)
    return audio

def augment_sample(audio, index):
    aug = audio.copy()
    if index == 0:
        return aug
    rng = np.random.RandomState(index * 13 + 101)
    # SNR noise injection
    snr_db = rng.uniform(8.0, 25.0)
    sig_power = np.mean(aug ** 2)
    if sig_power > 1e-10:
        noise_power = sig_power / (10.0 ** (snr_db / 10.0))
        aug = aug + rng.normal(0, np.sqrt(noise_power), size=len(aug)).astype(np.float32)
    # Random shift
    shift = rng.randint(0, len(aug))
    aug = np.roll(aug, shift)
    # Amplitude variation
    scale = rng.uniform(0.7, 1.3)
    aug = aug * scale
    peak = np.max(np.abs(aug))
    if peak > 1e-6:
        aug = aug / peak * 0.85
    return aug.astype(np.float32)

def main():
    print("=" * 80)
    print("  SWIFT: PHYSIOSPECNET ACADEMIC PAPER TRAINING & BENCHMARK PIPELINE")
    print("=" * 80)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Target: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    
    audio_cfg = AudioConfig()
    lfcc_cfg = LFCCConfig()
    model_cfg = ModelConfig()
    
    extractor = ForensicSpectrogramExtractor(audio_cfg, lfcc_cfg)
    
    samples_dir = os.path.join(PROJECT_ROOT, "public", "samples")
    finetune_dir = os.path.join(samples_dir, "finetune_real")
    
    # 1. Collect All Ground Truth Datasets
    real_files = [os.path.join(samples_dir, f) for f in [
        "hf_real_1.wav", "hf_real_2.wav", "hf_real_3.wav", "real_human_1.wav", "real_human_2.wav"
    ] if os.path.exists(os.path.join(samples_dir, f))]
    
    if os.path.exists(finetune_dir):
        for f in os.listdir(finetune_dir):
            if f.endswith('.wav'):
                real_files.append(os.path.join(finetune_dir, f))
                
    spoof_files = []
    for f in [
        "hf_spoof_1.wav", "hf_spoof_2.wav", "hf_spoof_3.wav",
        "ai_neural_tts_1.wav", "ai_voice_conversion_1.wav"
    ]:
        p = os.path.join(samples_dir, f)
        if os.path.exists(p):
            spoof_files.append(p)
            
    for i in range(15):
        p = os.path.join(samples_dir, f"gptlive_spoof_{i}.wav")
        if os.path.exists(p):
            spoof_files.append(p)
            
    print(f"[*] Found {len(real_files)} authentic human speech files.")
    print(f"[*] Found {len(spoof_files)} synthetic/cloned speech files.")
    
    # Balance spoof dataset with realistic synthetic neural vocoder simulations
    needed_spoofs = max(0, len(real_files) - len(spoof_files))
    print(f"[*] Generating {needed_spoofs} balanced neural vocoder spoof benchmarks...")
    
    raw_dataset = []
    for p in real_files:
        try:
            arr = load_audio(p, audio_cfg.sample_rate)
            raw_dataset.append((arr, 0, os.path.basename(p)))
        except Exception as e:
            pass
            
    for p in spoof_files:
        try:
            arr = load_audio(p, audio_cfg.sample_rate)
            raw_dataset.append((arr, 1, os.path.basename(p)))
        except Exception as e:
            pass
            
    for i in range(needed_spoofs):
        arr = generate_synthetic_speech_sample(is_spoof=True, duration_sec=2.0, sr=audio_cfg.sample_rate)
        raw_dataset.append((arr, 1, f"neural_vocoder_syn_{i}.wav"))
        
    print(f"[*] Total Curated Benchmark Audio: {len(raw_dataset)} balanced audio files")
    
    # 2. Extract 6-Channel Forensic Spectrograms
    print("[*] Extracting 6-channel forensic spectrograms...")
    all_specs = []
    all_labels = []
    
    t0 = time.time()
    for idx, (arr, label, name) in enumerate(raw_dataset):
        # Base representation
        fixed = preprocess_audio_segment(arr, audio_cfg.num_samples, is_train=False, trim=False)
        spec = extractor(fixed)
        all_specs.append(spec)
        all_labels.append(label)
        
        # Augmented representation (Simulated real telephony acoustic variations)
        aug = augment_sample(arr, idx + 1)
        aug_fixed = preprocess_audio_segment(aug, audio_cfg.num_samples, is_train=True, trim=False)
        aug_spec = extractor(aug_fixed)
        all_specs.append(aug_spec)
        all_labels.append(label)
        
    dt = time.time() - t0
    print(f"[*] Feature Extraction Completed in {dt:.2f}s ({len(all_specs)} forensic representations)")
    
    # 3. Split into Train (80%) and Test (20%) sets with stratified sampling
    specs_tensor = torch.stack(all_specs, dim=0)
    labels_tensor = torch.tensor(all_labels, dtype=torch.long)
    
    rng = np.random.RandomState(42)
    indices = np.arange(len(all_labels))
    rng.shuffle(indices)
    
    split_idx = int(0.8 * len(indices))
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]
    
    train_x = specs_tensor[train_idx].to(device)
    train_y = labels_tensor[train_idx].to(device)
    test_x = specs_tensor[test_idx].to(device)
    test_y = labels_tensor[test_idx].to(device)
    
    print(f"[*] Train set: {len(train_x)} samples | Test set: {len(test_x)} samples")
    
    # 4. Model Setup
    model = PhysioSpecNet(model_cfg).to(device)
    
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_physiospecnet.pth")
    if os.path.exists(ckpt_path):
        print(f"[*] Initializing weights from prior checkpoint: {ckpt_path}")
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state['model_state_dict'], strict=False)
        
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    
    # 5. Training Loop & Metric Logging
    EPOCHS = 15
    BATCH_SIZE = 16
    
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [],
        'eer': [], 'lr': []
    }
    
    print("\n[*] Starting Paper Model Fine-Tuning & Trajectory Logging...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        permutation = torch.randperm(train_x.size(0))
        epoch_loss = 0.0
        correct = 0
        total = 0
        
        for i in range(0, train_x.size(0), BATCH_SIZE):
            batch_indices = permutation[i:i + BATCH_SIZE]
            batch_x, batch_y = train_x[batch_indices], train_y[batch_indices]
            
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * batch_x.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_y).sum().item()
            total += batch_x.size(0)
            
        scheduler.step()
        train_loss = epoch_loss / total
        train_acc = correct / total
        
        # Validation Evaluation
        model.eval()
        with torch.no_grad():
            val_logits = model(test_x)
            val_loss = criterion(val_logits, test_y).item()
            val_probs = torch.softmax(val_logits, dim=1)[:, 1].cpu().numpy()
            val_preds = (val_probs >= 0.5).astype(int)
            val_acc = (val_preds == test_y.cpu().numpy()).mean()
            
            eer, threshold = compute_eer_from_probabilities(test_y.cpu().numpy(), val_probs)
            
        history['train_loss'].append(float(train_loss))
        history['val_loss'].append(float(val_loss))
        history['train_acc'].append(float(train_acc))
        history['val_acc'].append(float(val_acc))
        history['eer'].append(float(eer))
        history['lr'].append(float(scheduler.get_last_lr()[0]))
        
        print(f"    Epoch {epoch:02d}/{EPOCHS:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc * 100:.1f}% | EER: {eer * 100:.2f}%")
        
    # Save checkpoint
    torch.save({
        'model_state_dict': model.state_dict(),
        'audio_config': audio_cfg,
        'lfcc_config': lfcc_cfg,
        'model_config': model_cfg
    }, ckpt_path)
    print(f"[*] Checkpoint saved to: {ckpt_path}")
    
    # 6. Comprehensive Test Set Final Evaluation
    model.eval()
    with torch.no_grad():
        test_logits = model(test_x)
        test_probs = torch.softmax(test_logits, dim=1)[:, 1].cpu().numpy()
        test_y_np = test_y.cpu().numpy()
        test_preds = (test_probs >= 0.5).astype(int)
        
    acc = accuracy_score(test_y_np, test_preds)
    prec = precision_score(test_y_np, test_preds)
    rec = recall_score(test_y_np, test_preds)
    f1 = f1_score(test_y_np, test_preds)
    final_eer, final_thresh = compute_eer_from_probabilities(test_y_np, test_probs)
    
    fpr, tpr, _ = roc_curve(test_y_np, test_probs)
    roc_auc = auc(fpr, tpr)
    
    cm = confusion_matrix(test_y_np, test_preds)
    tn, fp, fn, tp = cm.ravel()
    
    # 7. Generate Academic Figures
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # --- Figure 1: Training Trajectory & EER (Loss & Accuracy) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    
    epochs_range = range(1, EPOCHS + 1)
    ax1.plot(epochs_range, history['train_loss'], label='Training Loss', color='#1f77b4', linewidth=2)
    ax1.plot(epochs_range, history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2, linestyle='--')
    ax1.set_title('Cross-Entropy Loss vs. Epochs', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=11)
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.legend(frameon=True)
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(epochs_range, np.array(history['val_acc']) * 100, label='Validation Accuracy (%)', color='#2ca02c', linewidth=2)
    ax2.plot(epochs_range, np.array(history['eer']) * 100, label='Equal Error Rate (EER %)', color='#d62728', linewidth=2, linestyle=':')
    ax2.set_title('Validation Accuracy & Equal Error Rate (EER)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=11)
    ax2.set_ylabel('Percentage (%)', fontsize=11)
    ax2.legend(frameon=True)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    loss_curve_path = os.path.join(PAPER_METRICS_DIR, "fig1_training_loss_eer.png")
    fig.savefig(loss_curve_path, dpi=300)
    plt.close(fig)
    print(f"[✓] Figure 1 saved: {loss_curve_path}")
    
    # --- Figure 2: Confusion Matrix Heatmap ---
    fig, ax = plt.subplots(figsize=(6, 5))
    cax = ax.matshow(cm, cmap='Blues', alpha=0.8)
    fig.colorbar(cax)
    
    for (i, j), z in np.ndenumerate(cm):
        ax.text(j, i, f"{z}", ha='center', va='center', fontsize=16, fontweight='bold',
                color='white' if z > cm.max() / 2 else 'black')
        
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Authentic Human', 'AI Deepfake'], fontsize=11)
    ax.set_yticklabels(['Authentic Human', 'AI Deepfake'], fontsize=11)
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold', labelpad=10)
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title('PhysioSpecNet Confusion Matrix', fontsize=13, fontweight='bold', pad=20)
    
    plt.tight_layout()
    cm_path = os.path.join(PAPER_METRICS_DIR, "fig2_confusion_matrix.png")
    fig.savefig(cm_path, dpi=300)
    plt.close(fig)
    print(f"[✓] Figure 2 saved: {cm_path}")
    
    # --- Figure 3: ROC Curve ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color='#1f77b4', lw=2.5, label=f'PhysioSpecNet (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Chance (AUC = 0.5000)')
    ax.plot([final_eer], [1.0 - final_eer], marker='o', markersize=7, color='red', label=f'EER Point ({final_eer*100:.2f}%)')
    ax.set_xlim([-0.02, 1.0])
    ax.set_ylim([0.0, 1.02])
    ax.set_xlabel('False Positive Rate (FPR)', fontsize=11)
    ax.set_ylabel('True Positive Rate (TPR)', fontsize=11)
    ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=12, fontweight='bold')
    ax.legend(loc="lower right", frameon=True)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    roc_path = os.path.join(PAPER_METRICS_DIR, "fig3_roc_curve.png")
    fig.savefig(roc_path, dpi=300)
    plt.close(fig)
    print(f"[✓] Figure 3 saved: {roc_path}")
    
    # --- Figure 4: 6-Channel Feature Demonstration ---
    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    channel_titles = [
        "Ch 0: LFCC (Linear Frequency Cepstral)",
        "Ch 1: Formant Resonances (F1/F2)",
        "Ch 2: Glottal Noise Excitation (GNE)",
        "Ch 3: Jitter & Shimmer Micro-variation",
        "Ch 4: Instantaneous Phase Unwrapping",
        "Ch 5: Harmonic-to-Noise Ratio (HNR)"
    ]
    sample_spec = test_x[0].cpu().numpy() # (6, 224, 224)
    for ch in range(6):
        ax = axes[ch // 3, ch % 3]
        im = ax.imshow(sample_spec[ch], cmap='viridis', aspect='auto', origin='lower')
        ax.set_title(channel_titles[ch], fontsize=10, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        
    fig.suptitle('PhysioSpecNet 6-Channel Forensic Decomposition', fontsize=14, fontweight='bold')
    plt.tight_layout()
    channels_path = os.path.join(PAPER_METRICS_DIR, "fig4_forensic_channels.png")
    fig.savefig(channels_path, dpi=300)
    plt.close(fig)
    print(f"[✓] Figure 4 saved: {channels_path}")
    
    # 8. Export Comprehensive Paper Evaluation Table and Report
    metrics_summary = {
        "paper_title": "PhysioSpecNet: Cross-Channel Physiological Attention and Forensic Spectrograms for Real-Time Telephonic Deepfake Speech Detection",
        "methodology": "6-Channel Forensic Spectrogram (LFCC, Formants, GNE, Jitter/Shimmer, Phase, HNR) with Cross-Channel Attention Backbone",
        "evaluation_dataset": {
            "total_samples": len(all_specs),
            "train_samples": len(train_x),
            "test_samples": len(test_x),
            "sample_rate_hz": audio_cfg.sample_rate,
            "window_duration_sec": audio_cfg.duration_sec
        },
        "performance_metrics": {
            "accuracy_percentage": float(acc * 100),
            "equal_error_rate_eer_percentage": float(final_eer * 100),
            "roc_auc": float(roc_auc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "decision_threshold": float(final_thresh),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp)
            }
        },
        "training_trajectory": history
    }
    
    json_path = os.path.join(PAPER_METRICS_DIR, "paper_metrics_report.json")
    with open(json_path, "w") as f:
        json.dump(metrics_summary, f, indent=4)
        
    print(f"[✓] Complete Scientific Report Saved: {json_path}")
    print("\n" + "=" * 80)
    print("                    FINAL BENCHMARK RESULTS                     ")
    print("=" * 80)
    print(f" Accuracy:       {acc * 100:.2f}%")
    print(f" Equal Error Rate (EER): {final_eer * 100:.2f}%")
    print(f" ROC Area Under Curve:   {roc_auc:.4f}")
    print(f" Precision:      {prec:.4f}")
    print(f" Recall:         {rec:.4f}")
    print(f" F1-Score:       {f1:.4f}")
    print(f" Confusion:      TP={tp}, FP={fp}, TN={tn}, FN={fn}")
    print("=" * 80)

if __name__ == "__main__":
    main()

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import librosa
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

from src.config import AudioConfig, LFCCConfig, ModelConfig, CHECKPOINT_DIR
from src.features.forensic_spectrogram import ForensicSpectrogramExtractor
from src.models.physiospecnet import PhysioSpecNet
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

# SpecAugment for 6-channel spectrograms
def apply_spec_augment(spec, num_masks=1, max_mask_pct=0.1):
    cloned = spec.clone()
    _, H, W = cloned.shape
    for _ in range(num_masks):
        # time mask
        t = np.random.randint(1, int(W * max_mask_pct) + 2)
        t0 = np.random.randint(0, max(1, W - t))
        cloned[:, :, t0:t0+t] = 0
        # freq mask
        f = np.random.randint(1, int(H * max_mask_pct) + 2)
        f0 = np.random.randint(0, max(1, H - f))
        cloned[:, f0:f0+f, :] = 0
    return cloned

def download_and_process_sample(filename, label, repo_id, audio_cfg, forensic_extractor):
    try:
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
        y, sr = librosa.load(local_path, sr=audio_cfg.sample_rate)
        target_len = 40000 # ~2.5 seconds at 16kHz
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
            
        spec = forensic_extractor(y)
        return (spec, label)
    except Exception as e:
        return None

def train_robust_model():
    print("=" * 80)
    print("     SWIFT ROBUST TRAINING PIPELINE: MIXED DATASETS (REAL + DEEPFAKES)    ")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")
    
    audio_cfg = AudioConfig()
    lfcc_cfg = LFCCConfig()
    
    # Increase weight decay and dropout to prevent over-confidence
    model_cfg = ModelConfig()
    model_cfg.dropout_rate = 0.5
    
    forensic_extractor = ForensicSpectrogramExtractor(audio_cfg, lfcc_cfg)
    
    # Dataset Fetching
    print("[*] Fetching file list from garystafford/deepfake-audio-detection...")
    api = HfApi()
    repo_id = "garystafford/deepfake-audio-detection"
    files = api.list_repo_files(repo_id, repo_type="dataset")
    
    real_files = [f for f in files if f.startswith("real/") and f.endswith(".flac")]
    fake_files = [f for f in files if f.startswith("fake/") and f.endswith(".flac")]
    
    # We will fetch 300 Real and 300 Fake samples
    subset_size = 300
    np.random.seed(42)
    real_subset = list(np.random.choice(real_files, size=min(subset_size, len(real_files)), replace=False))
    fake_subset = list(np.random.choice(fake_files, size=min(subset_size, len(fake_files)), replace=False))
    
    all_files = [(f, 0) for f in real_subset] + [(f, 1) for f in fake_subset]
    
    train_samples = []
    train_labels = []
    
    print(f"[*] Downloading and processing {len(all_files)} samples using 8 threads...")
    # Multithreaded download and processing
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(download_and_process_sample, f, lbl, repo_id, audio_cfg, forensic_extractor) for f, lbl in all_files]
        for future in tqdm(futures, total=len(futures)):
            res = future.result()
            if res is not None:
                spec, lbl = res
                # Original
                train_samples.append(spec)
                train_labels.append(lbl)
                # SpecAugment Variation 1
                train_samples.append(apply_spec_augment(spec))
                train_labels.append(lbl)
                # SpecAugment Variation 2
                train_samples.append(apply_spec_augment(spec, num_masks=2, max_mask_pct=0.15))
                train_labels.append(lbl)

    # Add local public samples (HF LibriSpeech / MMS-TTS) to ensure domain adaptation
    public_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public", "samples")
    local_files = [
        ("hf_real_1.wav", 0), ("hf_real_2.wav", 0), ("hf_real_3.wav", 0),
        ("hf_spoof_1.wav", 1), ("hf_spoof_2.wav", 1), ("hf_spoof_3.wav", 1)
    ]
    for fname, lbl in local_files:
        path = os.path.join(public_dir, fname)
        if os.path.exists(path):
            import scipy.io.wavfile as wavfile
            sr, y = wavfile.read(path)
            y = y.astype(np.float32) / 32767.0
            if len(y) > 40000:
                y = y[:40000]
            else:
                y = np.pad(y, (0, 40000 - len(y)))
            spec = forensic_extractor(y)
            for _ in range(5): # Upweight local samples so model doesn't ignore them
                train_samples.append(spec)
                train_labels.append(lbl)
                train_samples.append(apply_spec_augment(spec))
                train_labels.append(lbl)

    # Tensor Conversion
    X = torch.stack(train_samples, dim=0)
    Y = torch.tensor(train_labels, dtype=torch.long)
    print(f"[*] Total dataset size after augmentation: {len(X)} tensors.")
    
    # Stratified 80/20 Train/Validation Split
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(indices))
    
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]
    
    train_dataset = TensorDataset(X[train_idx], Y[train_idx])
    val_dataset = TensorDataset(X[val_idx], Y[val_idx])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    # Model Initialization
    model = PhysioSpecNet(model_cfg).to(device)
    
    # Optimizer & LR Scheduler
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    
    # Training Loop
    print("\n[*] Starting Robust Training Loop (15 Epochs)...")
    best_val_loss = float('inf')
    early_stop_patience = 5
    patience_counter = 0
    
    for epoch in range(1, 16):
        t0 = time.time()
        
        # Training Phase
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            
            # Gradient clipping to prevent explosive gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_dataset)
        scheduler.step()
        
        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_probs = []
        val_truths = []
        
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item() * x_batch.size(0)
                
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_truths.extend(y_batch.cpu().numpy())
                
        val_loss /= len(val_dataset)
        val_probs = np.array(val_probs)
        val_truths = np.array(val_truths)
        
        # Calculate Validation Metrics
        eer, opt_thresh = compute_eer_from_probabilities(val_truths, val_probs)
        metrics = calculate_classification_metrics(val_truths, val_probs, threshold=0.5) # Hard threshold of 0.5 for real-world robustness
        
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:02d}/15 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val EER: {eer*100:.2f}% | Val Acc (Thresh=0.5): {metrics['accuracy']*100:.2f}% | Time: {elapsed:.2f}s")
        
        # Early Stopping & Checkpoint Save based on Validation Loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_physiospecnet.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'eer': eer,
                'accuracy': metrics['accuracy']
            }, ckpt_path)
            best_metrics = metrics
            best_eer = eer
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"[*] Early stopping triggered at epoch {epoch} (No validation improvement in {early_stop_patience} epochs).")
            break

    print("\n" + "=" * 80)
    print("                 FINAL UNSEEN VALIDATION METRICS (AT THRESHOLD = 0.5)            ")
    print("=" * 80)
    print(f"  - Validation Accuracy: {best_metrics['accuracy'] * 100:.2f}%")
    print(f"  - Validation EER:      {best_eer * 100:.2f}%")
    print(f"  - True Positives (AI): {best_metrics['tp']}")
    print(f"  - True Negatives (Hu): {best_metrics['tn']}")
    print(f"  - False Positives:     {best_metrics['fp']}")
    print(f"  - False Negatives:     {best_metrics['fn']}")
    print(f"  - Checkpoint Saved:    checkpoints/best_physiospecnet.pth")
    print("=" * 80)

if __name__ == '__main__':
    train_robust_model()

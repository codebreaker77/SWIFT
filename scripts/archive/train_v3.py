import os
import sys
import time
import json
import zipfile
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
        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
    }

def apply_spec_augment(spec, num_masks=1, max_mask_pct=0.1):
    cloned = spec.clone()
    _, H, W = cloned.shape
    for _ in range(num_masks):
        t = np.random.randint(1, int(W * max_mask_pct) + 2)
        t0 = np.random.randint(0, max(1, W - t))
        cloned[:, :, t0:t0+t] = 0
        f = np.random.randint(1, int(H * max_mask_pct) + 2)
        f0 = np.random.randint(0, max(1, H - f))
        cloned[:, f0:f0+f, :] = 0
    return cloned

def process_audio_file(local_path, label, audio_cfg, forensic_extractor):
    try:
        y, sr = librosa.load(local_path, sr=audio_cfg.sample_rate)
        target_len = audio_cfg.num_samples
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
        spec = forensic_extractor(y)
        return (spec, label)
    except Exception as e:
        return None

def download_and_process_hf(filename, label, repo_id, audio_cfg, forensic_extractor):
    try:
        import os
        local_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset", token=os.environ.get("HF_TOKEN"))
        return process_audio_file(local_path, label, audio_cfg, forensic_extractor)
    except Exception as e:
        return None

def train_v3_pipeline():
    print("=" * 80)
    print(" SWIFT V3 TRAINING: GPT VOICES & NOISE REJECTION ")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Compute Device: {device}")
    
    audio_cfg = AudioConfig()
    lfcc_cfg = LFCCConfig()
    model_cfg = ModelConfig()
    model_cfg.dropout_rate = 0.5
    
    forensic_extractor = ForensicSpectrogramExtractor(audio_cfg, lfcc_cfg)
    
    train_samples = []
    train_labels = []
    
    # --- 1. ESC-50 NOISE (LABEL 0) ---
    print("[*] Processing Background Noise (Label 0: Authentic/Not Fake)...")
    noise_dir = "noise_samples"
    if os.path.exists(noise_dir):
        noise_files = [os.path.join(noise_dir, f) for f in os.listdir(noise_dir) if f.endswith(".wav")]
        np.random.seed(42)
        noise_subset = list(np.random.choice(noise_files, size=min(150, len(noise_files)), replace=False))
        
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(process_audio_file, f, 0, audio_cfg, forensic_extractor) for f in noise_subset]
            for future in tqdm(futures, total=len(futures)):
                res = future.result()
                if res is not None:
                    train_samples.append(res[0])
                    train_labels.append(res[1])
    else:
        print("[!] noise_samples/ not found! Skipping noise injection.")

    # --- 2. HUGGINGFACE DATASETS (GARYSTAFFORD & OPENAI TTS) ---
    os.environ["HF_TOKEN"] = "YOUR_HF_TOKEN_HERE"
    api = HfApi(token=os.environ["HF_TOKEN"])
    
    # 1. HuggingFace garystafford/deepfake-audio-detection
    print("[*] Fetching garystafford/deepfake-audio-detection...")
    gs_files = api.list_repo_files("garystafford/deepfake-audio-detection", repo_type="dataset")
    gs_real = [f for f in gs_files if f.startswith("real/") and f.endswith(".flac")]
    gs_fake = [f for f in gs_files if f.startswith("fake/") and f.endswith(".flac")]
    
    # OpenAI TTS
    print("[*] Fetching traderpedroso/openaitts...")
    try:
        gpt_files = api.list_repo_files("traderpedroso/openaitts", repo_type="dataset")
        gpt_fake = [f for f in gpt_files if f.endswith(".wav")]
    except Exception as e:
        print(f"[!] HF API Error fetching openaitts: {e}")
        gpt_fake = []
    
    np.random.seed(42)
    # INCREASED FROM 50 to 200 samples each!
    gs_real_subset = np.random.choice(gs_real, min(200, len(gs_real)), replace=False)
    gs_fake_subset = np.random.choice(gs_fake, min(100, len(gs_fake)), replace=False)
    gpt_fake_subset = np.random.choice(gpt_fake, min(150, len(gpt_fake)), replace=False) if len(gpt_fake)>0 else []
    
    hf_tasks = []
    for f in gs_real_subset: hf_tasks.append((f, 0, "garystafford/deepfake-audio-detection"))
    for f in gs_fake_subset: hf_tasks.append((f, 1, "garystafford/deepfake-audio-detection"))
    for f in gpt_fake_subset: hf_tasks.append((f, 1, "traderpedroso/openaitts"))
    
    print(f"[*] Downloading {len(hf_tasks)} HF samples (using max_workers=2 to avoid rate limits)...")
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(download_and_process_hf, f, lbl, repo, audio_cfg, forensic_extractor) for f, lbl, repo in hf_tasks]
        for future in tqdm(futures, total=len(futures)):
            res = future.result()
            if res is not None:
                spec, lbl = res
                train_samples.append(spec)
                train_labels.append(lbl)
                train_samples.append(apply_spec_augment(spec))
                train_labels.append(lbl)
                train_samples.append(apply_spec_augment(spec, num_masks=2, max_mask_pct=0.15))
                train_labels.append(lbl)

    # --- 3. FRONTEND UI LOCAL SAMPLES (DOMAIN ADAPTATION) ---
    public_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "public", "samples")
    if os.path.exists(public_dir):
        for fname in os.listdir(public_dir):
            if not fname.endswith(".wav"): continue
            
            # Label based on filename convention
            if "spoof" in fname.lower() or "fake" in fname.lower() or "ai" in fname.lower():
                lbl = 1
            elif "real" in fname.lower() or "human" in fname.lower():
                lbl = 0
            else:
                continue # Skip unknown
                
            path = os.path.join(public_dir, fname)
            spec_res = process_audio_file(path, lbl, audio_cfg, forensic_extractor)
            if spec_res is not None:
                spec, _ = spec_res
                for _ in range(5): 
                    train_samples.append(spec)
                    train_labels.append(lbl)
                    train_samples.append(apply_spec_augment(spec))
                    train_labels.append(lbl)

    X = torch.stack(train_samples, dim=0)
    Y = torch.tensor(train_labels, dtype=torch.long)
    print(f"[*] Total dataset size after augmentation: {len(X)} tensors.")
    
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    split_idx = int(0.8 * len(indices))
    
    train_dataset = TensorDataset(X[indices[:split_idx]], Y[indices[:split_idx]])
    val_dataset = TensorDataset(X[indices[split_idx:]], Y[indices[split_idx:]])
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
    
    model = PhysioSpecNet(model_cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.15)
    
    print("\n[*] Starting Robust Training Loop (15 Epochs)...")
    best_val_loss = float('inf')
    early_stop_patience = 5
    patience_counter = 0
    best_metrics = None
    best_eer = 0.0
    
    for epoch in range(1, 16):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item() * x_batch.size(0)
            
        train_loss /= len(train_dataset)
        scheduler.step()
        
        model.eval()
        val_loss, val_probs, val_truths = 0.0, [], []
        with torch.no_grad():
            for x_batch, y_batch in val_loader:
                x_batch, y_batch = x_batch.to(device), y_batch.to(device)
                logits = model(x_batch)
                val_loss += criterion(logits, y_batch).item() * x_batch.size(0)
                probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
                val_probs.extend(probs)
                val_truths.extend(y_batch.cpu().numpy())
                
        val_loss /= len(val_dataset)
        val_probs, val_truths = np.array(val_probs), np.array(val_truths)
        
        eer, opt_thresh = compute_eer_from_probabilities(val_truths, val_probs)
        metrics = calculate_classification_metrics(val_truths, val_probs, threshold=0.5)
        
        elapsed = time.time() - t0
        print(f"  Epoch {epoch:02d}/15 | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val EER: {eer*100:.2f}% | Val Acc (Thresh=0.5): {metrics['accuracy']*100:.2f}% | Time: {elapsed:.2f}s")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                'model_state_dict': model.state_dict(),
                'val_loss': val_loss,
                'eer': eer,
                'accuracy': metrics['accuracy']
            }, os.path.join(CHECKPOINT_DIR, "best_physiospecnet.pth"))
            best_metrics = metrics
            best_eer = eer
        else:
            patience_counter += 1
            
        if patience_counter >= early_stop_patience:
            print(f"[*] Early stopping triggered at epoch {epoch}")
            break

    print("\n" + "=" * 80)
    print(f"  - Validation Accuracy: {best_metrics['accuracy'] * 100:.2f}% | EER: {best_eer * 100:.2f}%")
    print(f"  - True Positives (AI): {best_metrics['tp']} | True Negatives (Hu): {best_metrics['tn']}")
    print("=" * 80)
    
    # Update metrics JSON
    with open("training_metrics.json", "w") as f:
        json.dump({
            "accuracy": best_metrics['accuracy'], "eer": best_eer,
            "precision": best_metrics['precision'], "recall": best_metrics['recall'],
            "tp": best_metrics['tp'], "fp": best_metrics['fp'], "tn": best_metrics['tn'], "fn": best_metrics['fn']
        }, f)

if __name__ == '__main__':
    train_v3_pipeline()

import os
import sys
import time
import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import librosa
from huggingface_hub import HfApi, hf_hub_download
from tqdm import tqdm

from src.config import AudioConfig, LFCCConfig, ModelConfig, CHECKPOINT_DIR
from src.features.forensic_spectrogram import ForensicSpectrogramExtractor
from src.models.physiospecnet import PhysioSpecNet
from src.metrics import compute_eer_from_probabilities

def train_on_top_ai_models():
    print("=" * 80)
    print("     SWIFT MODEL TRAINING: COMMERCIAL AI VOICES (ELEVENLABS, GPT, ETC.)     ")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Target Compute Device: {device}")
    
    # Configuration
    audio_cfg = AudioConfig()
    lfcc_cfg = LFCCConfig()
    model_cfg = ModelConfig()
    forensic_extractor = ForensicSpectrogramExtractor(audio_cfg, lfcc_cfg)
    
    # Fetching dataset files
    print("[*] Fetching file list from garystafford/deepfake-audio-detection...")
    api = HfApi()
    repo_id = "garystafford/deepfake-audio-detection"
    files = api.list_repo_files(repo_id, repo_type="dataset")
    
    real_files = [f for f in files if f.startswith("real/") and f.endswith(".flac")]
    fake_files = [f for f in files if f.startswith("fake/") and f.endswith(".flac")]
    
    # We want top models (el_=ElevenLabs, po_=Polly, hu_=Hume)
    top_fake_files = [f for f in fake_files if "el_" in f or "hu_" in f or "po_" in f]
    
    # Take a subset to train quickly (10-15 mins)
    subset_size = 40
    real_subset = real_files[:subset_size]
    fake_subset = top_fake_files[:subset_size]
    
    train_samples = []
    train_labels = []
    
    print(f"[*] Downloading & processing {subset_size} Real and {subset_size} Top AI Deepfake samples...")
    for f in tqdm(real_subset + fake_subset):
        label = 0 if f in real_subset else 1
        local_path = hf_hub_download(repo_id=repo_id, filename=f, repo_type="dataset")
        
        # Load and resample to 16000 Hz
        y, sr = librosa.load(local_path, sr=audio_cfg.sample_rate)
        
        # Pad or truncate to fixed length (e.g., 40000 samples for ~2.5 seconds)
        target_len = 40000
        if len(y) > target_len:
            y = y[:target_len]
        else:
            y = np.pad(y, (0, target_len - len(y)))
            
        # Data Augmentation: Create 4 variations per sample for robust features
        for aug in range(4):
            aug_y = y.copy()
            if aug > 0:
                noise = np.random.normal(0, 0.005 * aug, size=len(aug_y)).astype(np.float32)
                shift = (aug * 1000) % len(aug_y)
                aug_y = np.roll(aug_y, shift) + noise
                
            spec = forensic_extractor(aug_y)
            train_samples.append(spec)
            train_labels.append(label)

    train_x = torch.stack(train_samples, dim=0).to(device)
    train_y = torch.tensor(train_labels, dtype=torch.long).to(device)
    
    print(f"[*] Generated {len(train_x)} spectrogram tensors for training.")
    
    # Initialize Model
    model = PhysioSpecNet(model_cfg).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Training Loop
    print("\n[*] Starting Fast Epoch Training Loop (10 Epochs)...")
    batch_size = 8
    num_batches = int(np.ceil(len(train_x) / batch_size))
    
    best_loss = float('inf')
    best_eer = 1.0
    
    for epoch in range(1, 11):
        t0 = time.time()
        model.train()
        epoch_loss = 0.0
        
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
            optimizer.step()
            
            epoch_loss += loss.item() * x_batch.size(0)
            
        epoch_loss /= len(train_x)
        
        # Eval
        model.eval()
        with torch.no_grad():
            probs = model.predict_proba(train_x)[:, 1].cpu().numpy()
            
        eer, opt_thresh = compute_eer_from_probabilities(train_labels, probs)
        elapsed = time.time() - t0
        
        if epoch_loss < best_loss or eer < best_eer:
            best_loss = epoch_loss
            best_eer = eer
            
            ckpt_path = os.path.join(CHECKPOINT_DIR, "best_physiospecnet.pth")
            torch.save({
                'model_state_dict': model.state_dict(),
                'eer': best_eer,
                'threshold': opt_thresh
            }, ckpt_path)
            
        print(f"  Epoch {epoch:02d}/10 | Train Loss: {epoch_loss:.4f} | EER: {eer*100:.2f}% | Time: {elapsed:.2f}s")

    print("\n[*] Finished training. Checkpoint saved as best_physiospecnet.pth")

if __name__ == '__main__':
    train_on_top_ai_models()

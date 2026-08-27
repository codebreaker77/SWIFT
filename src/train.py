"""
Training loop for EfficientNet-B0 Baseline on ASVspoof LA dataset.
Monitors validation Equal Error Rate (EER) and saves best model checkpoint.
"""

import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from src.config import AudioConfig, LFCCConfig, ModelConfig, TrainingConfig, CHECKPOINT_DIR
from src.dataset import StreamingAudioDataset
from torch.utils.data import DataLoader
from src.models.efficientnet_b0 import EfficientNetB0Baseline
from src.metrics import compute_eer_from_probabilities

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_samples = 0
    
    for batch_idx, (spec, labels) in enumerate(dataloader):
        spec, labels = spec.to(device), labels.to(device)
        optimizer.zero_grad()
        
        logits = model(spec)
        loss = criterion(logits, labels)
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * spec.size(0)
        total_samples += spec.size(0)
        
    return total_loss / max(1, total_samples)

@torch.no_grad()
def evaluate_eer(model, dataloader, device):
    model.eval()
    all_labels = []
    all_p_fake = []
    
    for spec, labels in dataloader:
        spec = spec.to(device)
        probs = model.predict_proba(spec)  # (B, 2) [P(real), P(fake)]
        p_fake = probs[:, 1].cpu().numpy()
        
        all_labels.extend(labels.numpy())
        all_p_fake.extend(p_fake)
        
    all_labels = np.array(all_labels)
    all_p_fake = np.array(all_p_fake)
    
    eer, threshold = compute_eer_from_probabilities(all_labels, all_p_fake)
    return eer, threshold

def run_training(
    epochs: int = 10,
    train_samples: int = 320,
    dev_samples: int = 160,
    save_name: str = "best_baseline_model.pth"
):
    print("=" * 60)
    print("Starting EfficientNet-B0 Baseline Training Pipeline")
    print("=" * 60)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    model_cfg = ModelConfig()
    train_cfg = TrainingConfig()
    
    model = EfficientNetB0Baseline(model_cfg, pretrained=True).to(device)
    
    # BCE Loss with Label Smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg.label_smoothing)
    
    # AdamW Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)
    
    # ReduceLROnPlateau Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=train_cfg.lr_plateau_factor,
        patience=train_cfg.lr_plateau_patience
    )
    
    # Datasets and Loaders
    train_dataset = StreamingAudioDataset(split="train", is_train=True, num_samples=train_samples)
    dev_dataset = StreamingAudioDataset(split="dev", is_train=False, num_samples=dev_samples)
    
    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size)
    dev_loader = DataLoader(dev_dataset, batch_size=train_cfg.batch_size)
    
    best_eer = float('inf')
    patience_counter = 0
    best_model_path = os.path.join(CHECKPOINT_DIR, save_name)
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        dev_eer, dev_threshold = evaluate_eer(model, dev_loader, device)
        
        scheduler.step(dev_eer)
        elapsed = time.time() - start_time
        
        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Dev EER: {dev_eer * 100:.2f}% | Thresh: {dev_threshold:.4f} | Time: {elapsed:.2f}s")
        
        if dev_eer < best_eer:
            best_eer = dev_eer
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_eer': best_eer,
                'threshold': dev_threshold
            }, best_model_path)
            print(f" -> Best model saved to {best_model_path} (Dev EER: {best_eer * 100:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= train_cfg.early_stopping_patience:
                print(f"Early stopping triggered at epoch {epoch} (No dev EER improvement for {patience_counter} epochs).")
                break
                
    print(f"\nTraining Complete. Best Validation Dev EER: {best_eer * 100:.2f}%")
    return best_model_path

if __name__ == '__main__':
    run_training(epochs=5, train_samples=160, dev_samples=80)

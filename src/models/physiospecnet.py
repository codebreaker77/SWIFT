"""
PhysioSpecNet Architecture with Cross-Channel Attention for 6-Channel Forensic Spectrograms.
Physiological channels (Formants, GNE, Jitter/Shimmer, Phase, HNR) modulate LFCC spectral features via Cross-Attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig
from src.models.efficientnet_b0 import EfficientNetB0Backbone

class CrossChannelAttention(nn.Module):
    """
    Cross-Channel Attention Module.
    Allows physiological feature channels (Formants, GNE, Jitter/Shimmer, Phase, HNR)
    to dynamically attend to and modulate the LFCC spectral features.
    """
    def __init__(self, num_channels: int = 6, feature_dim: int = 32):
        super().__init__()
        self.num_channels = num_channels
        self.query_conv = nn.Conv2d(1, feature_dim, kernel_size=1)             # LFCC Query
        self.key_conv = nn.Conv2d(num_channels - 1, feature_dim, kernel_size=1) # Physio Channels Key
        self.value_conv = nn.Conv2d(num_channels - 1, feature_dim, kernel_size=1)# Physio Channels Value
        
        self.channel_weights = nn.Parameter(torch.ones(num_channels))
        self.out_conv = nn.Conv2d(feature_dim, num_channels, kernel_size=1)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (B, 6, H, W) e.g., (B, 6, 224, 224)
        Returns attended 6-channel tensor of shape: (B, 6, H, W)
        """
        B, C, H, W = x.shape
        lfcc_channel = x[:, 0:1, :, :]         # (B, 1, H, W)
        physio_channels = x[:, 1:, :, :]       # (B, 5, H, W)
        
        # Spatial pooling to 14x14 grid for memory-efficient cross-attention computation
        lfcc_pooled = F.adaptive_avg_pool2d(lfcc_channel, (14, 14))     # (B, 1, 14, 14)
        physio_pooled = F.adaptive_avg_pool2d(physio_channels, (14, 14)) # (B, 5, 14, 14)
        
        # Project Query, Key, Value
        Q = self.query_conv(lfcc_pooled).view(B, -1, 196).permute(0, 2, 1) # (B, 196, D)
        K = self.key_conv(physio_pooled).view(B, -1, 196)                  # (B, D, 196)
        V = self.value_conv(physio_pooled).view(B, -1, 196).permute(0, 2, 1)# (B, 196, D)
        
        # Attention scores: (B, 196, 196)
        attn_scores = torch.bmm(Q, K) / (Q.shape[-1] ** 0.5)
        attn_weights = self.softmax(attn_scores)
        
        # Attended representation
        attended = torch.bmm(attn_weights, V).permute(0, 2, 1).view(B, -1, 14, 14) # (B, D, 14, 14)
        
        # Upsample back to (H, W)
        attended_upsampled = F.interpolate(attended, size=(H, W), mode='bilinear', align_corners=False)
        
        # Modulate original 6 channels with cross-attention residual
        channel_delta = self.out_conv(attended_upsampled)                          # (B, 6, H, W)
        weighted_x = x * self.channel_weights.view(1, C, 1, 1)
        
        out = weighted_x + 0.2 * channel_delta
        return out

class PhysioSpecNet(nn.Module):
    """
    PhysioSpecNet model combining 6-channel forensic spectrograms,
    Cross-Channel Attention, and EfficientNet feature backbone.
    """
    def __init__(self, config: ModelConfig = ModelConfig(in_channels=6)):
        super().__init__()
        self.config = config
        
        # Cross-Channel Attention Module
        self.cross_attn = CrossChannelAttention(num_channels=6)
        
        # Pretrained Feature Extractor Backbone (6 channels -> 1280)
        # Force fallback to EfficientNetB0Backbone to match the keys saved in best_physiospecnet.pth
        self.backbone = EfficientNetB0Backbone(in_channels=6)
        
        # Global Average Pooling & Head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(1280, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input x shape: (B, 6, 224, 224)
        Output logits shape: (B, 2)
        """
        # Cross-Channel Attention Modulation
        attn_x = self.cross_attn(x)
        
        # Feature extraction
        feats = self.backbone(attn_x)
        pooled = self.gap(feats)
        flattened = torch.flatten(pooled, 1)
        logits = self.head(flattened)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

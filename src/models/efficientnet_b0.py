"""
EfficientNet-B0 adaptation for 1-channel LFCC Audio Deepfake Detection.
Matches exact head replacement & 1-channel averaged weight initialization specs.
Includes fallback clean MBConv backbone if torchvision weights are unavailable.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.config import ModelConfig

def _make_divisible(v: float, divisor: int = 8, min_value: int = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v

class ConvBNAct(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1, norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            norm_layer(out_planes),
            act_layer()
        )

class SqueezeExcitation(nn.Module):
    def __init__(self, input_channels, squeeze_channels):
        super().__init__()
        self.fc1 = nn.Conv2d(input_channels, squeeze_channels, 1)
        self.fc2 = nn.Conv2d(squeeze_channels, input_channels, 1)
        self.silu = nn.SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        scale = F.adaptive_avg_pool2d(input, 1)
        scale = self.fc1(scale)
        scale = self.silu(scale)
        scale = self.fc2(scale)
        scale = self.sigmoid(scale)
        return input * scale

class MBConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, expand_ratio, kernel_size, stride, se_ratio=0.25, drop_connect_rate=0.0):
        super().__init__()
        self.stride = stride
        self.use_res_connect = self.stride == 1 and in_planes == out_planes
        self.drop_connect_rate = drop_connect_rate
        hidden_dim = int(round(in_planes * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNAct(in_planes, hidden_dim, kernel_size=1))
        
        layers.append(ConvBNAct(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride, groups=hidden_dim))
        
        if se_ratio:
            sq_channels = max(1, int(in_planes * se_ratio))
            layers.append(SqueezeExcitation(hidden_dim, sq_channels))

        layers.append(nn.Conv2d(hidden_dim, out_planes, 1, bias=False))
        layers.append(nn.BatchNorm2d(out_planes))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.block(x)
        else:
            return self.block(x)

class EfficientNetB0Backbone(nn.Module):
    """
    Native PyTorch EfficientNet-B0 Backbone (5.3M params).
    """
    def __init__(self, in_channels=1):
        super().__init__()
        # EfficientNet-B0 architectural stages config:
        # [expand_ratio, out_channels, num_layers, stride, kernel_size]
        b0_configs = [
            [1, 16, 1, 1, 3],
            [6, 24, 2, 2, 3],
            [6, 40, 2, 2, 5],
            [6, 80, 3, 2, 3],
            [6, 112, 3, 1, 5],
            [6, 192, 4, 2, 5],
            [6, 320, 1, 1, 3],
        ]
        
        first_conv_out = 32
        self.stem = ConvBNAct(in_channels, first_conv_out, kernel_size=3, stride=2)
        
        in_c = first_conv_out
        blocks = []
        for t, c, n, s, k in b0_configs:
            out_c = c
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(MBConvBlock(in_c, out_c, expand_ratio=t, kernel_size=k, stride=stride))
                in_c = out_c
        self.blocks = nn.Sequential(*blocks)
        
        last_conv_out = 1280
        self.head_conv = ConvBNAct(in_c, last_conv_out, kernel_size=1)
        self.out_channels = last_conv_out

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head_conv(x)
        return x

class EfficientNetB0Baseline(nn.Module):
    """
    Adapted EfficientNet-B0 model for 1-channel 224x224 LFCC input.
    """
    def __init__(self, config: ModelConfig = ModelConfig(), pretrained: bool = True):
        super(EfficientNetB0Baseline, self).__init__()
        self.config = config
        
        try:
            import torchvision.models as models
            if hasattr(models, 'efficientnet_b0'):
                weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
                base_model = models.efficientnet_b0(weights=weights)
            else:
                base_model = models.efficientnet_b0(pretrained=pretrained)
                
            orig_conv = base_model.features[0][0]
            new_conv = nn.Conv2d(
                in_channels=config.in_channels,
                out_channels=orig_conv.out_channels,
                kernel_size=orig_conv.kernel_size,
                stride=orig_conv.stride,
                padding=orig_conv.padding,
                bias=orig_conv.bias is not None
            )
            with torch.no_grad():
                new_conv.weight.copy_(orig_conv.weight.mean(dim=1, keepdim=True))
                if orig_conv.bias is not None:
                    new_conv.bias.copy_(orig_conv.bias)
            base_model.features[0][0] = new_conv
            self.features = base_model.features
            in_features = 1280
        except Exception as e:
            # Fallback to native clean EfficientNet-B0 implementation
            self.features = EfficientNetB0Backbone(in_channels=config.in_channels)
            in_features = 1280
        
        # Head replacement: Global Average Pooling -> Dropout(0.3) -> Dense(1280, 256) -> ReLU -> Dropout(0.3) -> Dense(256, 2)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.custom_head = nn.Sequential(
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(in_features, config.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=config.dropout_rate),
            nn.Linear(config.hidden_dim, config.num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Input x shape: (B, 1, 224, 224)
        Output logits shape: (B, 2) [real_logit, fake_logit]
        """
        feats = self.features(x)
        pooled = self.gap(feats)
        flattened = torch.flatten(pooled, 1)
        logits = self.custom_head(flattened)
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns normalized probabilities [P(real), P(fake)] using Softmax.
        Shape: (B, 2)
        """
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

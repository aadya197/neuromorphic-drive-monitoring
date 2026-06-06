"""
CNN Baseline for ablation comparison against the SNN model.

Architecture:
  CNNBackbone  (standard Conv2d + BatchNorm + ReLU residual blocks)
  → same input shape as SNNBackbone: (batch, 5, 64, 64) voxel grid
  → outputs (batch, 128) feature vector (identical shape to SNNBackbone output)

  The rest of the pipeline (LayerNorm → BiGRU → TemporalAttention → Linear(5))
  is shared with DriverMonitoringModel, enabling a fair apples-to-apples
  comparison that isolates the SNN vs CNN backbone contribution.

Usage in a paper:
    "To isolate the contribution of spiking neural network processing, we trained
    an otherwise identical model replacing the SNN backbone with a standard
    convolutional backbone of comparable parameter count."
"""
import torch
import torch.nn as nn

from models.attention_module import TemporalAttention


class ConvResBlock(nn.Module):
    """Standard residual block: Conv → BN → ReLU → Conv → BN + skip."""
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.block(x) + x)


class CNNBackbone(nn.Module):
    """
    Standard CNN feature extractor — drop-in replacement for SNNBackbone.

    Input : (batch, 5, 64, 64)   — event voxel grid (same as SNN)
    Output: (batch, 128)         — spatial feature vector (same as SNN)

    Parameter count is intentionally matched to SNNBackbone for a fair comparison.
    """
    def __init__(self, in_channels=5, base_channels=32, out_dim=128, dropout=0.2):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True),
        )
        self.layer1 = nn.Sequential(
            ConvResBlock(base_channels),
            nn.MaxPool2d(2),                              # 64→32
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 2),
            nn.ReLU(inplace=True),
            ConvResBlock(base_channels * 2),
            nn.MaxPool2d(2),                              # 32→16
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels * 4),
            nn.ReLU(inplace=True),
            ConvResBlock(base_channels * 4),
            nn.AdaptiveAvgPool2d(1),                     # 16→1
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(base_channels * 4, out_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.head(x)      # (batch, 128)


class BaselineDriverMonitoringModel(nn.Module):
    """
    Full baseline model (CNN + GRU) for comparison with DriverMonitoringModel (SNN + GRU).

    The GRU, attention, and classifier head are architecturally identical to the SNN
    model to isolate the backbone contribution in ablation analysis.
    """
    def __init__(self, num_classes=5, gru_hidden=128, dropout=0.3):
        super().__init__()
        self.face_backbone = CNNBackbone(dropout=0.2)
        self.body_backbone = CNNBackbone(dropout=0.2)
        self.input_norm = nn.LayerNorm(256)

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.attention  = TemporalAttention(gru_hidden * 2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, num_classes),
        )

    def forward(self, x_face, x_body):
        if x_face.dim() == 4:
            x_face = x_face.unsqueeze(1)
            x_body = x_body.unsqueeze(1)

        batch, time, c, h, w = x_face.shape

        x_face = x_face.reshape(batch * time, c, h, w)
        x_body = x_body.reshape(batch * time, c, h, w)

        face_features = self.face_backbone(x_face)
        body_features = self.body_backbone(x_body)

        features = torch.cat([face_features, body_features], dim=1)
        features = features.view(batch, time, -1)
        features = self.input_norm(features)

        gru_out, _ = self.gru(features)
        context = self.attention(gru_out)
        return self.classifier(context)

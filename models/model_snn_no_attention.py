"""
Ablation variant: Dual-Stream SNN + BiGRU, WITHOUT Temporal Attention.

Used to answer: "Does the Temporal Attention module contribute?"

Difference from full DriverMonitoringModel:
  - TemporalAttention is replaced by simple mean pooling over GRU outputs.
  - All other components (SNNBackbone x2, LayerNorm, BiGRU, classifier) are identical.
"""
import torch
import torch.nn as nn
from models.snn_backbone import SNNBackbone


class SNNBiGRUNoAttention(nn.Module):
    """
    SNN + BiGRU without temporal attention.
    Mean-pools across the GRU time dimension instead of using learned attention weights.
    """
    def __init__(self, num_classes=5, gru_hidden=128, dropout=0.3):
        super().__init__()
        self.face_backbone = SNNBackbone(dropout=0.2)
        self.body_backbone = SNNBackbone(dropout=0.2)
        self.input_norm    = nn.LayerNorm(256)

        self.gru = nn.GRU(
            input_size=256,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        # No TemporalAttention — use mean pooling instead
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

        gru_out, _ = self.gru(features)            # (batch, time, 256)

        # Mean pooling instead of attention (ablation)
        context = gru_out.mean(dim=1)              # (batch, 256)

        return self.classifier(context)

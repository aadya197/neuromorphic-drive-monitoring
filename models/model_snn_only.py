"""
Ablation variant: Dual-Stream SNN ONLY — no BiGRU, no Temporal Attention.

Used to answer: "Does temporal modelling (BiGRU) contribute beyond static SNN features?"

Difference from full DriverMonitoringModel:
  - No GRU, no attention.
  - SNN backbone features from all 5 time-steps are mean-pooled directly.
  - Classifier operates on the pooled 256-dim feature vector.
  - Everything else (dual-stream, SNNBackbone, classifier head shape) is identical.
"""
import torch
import torch.nn as nn
from models.snn_backbone import SNNBackbone


class SNNOnlyNoGRU(nn.Module):
    """
    Pure SNN spatial feature extractor — no temporal modelling.
    Processes each of the 5 voxel time-steps independently through the SNN,
    then averages the resulting features before classification.
    """
    def __init__(self, num_classes=5, dropout=0.3):
        super().__init__()
        self.face_backbone = SNNBackbone(dropout=0.2)
        self.body_backbone = SNNBackbone(dropout=0.2)
        self.input_norm    = nn.LayerNorm(256)

        # No GRU — direct classifier from 256-dim pooled features
        self.classifier = nn.Sequential(
            nn.LayerNorm(256),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x_face, x_body):
        if x_face.dim() == 4:
            x_face = x_face.unsqueeze(1)
            x_body = x_body.unsqueeze(1)

        batch, time, c, h, w = x_face.shape
        x_face = x_face.reshape(batch * time, c, h, w)
        x_body = x_body.reshape(batch * time, c, h, w)

        face_features = self.face_backbone(x_face)             # (batch*time, 128)
        body_features = self.body_backbone(x_body)             # (batch*time, 128)

        features = torch.cat([face_features, body_features], dim=1)  # (batch*time, 256)
        features = features.view(batch, time, -1)                      # (batch, time, 256)
        features = self.input_norm(features)

        # Mean pool across time — no GRU (ablation)
        context = features.mean(dim=1)                                 # (batch, 256)

        return self.classifier(context)

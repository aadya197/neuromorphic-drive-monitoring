"""
Single-Stream SNN model — face camera only.

Used in the ablation study to quantify the contribution of the body camera.
Architecturally identical to DriverMonitoringModel, except:
  - Only one SNNBackbone (face) instead of two.
  - GRU input size is 128 (face only) instead of 256 (face + body).
  - All other components (LayerNorm, BiGRU, TemporalAttention, classifier) are unchanged.

This isolates the question: "Does the body camera stream improve detection?"
If dual-stream (256-dim) outperforms single-stream (128-dim), the body camera
contributes meaningful spatial context beyond the face stream alone.
"""
import torch
import torch.nn as nn

from models.snn_backbone import SNNBackbone
from models.attention_module import TemporalAttention


class SingleStreamSNN(nn.Module):
    """
    Face-only SNN driver monitoring model.

    Input : x_face  (batch, time, 5, 64, 64)  — face voxel stream only
            x_body is accepted but IGNORED to keep the same call signature
            as DriverMonitoringModel (simplifies comparison code).
    Output: (batch, num_classes)  logits
    """

    def __init__(self, num_classes=5, gru_hidden=128, dropout=0.3):
        super().__init__()

        self.face_backbone = SNNBackbone(dropout=0.2)
        # Input to GRU is 128 (face only) — half of the dual-stream model
        self.input_norm = nn.LayerNorm(128)

        self.gru = nn.GRU(
            input_size=128,
            hidden_size=gru_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        self.attention = TemporalAttention(gru_hidden * 2)

        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_hidden * 2),
            nn.Dropout(dropout),
            nn.Linear(gru_hidden * 2, num_classes),
        )

    def forward(self, x_face, x_body=None):   # x_body ignored
        if x_face.dim() == 4:
            x_face = x_face.unsqueeze(1)

        batch, time, c, h, w = x_face.shape

        x_face = x_face.reshape(batch * time, c, h, w)
        face_features = self.face_backbone(x_face)          # (batch*time, 128)

        features = face_features.view(batch, time, -1)      # (batch, time, 128)
        features = self.input_norm(features)

        gru_out, _ = self.gru(features)                     # (batch, time, 256)
        context = self.attention(gru_out)                   # (batch, 256)

        return self.classifier(context)

import torch
import torch.nn as nn

from models.snn_backbone import SNNBackbone
from models.attention_module import TemporalAttention


class DriverMonitoringModel(nn.Module):
    """
    Full driver monitoring model:
      SNNBackbone  (per-frame, single-step)  → (batch*time, 128)
      LayerNorm → Bidirectional GRU          → (batch, time, gru_hidden*2)
      Temporal Attention                     → (batch, gru_hidden*2)
      Classifier head                        → (batch, num_classes)
    """

    def __init__(self, num_classes=2, gru_hidden=128, dropout=0.3):
        super().__init__()

        self.face_backbone = SNNBackbone(dropout=0.2)
        self.body_backbone = SNNBackbone(dropout=0.2)
        self.input_norm = nn.LayerNorm(256)

        # Bidirectional GRU: 256 context dims (since 128+128=256)
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
            nn.Linear(gru_hidden * 2, 5), # [Drowsy, Distracted, Glasses, Sunglasses, Night]
        )

    def forward(self, x_face, x_body):
        # inputs: (batch, time, C, H, W)
        if x_face.dim() == 4:
            x_face = x_face.unsqueeze(1)
            x_body = x_body.unsqueeze(1)

        batch, time, c, h, w = x_face.shape

        # Process each time-step independently through the SNN backbones
        x_face = x_face.reshape(batch * time, c, h, w)
        x_body = x_body.reshape(batch * time, c, h, w)
        
        face_features = self.face_backbone(x_face)               # (batch*time, 128)
        body_features = self.body_backbone(x_body)               # (batch*time, 128)

        # Concatenate features
        features = torch.cat([face_features, body_features], dim=1) # (batch*time, 256)

        # Restore temporal axis: (batch, time, 256)
        features = features.view(batch, time, -1)
        features = self.input_norm(features)

        # Bidirectional GRU temporal modelling: (batch, time, 256)
        gru_out, _ = self.gru(features)

        # Attention-weighted pooling: (batch, 256)
        context = self.attention(gru_out)

        return self.classifier(context)
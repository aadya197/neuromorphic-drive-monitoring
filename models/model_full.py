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

        self.backbone   = SNNBackbone(dropout=0.2)
        self.input_norm = nn.LayerNorm(128)

        # Bidirectional GRU: 128 context dims (since gru_hidden=128)
        self.gru = nn.GRU(
            input_size=128,
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

    def forward(self, x):
        # x: (batch, time, C, H, W)  or  (batch, C, H, W)
        if x.dim() == 4:
            x = x.unsqueeze(1)

        batch, time, c, h, w = x.shape

        # Process each time-step independently through the SNN backbone
        # reshape: (batch*time, C, H, W)
        x = x.reshape(batch * time, c, h, w)
        features = self.backbone(x)               # (batch*time, 128)

        # Restore temporal axis: (batch, time, 128)
        features = features.view(batch, time, -1)
        features = self.input_norm(features)

        # Bidirectional GRU temporal modelling: (batch, time, 256)
        gru_out, _ = self.gru(features)

        # Attention-weighted pooling: (batch, 256)
        context = self.attention(gru_out)

        return self.classifier(context)
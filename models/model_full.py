import torch
import torch.nn as nn

from models.snn_backbone import SNNBackbone
from models.attention_module import TemporalAttention


class DriverMonitoringModel(nn.Module):

    def __init__(self, num_classes=2):

        super().__init__()

        self.backbone = SNNBackbone()

        # backbone output flattened size
        self.lstm = nn.LSTM(
            input_size=8192,
            hidden_size=256,
            batch_first=True
        )

        self.attention = TemporalAttention(256)

        self.fc = nn.Linear(256, num_classes)

        self.dropout = nn.Dropout(0.4)

    def forward(self, x):

        # add time dimension if missing
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        batch, time, c, h, w = x.shape

        x = x.view(batch * time, c, h, w)

        features = self.backbone(x)

        features = features.view(batch, time, -1)

        lstm_out, _ = self.lstm(features)

        attn_out = self.attention(lstm_out)
        attn_out = self.dropout(attn_out)

        output = self.fc(attn_out)

        return output
# This helps the model focus on hands, phone, face, steering wheel.

import torch
import torch.nn as nn

class TemporalAttention(nn.Module):

    def __init__(self, hidden_dim):

        super().__init__()

        self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x):

        weights = torch.softmax(self.attn(x), dim=1)

        output = torch.sum(weights * x, dim=1)

        return output
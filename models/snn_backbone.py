import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer


class SpikingResBlock(nn.Module):
    """
    Spiking residual block (single-step mode).
    Uses ParametricLIFNode so each neuron learns its own time constant.
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.lif1  = neuron.ParametricLIFNode(detach_reset=True)
        self.conv2 = layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.lif2  = neuron.ParametricLIFNode(detach_reset=True)

    def forward(self, x):
        out = self.lif1(self.bn1(self.conv1(x)))
        out = self.lif2(self.bn2(self.conv2(out)))
        return out + x      # residual skip


class SNNBackbone(nn.Module):
    """
    Single-step SNN feature extractor.
    Input : (batch, 5, H, W)   — one voxel frame at a time
    Output: (batch, 128)       — compact feature vector
    """

    def __init__(self, dropout=0.3):
        super().__init__()

        # Stage 1 — 5 → 32 channels
        self.conv1 = layer.Conv2d(5, 32, kernel_size=3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(32)
        self.lif1  = neuron.ParametricLIFNode(detach_reset=True)
        self.pool1 = nn.MaxPool2d(2)

        # Stage 2 — spiking residual block + pool
        self.res1  = SpikingResBlock(32)
        self.pool2 = nn.MaxPool2d(2)

        # Stage 3 — 32 → 64 channels
        self.conv2 = layer.Conv2d(32, 64, kernel_size=3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(64)
        self.lif2  = neuron.ParametricLIFNode(detach_reset=True)
        self.pool3 = nn.MaxPool2d(2)

        # Stage 4 — spiking residual block + pool
        self.res2  = SpikingResBlock(64)
        self.pool4 = nn.MaxPool2d(2)

        # Stage 5 — 64 → 128 channels
        self.conv3 = layer.Conv2d(64, 128, kernel_size=3, padding=1, bias=False)
        self.bn3   = nn.BatchNorm2d(128)
        self.lif3  = neuron.ParametricLIFNode(detach_reset=True)
        self.res3  = SpikingResBlock(128)
        self.dropout = nn.Dropout2d(dropout)

        # Global average pool → fixed 128-dim output
        self.gap = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        x = self.pool1(self.lif1(self.bn1(self.conv1(x))))
        x = self.pool2(self.res1(x))
        x = self.pool3(self.lif2(self.bn2(self.conv2(x))))
        x = self.pool4(self.res2(x))
        x = self.dropout(x)
        x = self.lif3(self.bn3(self.conv3(x)))
        x = self.res3(x)
        x = self.gap(x).flatten(1)   # (batch, 128)
        return x
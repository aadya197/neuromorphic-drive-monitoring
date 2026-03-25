import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer


class SNNBackbone(nn.Module):

    def __init__(self):
        super().__init__()

        # ✅ input channels = 5 (voxel bins)
        self.conv1 = layer.Conv2d(5, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.lif1 = neuron.LIFNode()

        self.conv2 = layer.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.lif2 = neuron.LIFNode()

        self.pool = nn.MaxPool2d(2)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lif1(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lif2(x)
        x = self.pool(x)

        return x
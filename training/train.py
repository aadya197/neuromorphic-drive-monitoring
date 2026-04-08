import torch
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional

from models.model_full import DriverMonitoringModel

# from dataset.event_dataset import EventDataset
# dataset = EventDataset("data/train.npy")

from dataset.video_dataset import VideoDataset
import os
from models.model_full import *
from dataset.event_dataset import EventDataset
from utils.visualization import *

data_path = os.path.expanduser("~/Desktop/dmd")

dataset = VideoDataset(data_path)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = DriverMonitoringModel()

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

loss_fn = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.3, 0.7]))


for epoch in range(20):

    correct = 0
    total = 0

    for x, y in loader:

        pred = model(x)

        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        functional.reset_net(model)

        _, predicted = torch.max(pred, 1)

        total += y.size(0)
        correct += int(torch.sum(predicted == y).item())

    accuracy = 100 * correct / total
    print(f"Epoch {epoch} Loss: {loss.item():.4f} Accuracy: {accuracy:.2f}%")

    if epoch == 9:
        torch.save(model.state_dict(), "model.pth")
        print("Model saved")
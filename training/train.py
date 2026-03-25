import torch
from torch.utils.data import DataLoader

from spikingjelly.activation_based import functional

from models.model_full import DriverMonitoringModel

# from dataset.event_dataset import EventDataset
# dataset = EventDataset("data/train.npy")

from dataset.video_dataset import VideoDataset
import os

data_path = os.path.expanduser("~/Desktop/dmd")

dataset = VideoDataset(data_path)

loader = DataLoader(dataset, batch_size=4, shuffle=True)

model = DriverMonitoringModel()

print(model)

optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

loss_fn = torch.nn.CrossEntropyLoss()


for epoch in range(10):

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

    acc = 100 * correct / total

    print("Epoch:", epoch, "Loss:", loss.item(), "Accuracy:", acc)

    if epoch == 9:
        torch.save(model.state_dict(), "model.pth")
        print("Model saved")
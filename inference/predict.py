import torch
from models.model_full import DriverMonitoringModel
from dataset.video_dataset import VideoDataset

model = DriverMonitoringModel()
model.load_state_dict(torch.load("model.pth"))
model.eval()

dataset = VideoDataset("/Users/aadya/Desktop/dmd")

x, y = dataset[0]
x = x.unsqueeze(0)

with torch.no_grad():
    pred = model(x)
    pred_class = torch.argmax(pred, dim=1).item()

print("Predicted:", "DROWSY 😴" if pred_class == 1 else "NORMAL 🙂")
print("Actual:", y)
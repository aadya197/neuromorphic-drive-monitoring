import torch
from models.model_full import DriverMonitoringModel

model = DriverMonitoringModel()
model.load_state_dict(torch.load("model.pth"))

model.eval()

def predict(x):

    with torch.no_grad():

        output = model(x)

        pred = torch.argmax(output, dim=1)

        return pred
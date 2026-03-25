import torch
from torch.utils.data import Dataset
import numpy as np

from dataset.preprocessing import events_to_voxel

class EventDataset(Dataset):

    def __init__(self, data_path):

        self.samples = np.load(data_path, allow_pickle=True)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        events = self.samples[idx]["events"]
        label = self.samples[idx]["label"]

        voxel = events_to_voxel(events)

        return voxel, label
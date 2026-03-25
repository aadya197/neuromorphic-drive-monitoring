import os
import torch
from torch.utils.data import Dataset

from dataset.event_from_video import video_to_events
from dataset.preprocessing import events_to_voxel


class VideoDataset(Dataset):

    def __init__(self, folder_path):

        self.files = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".mp4") or file.endswith(".avi"):
                    self.files.append(os.path.join(root, file))

    def __len__(self):
        return len(self.files)

    def get_label_from_filename(self, path):

        name = path.lower()

        if "face" in name:
            return 0
        elif "hands" in name:
            return 1
        elif "body" in name:
            return 2
        elif "drowsiness" in name:
            return 3
        else:
            return 4

    def __getitem__(self, idx):

        path = self.files[idx]

        import numpy as np

        file_name = os.path.basename(path) + ".npy"
        event_path = os.path.join("data/events", file_name)

        events = np.load(event_path)

        if len(events) < 50:
            return torch.zeros((3, 5, 128, 128)), 0

        num_chunks = 3
        chunk_size = len(events) // num_chunks

        voxels = []

        for i in range(num_chunks):

            chunk = events[i * chunk_size:(i + 1) * chunk_size]

            voxel = events_to_voxel(chunk)

            voxel = voxel / voxel.max() if voxel.max() != 0 else voxel

            voxels.append(voxel)

        voxels = torch.stack(voxels)

        label = self.get_label_from_filename(path)

        return voxels, label
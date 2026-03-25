import numpy as np
import torch

def events_to_voxel(events, num_bins=5, height=128, width=128):

    voxel = np.zeros((num_bins, height, width))

    if len(events) == 0:
        return torch.tensor(voxel, dtype=torch.float32)

    timestamps = events[:, 2]
    t_min = timestamps.min()
    t_max = timestamps.max()

    delta_t = t_max - t_min if (t_max - t_min) != 0 else 1

    # normalize coordinates
    x_max = events[:, 0].max()
    y_max = events[:, 1].max()

    for x, y, t, p in events:

        # scale x, y into 0–127
        x_scaled = int((x / x_max) * (width - 1)) if x_max != 0 else 0
        y_scaled = int((y / y_max) * (height - 1)) if y_max != 0 else 0

        bin_index = int((t - t_min) / delta_t * (num_bins - 1))

        voxel[bin_index, y_scaled, x_scaled] += p

    return torch.tensor(voxel, dtype=torch.float32)

# This converts asynchronous event camera data into a voxel grid representation for neural networks.
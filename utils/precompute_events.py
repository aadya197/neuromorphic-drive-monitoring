import os
import numpy as np
from dataset.event_from_video import video_to_events

input_folder = os.path.expanduser("~/Desktop/dmd")
output_folder = "data/events"

os.makedirs(output_folder, exist_ok=True)

for root, dirs, files in os.walk(input_folder):
    for file in files:
        if file.endswith(".mp4") or file.endswith(".avi"):

            path = os.path.join(root, file)

            print("Processing:", file)

            events = video_to_events(path)

            save_path = os.path.join(output_folder, file + ".npy")

            np.save(save_path, events)

print("Done!")
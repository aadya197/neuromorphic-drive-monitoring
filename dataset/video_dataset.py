import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.event_from_video import video_to_events
from dataset.preprocessing import events_to_voxel


def load_drowsiness_labels(json_path, total_frames):
    """
    Parse the DMD OpenLabel annotation JSON.
    Returns a boolean array of length total_frames:
        True  → frame is drowsy (eyes closed or yawning)
        False → frame is normal
    """
    drowsy_frames = np.zeros(total_frames, dtype=bool)

    if not os.path.exists(json_path):
        return drowsy_frames

    with open(json_path, "r") as f:
        data = json.load(f)

    actions = data.get("openlabel", {}).get("actions", {})

    drowsy_types = {
        "eyes_state/close",
        "eyes_state/closing",
        "yawning/Yawning with hand",
        "yawning/Yawning without hand",
    }

    for action in actions.values():
        if action.get("type") in drowsy_types:
            for interval in action.get("frame_intervals", []):
                start = interval["frame_start"]
                end   = min(interval["frame_end"] + 1, total_frames)
                drowsy_frames[start:end] = True

    return drowsy_frames


def get_label_from_annotation(json_path, total_frames):
    """
    Returns 1 (drowsy) if >20% of frames are drowsy, else 0 (normal).
    Lowered from 30% → 20% to catch sessions with subtle but real drowsiness.
    Falls back to 0 if no annotation found.
    """
    labels = load_drowsiness_labels(json_path, total_frames)
    if labels.sum() == 0:
        return 0
    ratio = labels.sum() / len(labels)
    return 1 if ratio > 0.20 else 0


def augment_voxels(voxels):
    """
    Advanced voxel augmentation to improve generalization:
    - Random horizontal flip
    - Random Gaussian noise
    - Random Spatial Cutout (simulating occlusion)
    """
    # Random horizontal flip (left-right mirror)
    if random.random() > 0.5:
        voxels = torch.flip(voxels, dims=[-1])

    # Random Gaussian noise
    if random.random() > 0.3:
        noise = torch.randn_like(voxels) * 0.05
        voxels = voxels + noise

    # Random Spatial Cutout
    if random.random() > 0.5:
        _, _, h, w = voxels.shape
        cut_h = random.randint(10, 20)
        cut_w = random.randint(10, 20)
        y = random.randint(0, h - cut_h)
        x = random.randint(0, w - cut_w)
        voxels[:, :, y:y+cut_h, x:x+cut_w] = 0.0

    # Clamp to valid range
    voxels = torch.clamp(voxels, -1.0, 1.0)

    return voxels


class VideoDataset(Dataset):

    def __init__(self, folder_path, augment=False):
        """
        Scans folder_path AND all dmd* folders in ~/Downloads for rgb_face videos.
        Deduplicates by filename so the same session isn't loaded twice.
        Pairs each video with its _ann_drowsiness.json annotation.

        Args:
            augment: if True, applies random voxel augmentation (use during training)
        """
        self.samples = []    # list of (video_path, label)
        self.augment  = augment

        # Build list of all roots to scan
        search_roots = [folder_path]
        downloads = os.path.expanduser("~/Downloads")
        for name in sorted(os.listdir(downloads)):
            candidate = os.path.join(downloads, name)
            if name.lower().startswith("dmd") and os.path.isdir(candidate):
                search_roots.append(candidate)

        face_videos = []
        seen_files  = set()   # deduplicate by filename

        for root_folder in search_roots:
            for root, dirs, files in os.walk(root_folder):
                for file in sorted(files):
                    if not (file.endswith(".mp4") or file.endswith(".avi")):
                        continue
                    if file in seen_files:
                        continue
                    if "rgb_face" not in file and "ir_face" not in file:
                        continue
                    seen_files.add(file)
                    face_videos.append(os.path.join(root, file))

        for video_path in face_videos:
            # Find paired annotation JSON in the same directory
            video_dir  = os.path.dirname(video_path)
            json_files = [
                f for f in os.listdir(video_dir)
                if f.endswith("_ann_drowsiness.json")
            ]

            if json_files:
                json_path = os.path.join(video_dir, json_files[0])
                import cv2
                cap          = cv2.VideoCapture(video_path)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
                label = get_label_from_annotation(json_path, max(total_frames, 1))
            else:
                label = 1 if "drowsiness" in video_path.lower() else 0

            self.samples.append((video_path, label))

        labels = [s[1] for s in self.samples]
        print(f"[VideoDataset] Loaded {len(self.samples)} videos | "
              f"Normal: {labels.count(0)} | Drowsy: {labels.count(1)} "
              f"| augment={augment}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        video_name = os.path.basename(video_path) + ".npy"
        event_path = os.path.join("data", "events", video_name)

        if os.path.exists(event_path):
            events = np.load(event_path)
        else:
            events = video_to_events(video_path)

        if len(events) < 50:
            return torch.zeros((5, 5, 64, 64)), label

        num_chunks = 5
        chunk_size = max(len(events) // num_chunks, 1)

        voxels = []
        for i in range(num_chunks):
            start_idx = i * chunk_size
            end_idx = (i + 1) * chunk_size
            
            # Temporal Augmentation: Random jitter of chunk boundaries
            if self.augment and chunk_size > 10:
                jitter = random.randint(-int(chunk_size * 0.15), int(chunk_size * 0.15))
                start_idx = max(0, start_idx + jitter)
                end_idx = min(len(events), end_idx + jitter)
                
            chunk = events[start_idx:end_idx]
            voxel = events_to_voxel(chunk)
            voxel = voxel / voxel.max() if voxel.max() != 0 else voxel
            voxels.append(voxel)

        voxels = torch.stack(voxels)   # (5, 5, H, W)
        voxels = torch.clamp(voxels, -1.0, 1.0)

        if self.augment:
            voxels = augment_voxels(voxels)

        return voxels, label
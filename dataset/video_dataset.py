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


def load_distraction_labels(json_path, total_frames):
    """
    Parse the DMD OpenLabel annotation JSON for distraction.
    Returns a boolean array of length total_frames.
    """
    distracted_frames = np.zeros(total_frames, dtype=bool)

    if not os.path.exists(json_path):
        return distracted_frames

    with open(json_path, "r") as f:
        data = json.load(f)

    actions = data.get("openlabel", {}).get("actions", {})

    # Any action that is a 'driver_action' but NOT 'safe_drive' is considered a distraction
    for action in actions.values():
        act_type = action.get("type", "")
        if act_type.startswith("driver_actions/") and act_type != "driver_actions/safe_drive":
            for interval in action.get("frame_intervals", []):
                start = interval["frame_start"]
                end   = min(interval["frame_end"] + 1, total_frames)
                distracted_frames[start:end] = True

    return distracted_frames


def get_label_from_annotation(drowsy_json_path, dist_json_path, total_frames, prefix=""):
    """
    Returns a multi-label tensor: [is_drowsy, is_distracted, is_glasses, is_sunglasses, is_night]
    Threshold: 20% of frames showing the behavior.
    """
    drowsy_frames = load_drowsiness_labels(drowsy_json_path, total_frames)
    dist_frames   = load_distraction_labels(dist_json_path, total_frames)

    is_drowsy = 1.0 if (drowsy_frames.sum() / max(len(drowsy_frames), 1)) > 0.20 else 0.0
    is_distracted = 1.0 if (dist_frames.sum() / max(len(dist_frames), 1)) > 0.20 else 0.0

    g_prefix = prefix.split('_')[0] if '_' in prefix else prefix
    is_glasses = 1.0 if g_prefix in ['gB', 'gE'] else 0.0
    is_sunglasses = 1.0 if g_prefix in ['gC', 'gF'] else 0.0
    is_night = 1.0 if g_prefix in ['gD', 'gE', 'gF'] else 0.0

    return torch.tensor([is_drowsy, is_distracted, is_glasses, is_sunglasses, is_night], dtype=torch.float32)


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

        session_dict = {}  # prefix -> { "face": path, "body": path }

        for root_folder in search_roots:
            for root, dirs, files in os.walk(root_folder):
                for file in sorted(files):
                    if not file.endswith(".mp4"):
                        continue
                    
                    if "rgb_face" in file or "ir_face" in file:
                        prefix = file.replace("_rgb_face.mp4", "").replace("_ir_face.mp4", "")
                        if prefix not in session_dict:
                            session_dict[prefix] = {}
                        session_dict[prefix]["face"] = os.path.join(root, file)
                        
                    elif "rgb_body" in file or "ir_body" in file:
                        prefix = file.replace("_rgb_body.mp4", "").replace("_ir_body.mp4", "")
                        if prefix not in session_dict:
                            session_dict[prefix] = {}
                        session_dict[prefix]["body"] = os.path.join(root, file)

        valid_sessions = []
        for prefix, paths in session_dict.items():
            if "face" in paths and "body" in paths:
                valid_sessions.append((paths["face"], paths["body"], prefix))

        for face_path, body_path, prefix in valid_sessions:
            # Find paired annotation JSON in the same directory as the face video
            video_dir  = os.path.dirname(face_path)
            json_files = [f for f in os.listdir(video_dir)]
            
            drowsy_json = [f for f in json_files if f.endswith("_ann_drowsiness.json")]
            dist_json   = [f for f in json_files if f.endswith("_ann_distraction.json")]

            drowsy_path = os.path.join(video_dir, drowsy_json[0]) if drowsy_json else ""
            dist_path   = os.path.join(video_dir, dist_json[0]) if dist_json else ""
            
            total_frames = 5000
            
            label = get_label_from_annotation(drowsy_path, dist_path, max(total_frames, 1), prefix)

            self.samples.append((face_path, body_path, label))

        labels = torch.stack([s[2] for s in self.samples])
        num_normal = ((labels[:, 0] == 0) & (labels[:, 1] == 0)).sum().item()
        num_drowsy = (labels[:, 0] == 1).sum().item()
        num_dist   = (labels[:, 1] == 1).sum().item()
        num_glasses = (labels[:, 2] == 1).sum().item()
        num_sunglasses = (labels[:, 3] == 1).sum().item()
        num_night = (labels[:, 4] == 1).sum().item()
        
        print(f"[VideoDataset] Loaded {len(self.samples)} dual-camera videos | "
              f"Normal: {num_normal} | Drowsy: {num_drowsy} | Distracted: {num_dist} | "
              f"Glasses: {num_glasses} | Sunglasses: {num_sunglasses} | Night: {num_night} "
              f"| augment={augment}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face_path, body_path, label = self.samples[idx]

        def process_video(video_path):
            video_name = os.path.basename(video_path) + ".npy"
            event_path = os.path.join("data", "events", video_name)
            
            if os.path.exists(event_path):
                events = np.load(event_path)
            else:
                events = video_to_events(video_path)
                os.makedirs(os.path.dirname(event_path), exist_ok=True)
                np.save(event_path, events)

            if len(events) < 50:
                return torch.zeros((5, 5, 64, 64))

            num_chunks = 5
            chunk_size = max(len(events) // num_chunks, 1)

            voxels = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size
                
                if self.augment and chunk_size > 10:
                    jitter = random.randint(-int(chunk_size * 0.15), int(chunk_size * 0.15))
                    start_idx = max(0, start_idx + jitter)
                    end_idx = min(len(events), end_idx + jitter)
                    
                chunk = events[start_idx:end_idx]
                voxel = events_to_voxel(chunk)
                voxel = voxel / voxel.max() if voxel.max() != 0 else voxel
                voxels.append(voxel)

            voxels = torch.stack(voxels)
            voxels = torch.clamp(voxels, -1.0, 1.0)
            if self.augment:
                voxels = augment_voxels(voxels)
            return voxels

        voxels_face = process_video(face_path)
        voxels_body = process_video(body_path)

        return (voxels_face, voxels_body), label
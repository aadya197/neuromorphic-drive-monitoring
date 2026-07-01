import os
import csv
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from dataset.event_from_video import video_to_events
from dataset.preprocessing import events_to_voxel
from dataset.video_dataset import augment_voxels

DISTRACTED_ACTIVITIES = {
    'drinking',
    'eating',
    'interacting_with_phone',
    'looking_or_moving_around (e.g. searching)',
    'reading_magazine',
    'reading_newspaper',
    'talking_on_phone',
    'using_multimedia_display',
    'working_on_laptop',
    'writing'
}

class DriveAndActDataset(Dataset):
    def __init__(self, downloads_path, augment=False, split=None, held_out_subjects=None):
        self.augment = augment
        self.split = split
        self.held_out_subjects = held_out_subjects or set()
        self.samples = []

        face_dir = os.path.join(downloads_path, "inner_mirror")
        body_dir = os.path.join(downloads_path, "a_column_driver")
        csv_path = os.path.join(downloads_path, "activities_3s", "inner_mirror", "midlevel.chunks_90.csv")

        if not os.path.exists(face_dir) or not os.path.exists(body_dir) or not os.path.exists(csv_path):
            print("[Drive&Act] Missing directories or CSV file. Skipping Drive&Act dataset.")
            return

        # Parse CSV
        # Format: participant_id,file_id,annotation_id,frame_start,frame_end,activity,chunk_id
        video_activities = {}
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                file_id = row['file_id']  # e.g. vp1/run1b_2018-05-29-14-02-47.ids_1
                if file_id not in video_activities:
                    video_activities[file_id] = []
                video_activities[file_id].append({
                    'start': int(row['frame_start']),
                    'end': int(row['frame_end']),
                    'activity': row['activity']
                })

        for file_id, actions in video_activities.items():
            face_path = os.path.join(face_dir, file_id + ".mp4")
            body_path = os.path.join(body_dir, file_id.replace(".ids_1", ".ids_5") + ".mp4")

            if not os.path.exists(face_path) or not os.path.exists(body_path):
                continue

            # Subject holdout logic
            # participant_id is implicitly the 'vp1' part
            subject_id = file_id.split('/')[0] # e.g., 'vp1'
            if self.split is not None:
                is_held_out = subject_id in self.held_out_subjects
                if self.split == 'test' and not is_held_out:
                    continue
                if self.split == 'train' and is_held_out:
                    continue

            # Determine label
            cap = cv2.VideoCapture(face_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()
            else:
                total_frames = 5000

            distracted_frames = np.zeros(max(total_frames, 1), dtype=bool)
            for act in actions:
                if act['activity'] in DISTRACTED_ACTIVITIES:
                    start = act['start']
                    end = min(act['end'] + 1, total_frames)
                    distracted_frames[start:end] = True
            
            # > 20% rule
            is_distracted = 1.0 if (distracted_frames.sum() / max(len(distracted_frames), 1)) > 0.20 else 0.0

            # Default others to 0 (Normal)
            label = torch.tensor([0.0, is_distracted, 0.0, 0.0, 0.0], dtype=torch.float32)

            self.samples.append((face_path, body_path, label))

        print(f"[Drive&Act] Loaded {len(self.samples)} valid dual-stream videos.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face_path, body_path, label = self.samples[idx]

        def process_video(video_path):
            events = video_to_events(video_path, max_frames=10)
            if len(events) < 50:
                return torch.zeros((5, 5, 64, 64))

            num_chunks = 5
            chunk_size = max(len(events) // num_chunks, 1)

            voxels = []
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size
                chunk = events[start_idx:end_idx]
                voxel = events_to_voxel(chunk, num_bins=5)
                voxel = voxel / voxel.max() if voxel.max() != 0 else voxel
                voxels.append(voxel)

            voxels = torch.stack(voxels)
            voxels = torch.clamp(voxels, -1.0, 1.0)
            if self.augment:
                voxels = augment_voxels(voxels)
            return voxels

        voxels_face = process_video(face_path)
        voxels_body = process_video(body_path)

        return (voxels_face, voxels_body), label, torch.tensor(-1, dtype=torch.long)

"""
dataset/dataset_3mdad.py
─────────────────────────────────────────────────────────────────────────────
Custom PyTorch Dataset for the 3MDAD (Multimodal, Multiview, Multispectral
Driver Action Dataset).

Dataset structure:
    data/3mdad/
        Day-RGB1/RGB1/S1/AC1/*.tiff   ← Face camera (day, RGB)
        Day-RGB2/RGB2/S1/AC1/*.tiff   ← Body camera (day, RGB)
        Night-IR1/IR1/S1/AC1/*.tiff   ← Face camera (night, IR)
        Night-IR2/IR2/S1/AC1/*.tiff   ← Body camera (night, IR)
        Day-Depth1/ Night-Depth1/ etc. ← unused for now

Action → Label mapping (5-class multi-label):
    [is_drowsy, is_distracted, is_glasses, is_sunglasses, is_night]

    AC1  → Safe driving → [0, 0, 0, 0, is_night]
    AC2  → Hair & makeup → distracted
    AC3  → Adjusting radio → distracted
    AC4  → GPS operating → distracted
    AC5  → Writing msg (right) → distracted
    AC6  → Writing msg (left)  → distracted
    AC7  → Talking phone (right) → distracted
    AC8  → Talking phone (left)  → distracted
    AC9  → Taking picture → distracted
    AC10 → Talking to passenger → distracted
    AC11 → Singing/dancing → distracted
    AC12 → Fatigue/somnolence → DROWSY
    AC13 → Drinking (right) → distracted
    AC14 → Drinking (left)  → distracted
    AC15 → Reaching behind → distracted
    AC16 → Smoking → distracted
"""

import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from dataset.preprocessing import events_to_voxel
from dataset.video_dataset import augment_voxels

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Actions that count as distracted (all except AC1=safe, AC12=drowsy)
DISTRACTED_ACTIONS = {
    "AC2", "AC3", "AC4", "AC5", "AC6", "AC7", "AC8",
    "AC9", "AC10", "AC11", "AC13", "AC14", "AC15", "AC16",
}
DROWSY_ACTIONS = {"AC12"}

# Camera folders inside data/3mdad for each modality
# Each entry: (face_folder_prefix, body_folder_prefix, inner_dir, is_night)
CAMERA_CONFIGS = [
    ("Day-RGB1",   "Day-RGB2",   "RGB1",  "RGB2",  False),
    ("Night-IR1",  "Night-IR2",  "IR1",   "IR2",   True),
]

# Number of .tiff frames to sample per clip
MAX_FRAMES = 20
# Temporal step between sampled frames
FRAME_STEP = 3


# ---------------------------------------------------------------------------
# Helper: convert a list of .tiff images into event voxels via frame diff
# ---------------------------------------------------------------------------

def tiff_sequence_to_events(img_paths: list, target_h=64, target_w=64):
    """
    Simulate asynchronous events from a sequence of grayscale images using
    simple frame differencing. Returns an (N, 4) array: [x, y, t, polarity].
    """
    import cv2

    all_events = []
    prev_gray = None

    for t_idx, path in enumerate(img_paths):
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue

        # Resize to target resolution
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)
        img_f = img.astype(np.float32) / 255.0

        if prev_gray is not None:
            diff = img_f - prev_gray

            # Positive events (brightness increase)
            pos_y, pos_x = np.where(diff > 0.05)
            for y, x in zip(pos_y, pos_x):
                all_events.append([x, y, t_idx, 1.0])

            # Negative events (brightness decrease)
            neg_y, neg_x = np.where(diff < -0.05)
            for y, x in zip(neg_y, neg_x):
                all_events.append([x, y, t_idx, -1.0])

        prev_gray = img_f

    if len(all_events) == 0:
        return np.zeros((0, 4), dtype=np.float32)

    return np.array(all_events, dtype=np.float32)


def load_tiff_stream(folder: str, max_frames=MAX_FRAMES, step=FRAME_STEP):
    """
    Load up to `max_frames` frames from a folder of .tiff files.
    Returns a (5, 5, 64, 64) voxel tensor.
    """
    if not os.path.isdir(folder):
        return torch.zeros((5, 5, 64, 64))

    files = sorted([
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".tiff") or f.lower().endswith(".tif")
    ])

    if len(files) == 0:
        return torch.zeros((5, 5, 64, 64))

    # Sub-sample evenly
    sampled = files[::step][:max_frames]
    if len(sampled) < 2:
        sampled = files[:max_frames]

    events = tiff_sequence_to_events(sampled)

    if len(events) < 10:
        return torch.zeros((5, 5, 64, 64))

    # Split events into 5 temporal chunks → one voxel per chunk
    num_chunks = 5
    chunk_size = max(len(events) // num_chunks, 1)
    voxels = []
    for i in range(num_chunks):
        chunk = events[i * chunk_size: (i + 1) * chunk_size]
        voxel = events_to_voxel(chunk, num_bins=5, height=64, width=64)
        vmax = voxel.max()
        voxel = voxel / vmax if vmax != 0 else voxel
        voxels.append(voxel)

    voxels = torch.stack(voxels)           # (5, 5, 64, 64)
    voxels = torch.clamp(voxels, -1.0, 1.0)
    return voxels


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class Dataset3MDAD(Dataset):
    """
    Loads 3MDAD samples.

    Each sample is a pair:
        (voxels_face, voxels_body) → dual-stream event voxels
        label → 5-dim tensor [is_drowsy, is_distracted, is_glasses, is_sunglasses, is_night]

    Subject-level train/val split is supported via `split` and
    `held_out_subjects` arguments (e.g. {"S1", "S4"}).
    """

    def __init__(
        self,
        root: str,
        augment: bool = False,
        split: str = None,
        held_out_subjects: set = None,
    ):
        self.root = os.path.expanduser(root)
        self.augment = augment
        self.split = split
        self.held_out_subjects = {s.upper() for s in (held_out_subjects or set())}
        self.samples = []   # list of (face_dir, body_dir, label_tensor)

        self._build_sample_list()

        drowsy_ct = sum(1 for _, _, l in self.samples if l[0] == 1.0)
        distract_ct = sum(1 for _, _, l in self.samples if l[1] == 1.0)
        night_ct = sum(1 for _, _, l in self.samples if l[4] == 1.0)

        print(
            f"[3MDAD] split={split} | total={len(self.samples)} | "
            f"drowsy={drowsy_ct} | distracted={distract_ct} | night={night_ct}"
        )

    # ------------------------------------------------------------------

    def _build_sample_list(self):
        for (face_outer, body_outer, face_inner, body_inner, is_night) in CAMERA_CONFIGS:

            face_root = os.path.join(self.root, face_outer, face_inner)
            body_root = os.path.join(self.root, body_outer, body_inner)

            if not os.path.isdir(face_root) or not os.path.isdir(body_root):
                continue

            # Iterate subjects (S1 … S50)
            for subj in sorted(os.listdir(face_root)):
                if not subj.upper().startswith("S"):
                    continue

                # --- Subject-level split filtering ---
                is_held_out = subj.upper() in self.held_out_subjects
                if self.split == "test" and not is_held_out:
                    continue
                if self.split == "train" and is_held_out:
                    continue

                subj_face = os.path.join(face_root, subj)
                subj_body = os.path.join(body_root, subj)

                if not os.path.isdir(subj_face):
                    continue

                for action in sorted(os.listdir(subj_face)):
                    if not action.upper().startswith("AC"):
                        continue

                    face_dir = os.path.join(subj_face, action)
                    body_dir = os.path.join(subj_body, action) if os.path.isdir(subj_body) else face_dir

                    label = self._make_label(action.upper(), is_night)
                    self.samples.append((face_dir, body_dir, label))

    # ------------------------------------------------------------------

    @staticmethod
    def _make_label(action_str: str, is_night: bool) -> torch.Tensor:
        """Map action name like 'AC12' → 5-dim label tensor."""
        is_drowsy    = 1.0 if action_str in DROWSY_ACTIONS else 0.0
        is_distracted = 1.0 if action_str in DISTRACTED_ACTIONS else 0.0
        is_glasses    = 0.0   # 3MDAD does not track glasses
        is_sunglasses = 0.0
        is_night_f    = 1.0 if is_night else 0.0

        return torch.tensor(
            [is_drowsy, is_distracted, is_glasses, is_sunglasses, is_night_f],
            dtype=torch.float32,
        )

    # ------------------------------------------------------------------

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face_dir, body_dir, label = self.samples[idx]

        voxels_face = load_tiff_stream(face_dir)
        voxels_body = load_tiff_stream(body_dir)

        if self.augment:
            voxels_face = augment_voxels(voxels_face)
            voxels_body = augment_voxels(voxels_body)

        # subject_idx = -1 (3MDAD subjects not mapped to adversarial head)
        return (voxels_face, voxels_body), label, torch.tensor(-1, dtype=torch.long)

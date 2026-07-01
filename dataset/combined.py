"""
dataset/combined.py
───────────────────
Utility that merges the DMD (primary) and Drive&Act (secondary) datasets
into a single ConcatDataset for training and validation.

Both datasets now emit 3-tuples: (x, label, subject_idx)
  - VideoDataset    → subject_idx is the DMD subject index (0-based) or -1
  - DriveAndActDataset → subject_idx is always -1 (unknown/excluded from
                         the adversarial subject head)

Usage
-----
    from dataset.combined import build_combined_datasets

    train_ds, val_ds, weights = build_combined_datasets(
        dmd_path           = "~/Desktop/dmd",
        driveandact_path   = "~/Downloads/driveandact",   # or None
        held_out_subjects  = {"s1", "s4"},
        device             = torch.device("cpu"),
    )
"""
import os
import torch
from torch.utils.data import ConcatDataset

from dataset.video_dataset import VideoDataset
from dataset.drive_and_act import DriveAndActDataset
from dataset.dataset_3mdad import Dataset3MDAD


# Drive&Act subject IDs used as the held-out val split for that dataset.
# (Drive&Act uses "vp<N>" naming — pick a small fixed subset.)
DRIVEANDACT_HELD_OUT = {"vp1", "vp2"}


def _compute_weights(datasets, device, num_classes=5, max_weight=5.0):
    """
    Compute per-class inverse-frequency weights from the *training* labels
    of one or more datasets (each must have a .samples list of tuples where
    index 2 is the label tensor).
    """
    all_labels = []
    for ds in datasets:
        for sample in ds.samples:
            # sample[2] is the label tensor for both VideoDataset and DriveAndActDataset
            all_labels.append(sample[2])

    if not all_labels:
        return torch.ones(num_classes, dtype=torch.float, device=device)

    labels = torch.stack(all_labels)          # (N, num_classes)
    total  = len(labels)
    weights = torch.tensor(
        [min(total / max(labels[:, i].sum().item(), 1), max_weight) for i in range(num_classes)],
        dtype=torch.float,
        device=device,
    )
    return weights


def build_combined_datasets(
    dmd_path,
    driveandact_path=None,
    mdad_path=None,
    held_out_subjects=None,
    device=None,
    augment_train=True,
    num_classes=5,
):
    """
    Build combined train and val datasets from DMD, Drive&Act, and/or 3MDAD.

    Parameters
    ----------
    dmd_path : str
        Path to the DMD dataset root (same as DATA_PATH in training scripts).
    driveandact_path : str | None
        Path to the Drive&Act dataset root.  Pass None (or a non-existent
        path) to skip — the function degrades gracefully to DMD-only.
    mdad_path : str | None
        Path to the 3MDAD dataset root (data/3mdad).  Pass None to skip.
    held_out_subjects : set[str] | None
        DMD subject IDs to reserve for validation (e.g. {"s1", "s4"}).
    device : torch.device | None
        Device to place class-weight tensors on.  Defaults to CPU.
    augment_train : bool
        Whether to enable data augmentation on the training split.
    num_classes : int
        Number of output classes (default 5).

    Returns
    -------
    train_ds : Dataset
        Combined training dataset (ConcatDataset or plain VideoDataset).
    val_ds : Dataset
        Combined validation dataset (ConcatDataset or plain VideoDataset).
    weights : torch.Tensor  shape (num_classes,)
        Per-class inverse-frequency weights computed from *training* data only.
    """
    held_out_subjects = held_out_subjects or set()
    device = device or torch.device("cpu")

    # ── Primary dataset: DMD ──────────────────────────────────────────────────
    dmd_train = VideoDataset(
        os.path.expanduser(dmd_path),
        augment=augment_train,
        split="train",
        held_out_subjects=held_out_subjects,
    )
    dmd_val = VideoDataset(
        os.path.expanduser(dmd_path),
        augment=False,
        split="test",
        held_out_subjects=held_out_subjects,
    )

    train_datasets = [dmd_train] if len(dmd_train) > 0 else []
    val_datasets   = [dmd_val]   if len(dmd_val) > 0   else []

    # ── Secondary dataset: Drive&Act (optional) ───────────────────────────────
    daa_path = os.path.expanduser(driveandact_path) if driveandact_path else ""
    if daa_path and os.path.isdir(daa_path):
        daa_train = DriveAndActDataset(
            daa_path,
            augment=augment_train,
            split="train",
            held_out_subjects=DRIVEANDACT_HELD_OUT,
        )
        daa_val = DriveAndActDataset(
            daa_path,
            augment=False,
            split="test",
            held_out_subjects=DRIVEANDACT_HELD_OUT,
        )

        if len(daa_train) > 0:
            print(f"[Combined] Drive&Act train: {len(daa_train)} | val: {len(daa_val)} videos")
            train_datasets.append(daa_train)
        if len(daa_val) > 0:
            val_datasets.append(daa_val)
    else:
        if driveandact_path:
            print(f"[Combined] Drive&Act path not found ({daa_path}) — skipping.")
        else:
            print("[Combined] No Drive&Act path provided — skipping.")

    # ── Tertiary dataset: 3MDAD (optional) ───────────────────────────────────
    # 3MDAD held-out subjects for validation: S1, S4 (same naming convention)
    MDAD_HELD_OUT = {"S1", "S4"}
    mdad_root = os.path.expanduser(mdad_path) if mdad_path else ""
    if mdad_root and os.path.isdir(mdad_root):
        mdad_train = Dataset3MDAD(
            mdad_root,
            augment=augment_train,
            split="train",
            held_out_subjects=MDAD_HELD_OUT,
        )
        mdad_val = Dataset3MDAD(
            mdad_root,
            augment=False,
            split="test",
            held_out_subjects=MDAD_HELD_OUT,
        )
        if len(mdad_train) > 0:
            print(f"[Combined] 3MDAD train: {len(mdad_train)} | val: {len(mdad_val)} samples")
            train_datasets.append(mdad_train)
        if len(mdad_val) > 0:
            val_datasets.append(mdad_val)
    else:
        if mdad_path:
            print(f"[Combined] 3MDAD path not found ({mdad_root}) — skipping.")
        else:
            print("[Combined] No 3MDAD path provided — skipping.")

    # ── Merge ─────────────────────────────────────────────────────────────────
    train_ds = ConcatDataset(train_datasets) if len(train_datasets) > 1 else train_datasets[0]
    val_ds   = ConcatDataset(val_datasets)   if len(val_datasets) > 1   else val_datasets[0]

    print(
        f"[Combined] Total train: {len(train_ds)} | Total val: {len(val_ds)} samples"
    )

    # ── Class weights (train only) ────────────────────────────────────────────
    weights = _compute_weights(train_datasets, device, num_classes=num_classes)

    return train_ds, val_ds, weights

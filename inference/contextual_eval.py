"""
Contextual Accuracy Evaluation for the Full SNN + BiGRU + Attention model.

Evaluates Drowsy/Distracted detection accuracy CONDITIONED on environmental context:
  - Daylight vs Nighttime
  - With Glasses vs Without Glasses
  - With Sunglasses vs Without Sunglasses

Uses the FULL dataset (no train/test split) to ensure enough samples per condition
for a meaningful breakdown. Results are reported separately for each context group.

NOTE for paper: Flag these as full-dataset figures (not held-out test set) to avoid
any misinterpretation. The held-out test set results come from compare_baselines.py.

Usage:
    python3 -m inference.contextual_eval
"""
import os
import sys
import torch
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_full import DriverMonitoringModel
from dataset.video_dataset import VideoDataset
from dataset.combined import build_combined_datasets
from spikingjelly.activation_based import functional

DATA_PATH        = os.path.expanduser("~/Desktop/dmd")
# Set to your Drive&Act root to include it in the contextual breakdown, or None to skip.
DRIVEANDACT_PATH = os.path.expanduser("~/Downloads")  # folders: inner_mirror, a_column_driver, activities_3s
THRESHOLD = 0.5

# Label indices
IDX_DROWSY     = 0
IDX_DISTRACTED = 1
IDX_GLASSES    = 2
IDX_SUNGLASSES = 3
IDX_NIGHT      = 4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def run_inference(model, dataset):
    """Run full inference, return arrays of shape (N, 5)."""
    model.eval()
    all_targets, all_probs = [], []

    for i in range(len(dataset)):
        # dataset[i] is a 3-tuple: (x, label, subject_idx)
        x_face, x_body = dataset[i][0]
        y = dataset[i][1]
        x_face = x_face.unsqueeze(0).to(device)
        x_body = x_body.unsqueeze(0).to(device)

        with torch.no_grad():
            functional.reset_net(model)
            out = model(x_face, x_body)
            # DriverMonitoringModel returns (behavior_logits, subject_logits)
            logits = out[0] if isinstance(out, tuple) else out
            probs  = torch.sigmoid(logits)[0].cpu().numpy()

        all_targets.append(y.numpy())
        all_probs.append(probs)
        if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
            print(f"  {i+1}/{len(dataset)}", end="\r")

    print()
    return np.array(all_targets), np.array(all_probs)


def group_metrics(targets, probs, mask, group_name, task_names, task_indices):
    """
    Evaluate detection accuracy for a subset of samples defined by mask.
    Reports precision, recall, and F1 for each task (drowsy/distracted).
    """
    if mask.sum() == 0:
        print(f"  {group_name}: No samples found.")
        return {}

    t_sub = targets[mask]
    p_sub = (probs[mask] > THRESHOLD).astype(int)

    results = {}
    rows = []
    for name, idx in zip(task_names, task_indices):
        if t_sub[:, idx].sum() == 0:
            rows.append(f"    {name:<14}: No positive samples in this group")
            results[name] = None
            continue
        prec = precision_score(t_sub[:, idx], p_sub[:, idx], zero_division=0)
        rec  = recall_score   (t_sub[:, idx], p_sub[:, idx], zero_division=0)
        f1   = f1_score       (t_sub[:, idx], p_sub[:, idx], zero_division=0)
        exact = 100 * (p_sub[:, idx] == t_sub[:, idx]).mean()
        rows.append(f"    {name:<14}: Prec={prec:.2f}  Rec={rec:.2f}  F1={f1:.2f}  Acc={exact:.0f}%")
        results[name] = {"precision": prec, "recall": rec, "f1": f1, "accuracy": exact}

    exact_all = 100 * ((p_sub == t_sub).all(axis=1)).mean()
    print(f"\n  [{group_name}]  ({mask.sum()} samples)")
    for r in rows:
        print(r)
    print(f"    {'Exact Match':<14}: {exact_all:.0f}% (all 5 labels correct)")
    return results


def main():
    if not os.path.exists("model.pth"):
        print("ERROR: model.pth not found. Train the model first.")
        sys.exit(1)

    # Load FULL dataset (no split — to maximise samples per condition group)
    # Combines DMD + Drive&Act (if available) with no held-out split.
    dataset, _, _ = build_combined_datasets(
        dmd_path          = DATA_PATH,
        driveandact_path  = DRIVEANDACT_PATH,
        held_out_subjects = set(),  # no holdout — use all data for condition breakdown
        augment_train     = False,
    )
    if len(dataset) == 0:
        print("ERROR: No videos found.")
        sys.exit(1)

    print(f"Full dataset: {len(dataset)} videos\n")
    print("Running inference...")

    model = DriverMonitoringModel(num_classes=5)
    model.load_state_dict(torch.load("model.pth", map_location=device))
    model.to(device)

    targets, probs = run_inference(model, dataset)

    task_names   = ["Drowsy", "Distracted"]
    task_indices = [IDX_DROWSY, IDX_DISTRACTED]

    sep = "=" * 65

    print(f"\n{sep}")
    print("  SNN + BiGRU + Attention — Contextual Accuracy Breakdown")
    print(f"  Dataset: {len(dataset)} videos (full dataset, no train/test split)")
    print(sep)

    # ── Lighting Condition ────────────────────────────────────────────────────
    print("\n── Lighting Condition ─────────────────────────────────────────")
    daylight_mask = targets[:, IDX_NIGHT] == 0
    night_mask    = targets[:, IDX_NIGHT] == 1

    group_metrics(targets, probs, daylight_mask, "DAYLIGHT",  task_names, task_indices)
    group_metrics(targets, probs, night_mask,    "NIGHTTIME", task_names, task_indices)

    # ── Eyewear Condition ─────────────────────────────────────────────────────
    print("\n── Eyewear Condition ──────────────────────────────────────────")
    glasses_mask    = targets[:, IDX_GLASSES] == 1
    no_glasses_mask = (targets[:, IDX_GLASSES] == 0) & (targets[:, IDX_SUNGLASSES] == 0)
    sungl_mask      = targets[:, IDX_SUNGLASSES] == 1

    group_metrics(targets, probs, no_glasses_mask, "NO EYEWEAR",   task_names, task_indices)
    group_metrics(targets, probs, glasses_mask,    "GLASSES",      task_names, task_indices)
    group_metrics(targets, probs, sungl_mask,      "SUNGLASSES",   task_names, task_indices)

    # ── Overall ───────────────────────────────────────────────────────────────
    print("\n── Overall ────────────────────────────────────────────────────")
    all_mask = np.ones(len(targets), dtype=bool)
    group_metrics(targets, probs, all_mask, "ALL CONDITIONS", task_names, task_indices)

    print(f"\n{sep}\n")
    print("NOTE: These figures use the full dataset (train + test) for")
    print("      sufficient samples per condition. For paper evaluation")
    print("      on unseen subjects only, run: python3 -m inference.compare_baselines")


if __name__ == "__main__":
    main()

import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_full import DriverMonitoringModel
from dataset.video_dataset import VideoDataset
from dataset.combined import build_combined_datasets
from spikingjelly.activation_based import functional

# Multi-label class names for the 5-target SNN classifier
CLASS_NAMES = ["Drowsy", "Distracted", "Glasses", "Sunglasses", "Night"]

# Must match HELD_OUT_SUBJECTS in training/train.py exactly
HELD_OUT_SUBJECTS = {"s1", "s4"}

# ── Dataset config ────────────────────────────────────────────────────────────
DATA_PATH        = os.path.expanduser("~/Downloads")
# Set to your Drive&Act root to include it in evaluation, or None to skip.
DRIVEANDACT_PATH = os.path.expanduser("~/Downloads")  # folders: inner_mirror, a_column_driver, activities_3s
MDAD_PATH        = os.path.expanduser("~/Downloads/neuromorphic-drive-monitoring/data/3mdad")

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists("model.pth"):
    print("ERROR: model.pth not found. Run training first.")
    sys.exit(1)

state_dict = torch.load("model.pth", map_location="cpu")
num_subjects = state_dict["subject_classifier.2.weight"].shape[0] if "subject_classifier.2.weight" in state_dict else 1

model = DriverMonitoringModel(num_subjects=num_subjects)
model.load_state_dict(state_dict)
model.eval()

# ── Dataset (held-out test subjects — DMD + Drive&Act if available) ───────────
_, dataset, _ = build_combined_datasets(
    dmd_path          = DATA_PATH,
    driveandact_path  = DRIVEANDACT_PATH,
    mdad_path         = MDAD_PATH,
    held_out_subjects = HELD_OUT_SUBJECTS,
)

if len(dataset) == 0:
    print(f"No test videos found for held-out subjects {HELD_OUT_SUBJECTS}.")
    print("Check that these subject IDs exist in your dataset blocks.")
    sys.exit(1)

print(f"Evaluating on {len(dataset)} held-out test videos "
      f"(subjects: {HELD_OUT_SUBJECTS}) — never seen during training.\n")

# ── Inference ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Neuromorphic Driver Monitoring — Multi-Label Inference")
print("=" * 65)

exact_matches = 0
all_preds = []
all_targets = []

def get_label_str(pred):
    labels = []
    if pred[0]: labels.append("DROWSY 😴")
    if pred[1]: labels.append("DISTRACTED 📱")
    if pred[2]: labels.append("GLASSES 👓")
    if pred[3]: labels.append("SUNGLASSES 🕶️")
    if pred[4]: labels.append("NIGHT 🌙")
    return " + ".join(labels) if labels else "NORMAL 🙂"


def get_video_name(ds, idx):
    """Return the face-video basename for sample idx.
    Works for both VideoDataset (has .samples) and ConcatDataset."""
    try:
        return os.path.basename(ds.samples[idx][0])
    except AttributeError:
        # ConcatDataset — walk sub-datasets to find the local index
        cumulative = 0
        for sub_ds in ds.datasets:
            if idx < cumulative + len(sub_ds):
                return os.path.basename(sub_ds.samples[idx - cumulative][0])
            cumulative += len(sub_ds)
        return f"sample_{idx}"

for i in range(len(dataset)):
    (x_face, x_body), y, _ = dataset[i]
    
    x_face = x_face.unsqueeze(0)  # (1, time, C, H, W)
    x_body = x_body.unsqueeze(0)

    with torch.no_grad():
        functional.reset_net(model)
        logits, _ = model(x_face, x_body)
        probs  = torch.sigmoid(logits)[0]
        
    pred = (probs > 0.5).float()

    all_preds.append(pred.cpu().numpy())
    all_targets.append(y.cpu().numpy())

    video_name = get_video_name(dataset, i)
    
    d_pct, di_pct, g_pct, sg_pct, n_pct = (probs * 100).tolist()
    
    pred_str = get_label_str(pred)
    true_str = get_label_str(y)
    
    match = "✓" if (pred == y).all() else "✗"

    print(f"\n[{match}] {video_name}")
    print(f"    Predicted : {pred_str}  (Dr:{d_pct:.1f}% Di:{di_pct:.1f}% G:{g_pct:.1f}% SG:{sg_pct:.1f}% N:{n_pct:.1f}%)")
    print(f"    Actual    : {true_str}")

    if (pred == y).all():
        exact_matches += 1

import numpy as np
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

all_targets = np.array(all_targets)
all_preds = np.array(all_preds)

accuracy = 100 * exact_matches / len(dataset)
print("\n" + "=" * 65)
print(f"  Exact Match Accuracy: {exact_matches}/{len(dataset)} ({accuracy:.0f}%)")
print("=" * 65)

print("\n" + "=" * 65)
print("  Multi-Label Classification Report")
print("=" * 65)
print(classification_report(all_targets, all_preds, target_names=["Drowsy", "Distracted", "Glasses", "Sunglasses", "Night"], zero_division=0))

print("\n" + "=" * 65)
print("  Multi-Label Confusion Matrices")
print("=" * 65)
mcm = multilabel_confusion_matrix(all_targets, all_preds)
names = ["Drowsy", "Distracted", "Glasses", "Sunglasses", "Night"]
for i, matrix in enumerate(mcm):
    print(f"\n[{names[i]}]")
    print(f"TN: {matrix[0,0]} | FP: {matrix[0,1]}")
    print(f"FN: {matrix[1,0]} | TP: {matrix[1,1]}")
print("=" * 65 + "\n")

# Plot the confusion matrices
fig, axes = plt.subplots(1, 5, figsize=(20, 4))
for i, matrix in enumerate(mcm):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[i].set_title(f'{names[i]}')
    axes[i].set_ylabel('Actual')
    axes[i].set_xlabel('Predicted')
    
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
print("Saved confusion matrix graph to: confusion_matrix.png\n")
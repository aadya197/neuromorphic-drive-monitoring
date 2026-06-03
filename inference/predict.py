import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_full import DriverMonitoringModel
from dataset.video_dataset import VideoDataset
from spikingjelly.activation_based import functional

LABELS = {0: "NORMAL 🙂", 1: "DROWSY 😴", 2: "DISTRACTED 📱", 3: "DROWSY+DISTRACTED 😴📱"}

# ── Load model ────────────────────────────────────────────────────────────────
if not os.path.exists("model.pth"):
    print("ERROR: model.pth not found. Run training first.")
    sys.exit(1)

model = DriverMonitoringModel(num_classes=2)
model.load_state_dict(torch.load("model.pth", map_location="cpu"))
model.eval()

# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = VideoDataset(os.path.expanduser("~/Desktop/dmd"), augment=False)

if len(dataset) == 0:
    print("No videos found.")
    sys.exit(1)

# ── Inference ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("  Neuromorphic Driver Monitoring — Multi-Label Inference")
print("=" * 65)

exact_matches = 0
all_preds = []
all_targets = []

def get_label_str(is_drowsy, is_dist):
    if is_drowsy and is_dist: return LABELS[3]
    if is_drowsy: return LABELS[1]
    if is_dist: return LABELS[2]
    return LABELS[0]

for i in range(len(dataset)):
    x_face, x_body = dataset[i][0]
    y = dataset[i][1]
    
    x_face = x_face.unsqueeze(0)  # (1, time, C, H, W)
    x_body = x_body.unsqueeze(0)

    with torch.no_grad():
        functional.reset_net(model)
        logits = model(x_face, x_body)
        probs  = torch.sigmoid(logits)[0]
        
    pred = (probs > 0.5).float()

    all_preds.append(pred.cpu().numpy())
    all_targets.append(y.cpu().numpy())

    video_name = os.path.basename(dataset.samples[i][0])
    
    drowsy_pct = probs[0].item() * 100
    dist_pct   = probs[1].item() * 100
    
    pred_str = get_label_str(pred[0].item() > 0.5, pred[1].item() > 0.5)
    true_str = get_label_str(y[0].item() > 0.5, y[1].item() > 0.5)
    
    match = "✓" if (pred == y).all() else "✗"

    print(f"\n[{match}] {video_name}")
    print(f"    Predicted : {pred_str}  (Drowsy {drowsy_pct:.1f}% | Distracted {dist_pct:.1f}%)")
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
print(classification_report(all_targets, all_preds, target_names=["Drowsy", "Distracted"], zero_division=0))

print("\n" + "=" * 65)
print("  Multi-Label Confusion Matrices")
print("=" * 65)
mcm = multilabel_confusion_matrix(all_targets, all_preds)
names = ["Drowsy", "Distracted"]
for i, matrix in enumerate(mcm):
    print(f"\n[{names[i]}]")
    print(f"TN: {matrix[0,0]} | FP: {matrix[0,1]}")
    print(f"FN: {matrix[1,0]} | TP: {matrix[1,1]}")
print("=" * 65 + "\n")

# Plot the confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for i, matrix in enumerate(mcm):
    sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'])
    axes[i].set_title(f'Confusion Matrix: {names[i]}')
    axes[i].set_ylabel('Actual')
    axes[i].set_xlabel('Predicted')
    
plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
print("Saved confusion matrix graph to: confusion_matrix.png\n")
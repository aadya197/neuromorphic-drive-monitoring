import os
import sys
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_full import DriverMonitoringModel
from dataset.video_dataset import VideoDataset
from spikingjelly.activation_based import functional

LABELS = {0: "NORMAL 🙂", 1: "DROWSY 😴"}

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
print("\n" + "=" * 55)
print("  Neuromorphic Driver Monitoring — Inference Results")
print("=" * 55)

correct = 0
all_preds = []
all_targets = []

for i in range(len(dataset)):
    x, y = dataset[i]
    x = x.unsqueeze(0)  # (1, time, C, H, W)

    with torch.no_grad():
        # CRITICAL: reset SNN neuron states before every sample
        functional.reset_net(model)
        logits = model(x)
        probs  = F.softmax(logits, dim=1)[0]
        pred   = torch.argmax(probs).item()

    all_preds.append(pred)
    all_targets.append(y)

    video_name = os.path.basename(dataset.samples[i][0])
    normal_pct = probs[0].item() * 100
    drowsy_pct = probs[1].item() * 100
    match      = "✓" if pred == y else "✗"

    print(f"\n[{match}] {video_name}")
    print(f"    Predicted : {LABELS[pred]}  (Normal {normal_pct:.1f}% | Drowsy {drowsy_pct:.1f}%)")
    print(f"    Actual    : {LABELS[y]}")

    if pred == y:
        correct += 1

from sklearn.metrics import classification_report, confusion_matrix

accuracy = 100 * correct / len(dataset)
print("\n" + "=" * 55)
print(f"  Accuracy: {correct}/{len(dataset)} ({accuracy:.0f}%)")
print("=" * 55)

print("\n" + "=" * 55)
print("  Classification Report (0: Normal, 1: Drowsy)")
print("=" * 55)
print(classification_report(all_targets, all_preds, target_names=["Normal", "Drowsy"], zero_division=0))

print("\n" + "=" * 55)
print("  Confusion Matrix")
print("=" * 55)
cm = confusion_matrix(all_targets, all_preds)
print(f"                Predicted Normal | Predicted Drowsy")
print(f"Actual Normal :       {cm[0][0]:<10} |       {cm[0][1]:<10}")
if len(cm) > 1:
    print(f"Actual Drowsy :       {cm[1][0]:<10} |       {cm[1][1]:<10}")
print("=" * 55 + "\n")
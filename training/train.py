import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spikingjelly.activation_based import functional
from models.model_full import DriverMonitoringModel
from dataset.video_dataset import VideoDataset

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH  = os.path.expanduser("~/Desktop/dmd")
EPOCHS     = 200
BATCH_SIZE = 4
LR         = 3e-4
SAVE_PATH  = "model.pth"

# Subject IDs held out exclusively for testing — NEVER seen during training.
# These must remain fixed for the lifetime of the paper to ensure reproducibility.
# Based on the DMD dataset, subjects s1 and s4 are held out (~20% of data).
HELD_OUT_SUBJECTS = {"s1", "s4"}


# ── Multi-Label Focal Loss ──────────────────────────────────────────────────
class MultiLabelFocalLoss(nn.Module):
    """
    Focal Loss for Multi-label classification (BCE based).
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha  # (num_classes,)
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            loss = loss * self.alpha
        return loss.mean()


# ── Dataset (subject-level split) ─────────────────────────────────────────────
train_dataset = VideoDataset(DATA_PATH, augment=True,
                             split='train', held_out_subjects=HELD_OUT_SUBJECTS)
val_dataset   = VideoDataset(DATA_PATH, augment=False,
                             split='test',  held_out_subjects=HELD_OUT_SUBJECTS)

if len(train_dataset) == 0:
    print("ERROR: No training videos found. Check DATA_PATH and HELD_OUT_SUBJECTS.")
    sys.exit(1)

if len(val_dataset) == 0:
    print("WARNING: No validation videos found. Check HELD_OUT_SUBJECTS — they may "
          "not exist in your current dataset blocks.")

print(f"\nTrain set: {len(train_dataset)} videos | "
      f"Val set (held-out subjects {HELD_OUT_SUBJECTS}): {len(val_dataset)} videos\n")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=0, pin_memory=False)

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = DriverMonitoringModel().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total_params:,}\n")

# Auto class weights computed from TRAIN set only (not the whole dataset)
labels = torch.stack([s[2] for s in train_dataset.samples])
total  = len(labels)
n_drowsy     = labels[:, 0].sum().item()
n_dist       = labels[:, 1].sum().item()
n_glasses    = labels[:, 2].sum().item()
n_sunglasses = labels[:, 3].sum().item()
n_night      = labels[:, 4].sum().item()

w_drowsy     = min(total / max(n_drowsy, 1), 5.0)
w_dist       = min(total / max(n_dist, 1), 5.0)
w_glasses    = min(total / max(n_glasses, 1), 5.0)
w_sunglasses = min(total / max(n_sunglasses, 1), 5.0)
w_night      = min(total / max(n_night, 1), 5.0)

weights = torch.tensor([w_drowsy, w_dist, w_glasses, w_sunglasses, w_night],
                       dtype=torch.float).to(device)
print(f"Train class counts  → Drowsy: {int(n_drowsy)} | Dist: {int(n_dist)} | "
      f"Glasses: {int(n_glasses)} | Sunglasses: {int(n_sunglasses)} | Night: {int(n_night)}")
print(f"Train class weights → Drowsy: {w_drowsy:.2f} | Dist: {w_dist:.2f} | "
      f"Glasses: {w_glasses:.2f} | Sunglasses: {w_sunglasses:.2f} | Night: {w_night:.2f}\n")

loss_fn   = MultiLabelFocalLoss(alpha=weights, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_acc = -1.0

for epoch in range(EPOCHS):
    # --- Train ---
    model.train()
    total_loss, correct, total_preds = 0.0, 0, 0

    for x, y in train_loader:
        x_face, x_body = x
        x_face, x_body = x_face.to(device), x_body.to(device)
        y = y.to(device)

        functional.reset_net(model)

        pred = model(x_face, x_body)
        
        # Label smoothing to prevent overconfidence (combats overfitting)
        y_smoothed = y * (1 - 0.1) + 0.05
        loss = loss_fn(pred, y_smoothed)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        probs = torch.sigmoid(pred)
        predicted = (probs > 0.5).float()
        correct     += (predicted == y).all(dim=1).sum().item()
        total_preds += y.size(0)
        total_loss  += loss.item()

    scheduler.step()

    train_acc = 100 * correct / max(total_preds, 1)
    avg_loss  = total_loss / len(train_loader)
    cur_lr    = scheduler.get_last_lr()[0]

    # --- Validate on held-out subjects ---
    val_acc = 0.0
    if len(val_dataset) > 0:
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x_face, x_body = x
                x_face, x_body = x_face.to(device), x_body.to(device)
                y = y.to(device)
                functional.reset_net(model)
                logits = model(x_face, x_body)
                predicted = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (predicted == y).all(dim=1).sum().item()
                val_total   += y.size(0)
        val_acc = 100 * val_correct / max(val_total, 1)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Loss: {avg_loss:.4f} | Train: {train_acc:.0f}% | Val: {val_acc:.0f}% | LR: {cur_lr:.2e}")

    # Save best model based on VALIDATION accuracy (not training accuracy)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Best model saved (val={val_acc:.0f}%)")

print(f"\nDone. Best Val Exact Match Accuracy: {best_val_acc:.0f}%")
print(f"Held-out test subjects: {HELD_OUT_SUBJECTS}")
print(f"Model saved to: {SAVE_PATH}")

# ── Run final evaluation on held-out test set ─────────────────────────────────
print("\nRunning final evaluation on held-out test set...")
os.system("python3 -m inference.predict")
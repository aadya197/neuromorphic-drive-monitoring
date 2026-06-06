"""
Training script for the CNN baseline model.

Intentionally uses identical hyperparameters, dataset split, loss function,
and evaluation protocol as training/train.py (SNN model) so that the only
variable in the comparison is the model backbone (SNN vs CNN).

Saves weights to: model_baseline_cnn.pth
"""
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.baseline_cnn_gru import BaselineDriverMonitoringModel
from dataset.video_dataset import VideoDataset

# ── Config — identical to train.py ───────────────────────────────────────────
DATA_PATH         = os.path.expanduser("~/Desktop/dmd")
EPOCHS            = 120
BATCH_SIZE        = 4
LR                = 3e-4
SAVE_PATH         = "model_baseline_cnn.pth"
HELD_OUT_SUBJECTS = {"s1", "s4"}   # MUST match train.py


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt  = torch.exp(-bce)
        loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            loss = loss * self.alpha
        return loss.mean()


# ── Dataset (same subject-level split as SNN) ─────────────────────────────────
train_dataset = VideoDataset(DATA_PATH, augment=True,
                             split='train', held_out_subjects=HELD_OUT_SUBJECTS)
val_dataset   = VideoDataset(DATA_PATH, augment=False,
                             split='test',  held_out_subjects=HELD_OUT_SUBJECTS)

if len(train_dataset) == 0:
    print("ERROR: No training videos found.")
    sys.exit(1)

print(f"\n[Baseline CNN] Train: {len(train_dataset)} | Val: {len(val_dataset)} videos\n")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = BaselineDriverMonitoringModel(num_classes=5).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"[Baseline CNN] Parameters: {total_params:,}\n")

# Class weights from train set
labels_t     = torch.stack([s[2] for s in train_dataset.samples])
total        = len(labels_t)
weights_vals = [min(total / max(labels_t[:, i].sum().item(), 1), 5.0) for i in range(5)]
weights      = torch.tensor(weights_vals, dtype=torch.float).to(device)

loss_fn   = MultiLabelFocalLoss(alpha=weights, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# ── Training loop ─────────────────────────────────────────────────────────────
best_val_acc = -1.0

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total_preds = 0.0, 0, 0

    for x, y in train_loader:
        x_face, x_body = x
        x_face, x_body = x_face.to(device), x_body.to(device)
        y = y.to(device)

        pred = model(x_face, x_body)    # CNN — no SNN state to reset
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        probs     = torch.sigmoid(pred)
        predicted = (probs > 0.5).float()
        correct     += (predicted == y).all(dim=1).sum().item()
        total_preds += y.size(0)
        total_loss  += loss.item()

    scheduler.step()
    train_acc = 100 * correct / max(total_preds, 1)
    avg_loss  = total_loss / len(train_loader)
    cur_lr    = scheduler.get_last_lr()[0]

    val_acc = 0.0
    if len(val_dataset) > 0:
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x_face, x_body = x
                x_face, x_body = x_face.to(device), x_body.to(device)
                y = y.to(device)
                logits    = model(x_face, x_body)
                predicted = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (predicted == y).all(dim=1).sum().item()
                val_total   += y.size(0)
        val_acc = 100 * val_correct / max(val_total, 1)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Loss: {avg_loss:.4f} | Train: {train_acc:.0f}% | Val: {val_acc:.0f}% | LR: {cur_lr:.2e}")

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Best CNN baseline saved (val={val_acc:.0f}%)")

print(f"\n[Baseline CNN] Best Val Accuracy: {best_val_acc:.0f}%")
print(f"Weights saved to: {SAVE_PATH}")

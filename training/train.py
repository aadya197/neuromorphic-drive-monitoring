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
EPOCHS     = 80
BATCH_SIZE = 4
LR         = 3e-4
SAVE_PATH  = "model.pth"


# ── Focal Loss ─────────────────────────────────────────────────────────────────
class FocalLoss(nn.Module):
    """
    Focal Loss: down-weights easy examples so the model focuses on
    hard-to-classify drowsy samples. gamma=2 is standard.
    """
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce   = F.cross_entropy(inputs, targets, weight=self.alpha, reduction="none")
        pt   = torch.exp(-ce)
        loss = ((1 - pt) ** self.gamma) * ce
        return loss.mean()


# ── Dataset ───────────────────────────────────────────────────────────────────
dataset = VideoDataset(DATA_PATH, augment=True)

if len(dataset) == 0:
    print("ERROR: No videos found in", DATA_PATH)
    sys.exit(1)

loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=0, pin_memory=False)

# ── Model ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = DriverMonitoringModel(num_classes=2).to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total_params:,}\n")

# Auto class weights from data distribution
labels   = [s[1] for s in dataset.samples]
n_normal = labels.count(0)
n_drowsy = labels.count(1)
total    = len(labels)
w_normal = total / (2 * max(n_normal, 1))
w_drowsy = total / (2 * max(n_drowsy, 1))
weights  = torch.tensor([w_normal, w_drowsy], dtype=torch.float).to(device)
print(f"Class dist  → Normal: {n_normal} | Drowsy: {n_drowsy}")
print(f"Class weights → Normal: {w_normal:.2f} | Drowsy: {w_drowsy:.2f}\n")

loss_fn   = FocalLoss(alpha=weights, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# ── Training loop ─────────────────────────────────────────────────────────────
best_acc = -1.0

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0.0, 0, 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # CRITICAL: reset SNN neuron states before each forward pass
        functional.reset_net(model)

        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        _, predicted = torch.max(pred, 1)
        total       += y.size(0)
        correct     += int((predicted == y).sum().item())
        total_loss  += loss.item()

    scheduler.step()

    acc     = 100 * correct / total
    avg_lss = total_loss / len(loader)
    cur_lr  = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Loss: {avg_lss:.4f} | Acc: {acc:.0f}% | LR: {cur_lr:.2e}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Best model saved ({acc:.0f}%)")

print(f"\nDone. Best accuracy: {best_acc:.0f}%")
print(f"Model saved to: {SAVE_PATH}")
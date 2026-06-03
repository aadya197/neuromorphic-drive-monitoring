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
EPOCHS     = 120
BATCH_SIZE = 4
LR         = 3e-4
SAVE_PATH  = "model.pth"


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

model = DriverMonitoringModel().to(device)
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Parameters: {total_params:,}\n")

# Auto class weights for independent multi-label binary classes
labels = torch.stack([s[2] for s in dataset.samples])
total  = len(labels)
n_drowsy = labels[:, 0].sum().item()
n_dist   = labels[:, 1].sum().item()
n_glasses = labels[:, 2].sum().item()
n_sunglasses = labels[:, 3].sum().item()
n_night = labels[:, 4].sum().item()

w_drowsy = min(total / max(n_drowsy, 1), 5.0)
w_dist   = min(total / max(n_dist, 1), 5.0)
w_glasses = min(total / max(n_glasses, 1), 5.0)
w_sunglasses = min(total / max(n_sunglasses, 1), 5.0)
w_night = min(total / max(n_night, 1), 5.0)

weights = torch.tensor([w_drowsy, w_dist, w_glasses, w_sunglasses, w_night], dtype=torch.float).to(device)
print(f"Class counts → Drowsy: {int(n_drowsy)} | Dist: {int(n_dist)} | Glasses: {int(n_glasses)} | Sunglasses: {int(n_sunglasses)} | Night: {int(n_night)}")
print(f"Class weights → Drowsy: {w_drowsy:.2f} | Dist: {w_dist:.2f} | Glasses: {w_glasses:.2f} | Sunglasses: {w_sunglasses:.2f} | Night: {w_night:.2f}\n")

loss_fn   = MultiLabelFocalLoss(alpha=weights, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

# ── Training loop ─────────────────────────────────────────────────────────────
best_acc = -1.0

for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total_preds = 0.0, 0, 0

    for x, y in loader:
        x_face, x_body = x
        x_face, x_body = x_face.to(device), x_body.to(device)
        y = y.to(device)

        # CRITICAL: reset SNN neuron states before each forward pass
        functional.reset_net(model)

        pred = model(x_face, x_body)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        probs = torch.sigmoid(pred)
        predicted = (probs > 0.5).float()
        
        # Calculate exact match accuracy (both labels must be correct for the sample)
        matches = (predicted == y).all(dim=1)
        correct += matches.sum().item()
        total_preds += y.size(0)
        total_loss += loss.item()

    scheduler.step()

    acc     = 100 * correct / total_preds
    avg_lss = total_loss / len(loader)
    cur_lr  = scheduler.get_last_lr()[0]

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | "
          f"Loss: {avg_lss:.4f} | Exact Match Acc: {acc:.0f}% | LR: {cur_lr:.2e}")

    if acc > best_acc:
        best_acc = acc
        torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Best model saved ({acc:.0f}%)")

print(f"\nDone. Best Exact Match Accuracy: {best_acc:.0f}%")
print(f"Model saved to: {SAVE_PATH}")

# ── Automatically Run Inference ───────────────────────────────────────────────
print("\nRunning full inference evaluation on the saved model...")
os.system("python3 -m inference.predict")
"""Training script for SNN + BiGRU (no attention) ablation. Saves: model_snn_no_attention.pth"""
import os, sys, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from spikingjelly.activation_based import functional
from models.model_snn_no_attention import SNNBiGRUNoAttention
from dataset.video_dataset import VideoDataset

DATA_PATH         = os.path.expanduser("~/Desktop/dmd")
EPOCHS, BATCH_SIZE, LR = 120, 4, 3e-4
SAVE_PATH         = "model_snn_no_attention.pth"
HELD_OUT_SUBJECTS = {"s1", "s4"}

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__(); self.alpha = alpha; self.gamma = gamma
    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce); loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None: loss = loss * self.alpha
        return loss.mean()

train_ds = VideoDataset(DATA_PATH, augment=True,  split='train', held_out_subjects=HELD_OUT_SUBJECTS)
val_ds   = VideoDataset(DATA_PATH, augment=False, split='test',  held_out_subjects=HELD_OUT_SUBJECTS)
if len(train_ds) == 0: print("ERROR: No training videos found."); sys.exit(1)
print(f"\n[SNN no-Attention] Train: {len(train_ds)} | Val: {len(val_ds)} videos\n")

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = SNNBiGRUNoAttention(num_classes=5).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}\n")

labels_t = torch.stack([s[2] for s in train_ds.samples]); total = len(labels_t)
weights  = torch.tensor([min(total / max(labels_t[:, i].sum().item(), 1), 5.0) for i in range(5)], dtype=torch.float).to(device)
loss_fn  = FocalLoss(alpha=weights, gamma=2.0)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

best_val = -1.0
for epoch in range(EPOCHS):
    model.train(); total_loss, correct, total_preds = 0.0, 0, 0
    for x, y in train_loader:
        xf, xb = x; xf, xb, y = xf.to(device), xb.to(device), y.to(device)
        functional.reset_net(model)
        pred = model(xf, xb); loss = loss_fn(pred, y)
        optimizer.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); optimizer.step()
        probs = torch.sigmoid(pred); predicted = (probs > 0.5).float()
        correct += (predicted == y).all(dim=1).sum().item(); total_preds += y.size(0); total_loss += loss.item()
    scheduler.step(); train_acc = 100 * correct / max(total_preds, 1)

    val_acc = 0.0
    if len(val_ds) > 0:
        model.eval(); vc, vt = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                xf, xb = x; xf, xb, y = xf.to(device), xb.to(device), y.to(device)
                functional.reset_net(model)
                predicted = (torch.sigmoid(model(xf, xb)) > 0.5).float()
                vc += (predicted == y).all(dim=1).sum().item(); vt += y.size(0)
        val_acc = 100 * vc / max(vt, 1)

    print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f} | Train: {train_acc:.0f}% | Val: {val_acc:.0f}%")
    if val_acc > best_val:
        best_val = val_acc; torch.save(model.state_dict(), SAVE_PATH)
        print(f"  ✓ Best saved (val={val_acc:.0f}%)")

print(f"\nBest Val: {best_val:.0f}% | Saved to: {SAVE_PATH}")

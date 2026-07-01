import os
import sys
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spikingjelly.activation_based import functional
from dataset.video_dataset import VideoDataset
from dataset.combined import build_combined_datasets

from models.model_full import DriverMonitoringModel
from models.baseline_cnn_gru import BaselineDriverMonitoringModel
from models.model_snn_no_attention import SNNBiGRUNoAttention
from models.model_snn_only import SNNOnlyNoGRU
from models.model_single_stream import SingleStreamSNN

def parse_args():
    parser = argparse.ArgumentParser(description="Train Driver Monitoring Models")
    parser.add_argument("--model", type=str, default="full",
                        choices=["full", "baseline_cnn", "snn_no_attention", "snn_only", "single_stream"],
                        help="Which model variant to train")
    parser.add_argument("--epochs", type=int, default=120, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    return parser.parse_args()

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        loss = ((1 - pt) ** self.gamma) * bce
        if self.alpha is not None:
            loss = loss * self.alpha
        return loss.mean()

def main():
    args = parse_args()
    
    # Config paths
    # DMD dataset root (may be empty/absent — graceful degradation)
    DATA_PATH       = os.path.expanduser("~/Desktop/dmd")
    DRIVEANDACT_PATH = os.path.expanduser("~/Downloads")
    # 3MDAD dataset root (inside the project)
    MDAD_PATH       = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data", "3mdad"
    )
    HELD_OUT_SUBJECTS = {"s1", "s4"}
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | Model: {args.model}")

    train_dataset, val_dataset, weights = build_combined_datasets(
        dmd_path          = DATA_PATH,
        driveandact_path  = DRIVEANDACT_PATH,
        mdad_path         = MDAD_PATH,
        held_out_subjects = HELD_OUT_SUBJECTS,
        device            = device,
    )

    if len(train_dataset) == 0:
        print("ERROR: No training videos found.")
        sys.exit(1)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False)

    _dmd_train_ds = train_dataset.datasets[0] if hasattr(train_dataset, 'datasets') else train_dataset
    num_subj = max(getattr(_dmd_train_ds, 'num_subjects', 0), 1)

    # Initialize requested model
    if args.model == "full":
        model = DriverMonitoringModel(num_subjects=num_subj)
        SAVE_PATH = "model.pth"
    elif args.model == "baseline_cnn":
        model = BaselineDriverMonitoringModel(num_classes=5)
        SAVE_PATH = "model_baseline_cnn.pth"
    elif args.model == "snn_no_attention":
        model = SNNBiGRUNoAttention(num_classes=5)
        SAVE_PATH = "model_snn_no_attention.pth"
    elif args.model == "snn_only":
        model = SNNOnlyNoGRU(num_classes=5)
        SAVE_PATH = "model_snn_only.pth"
    elif args.model == "single_stream":
        model = SingleStreamSNN(num_classes=5)
        SAVE_PATH = "model_single_stream.pth"

    model.to(device)
    # Set single-step mode explicitly for all SNN layers
    try:
        functional.set_step_mode(model, step_mode='s')
    except Exception:
        pass
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    loss_fn = MultiLabelFocalLoss(alpha=weights, gamma=2.0)
    subject_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_val_acc = -1.0
    
    for epoch in range(args.epochs):
        model.train()
        total_loss, correct, total_preds = 0.0, 0, 0
        subj_correct = 0

        p = float(epoch) / args.epochs
        lam = 2. / (1. + math.exp(-10 * p)) - 1.
        if hasattr(model, 'grl'):
            model.grl.alpha = lam

        for x, y, subj in train_loader:
            x_face, x_body = x
            x_face, x_body = x_face.to(device), x_body.to(device)
            y, subj = y.to(device), subj.to(device)

            # Zero gradients and reset SNN state BEFORE the forward pass
            optimizer.zero_grad()
            functional.reset_net(model)

            if args.model == "single_stream":
                out = model(x_face)
            else:
                out = model(x_face, x_body)

            pred_behavior = out[0] if isinstance(out, tuple) else out
            y_smoothed = y * 0.9 + 0.05
            loss = loss_fn(pred_behavior, y_smoothed)

            valid_mask = subj >= 0
            if isinstance(out, tuple) and len(out) > 1 and valid_mask.any():
                pred_subject = out[1]
                loss_subject = subject_loss_fn(pred_subject[valid_mask], subj[valid_mask])
                loss = loss + loss_subject
                with torch.no_grad():
                    subj_pred = pred_subject.argmax(dim=1)
                    subj_correct += (subj_pred[valid_mask] == subj[valid_mask]).sum().item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            # Detach SNN state after update to prevent graph leakage
            functional.reset_net(model)

            probs = torch.sigmoid(pred_behavior.detach())
            predicted = (probs > 0.5).float()
            correct += (predicted == y).all(dim=1).sum().item()
            total_preds += y.size(0)
            total_loss += loss.item()

        scheduler.step()

        train_acc = 100 * correct / max(total_preds, 1)
        subj_acc = 100 * subj_correct / max(total_preds, 1) if isinstance(out, tuple) else 0
        avg_loss = total_loss / len(train_loader)
        cur_lr = scheduler.get_last_lr()[0]

        val_acc = 0.0
        if len(val_dataset) > 0:
            model.eval()
            val_correct, val_total = 0, 0
            with torch.no_grad():
                for x, y, _ in val_loader:
                    x_face, x_body = x
                    x_face, x_body = x_face.to(device), x_body.to(device)
                    y = y.to(device)
                    
                    # Always reset SNN state before val forward pass
                    try:
                        functional.reset_net(model)
                    except Exception:
                        pass
                        
                    if args.model == "single_stream":
                        out = model(x_face)
                    else:
                        out = model(x_face, x_body)
                        
                    logits = out[0] if isinstance(out, tuple) else out
                    predicted = (torch.sigmoid(logits) > 0.5).float()
                    val_correct += (predicted == y).all(dim=1).sum().item()
                    val_total += y.size(0)
            val_acc = 100 * val_correct / max(val_total, 1)

        print(f"Epoch {epoch+1:02d}/{args.epochs} | Loss: {avg_loss:.4f} | "
              f"Train: {train_acc:.0f}% | Subj: {subj_acc:.0f}% | Val: {val_acc:.0f}% | "
              f"LR: {cur_lr:.2e} | Lam: {lam:.2f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), SAVE_PATH)
            
    print(f"\nDone. Best Val Exact Match: {best_val_acc:.0f}%")
    print(f"Model saved to: {SAVE_PATH}")
    
    if args.model == "full":
        print("\nRunning final evaluation on held-out test set...")
        os.system("python3 -m inference.predict")

if __name__ == "__main__":
    main()
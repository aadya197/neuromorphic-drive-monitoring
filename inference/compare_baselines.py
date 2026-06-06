"""
Ablation comparison: Dual-Stream SNN vs Single-Stream SNN vs CNN Baseline.

Produces a 3-way paper-ready table answering two ablation questions:
  Q1. Does the SNN backbone outperform a standard CNN? (Dual-SNN vs CNN)
  Q2. Does the body camera contribute? (Dual-SNN vs Single-SNN)

Usage:
    python3 -m inference.compare_baselines
"""
import os
import sys
import torch
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, roc_auc_score
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_full import DriverMonitoringModel
from models.model_single_stream import SingleStreamSNN
from models.model_snn_no_attention import SNNBiGRUNoAttention
from models.model_snn_only import SNNOnlyNoGRU
from models.baseline_cnn_gru import BaselineDriverMonitoringModel
from dataset.video_dataset import VideoDataset
from spikingjelly.activation_based import functional

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH         = os.path.expanduser("~/Desktop/dmd")
HELD_OUT_SUBJECTS = {"s1", "s4"}
CLASS_NAMES       = ["Drowsy", "Distracted", "Glasses", "Sunglasses", "Night"]
THRESHOLD         = 0.5

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def evaluate_model(model, dataset, is_snn=True, single_stream=False):
    """Run inference, return (all_targets, all_probs)."""
    model.eval()
    all_targets, all_probs = [], []

    for i in range(len(dataset)):
        x_face, x_body = dataset[i][0]
        y = dataset[i][1]

        x_face = x_face.unsqueeze(0).to(device)
        x_body = x_body.unsqueeze(0).to(device)

        with torch.no_grad():
            if is_snn:
                functional.reset_net(model)
            if single_stream:
                logits = model(x_face)        # body ignored
            else:
                logits = model(x_face, x_body)
            probs = torch.sigmoid(logits)[0].cpu().numpy()

        all_targets.append(y.numpy())
        all_probs.append(probs)

        if (i + 1) % 5 == 0 or (i + 1) == len(dataset):
            print(f"  {i+1}/{len(dataset)}", end="\r")

    print()
    return np.array(all_targets), np.array(all_probs)


def per_class_metrics(targets, probs):
    """Return list of (prec, rec, f1, auc) per class."""
    preds = (probs > THRESHOLD).astype(int)
    metrics = []
    for i in range(len(CLASS_NAMES)):
        p = precision_score(targets[:, i], preds[:, i], zero_division=0)
        r = recall_score   (targets[:, i], preds[:, i], zero_division=0)
        f = f1_score       (targets[:, i], preds[:, i], zero_division=0)
        try:
            a = roc_auc_score(targets[:, i], probs[:, i])
        except Exception:
            a = float('nan')
        metrics.append((p, r, f, a))
    return metrics


def print_3way_table(results):
    """
    results: list of (name, targets, probs) tuples
    Prints a multi-column ablation table.
    """
    col_w = 32
    sep = "=" * (16 + col_w * len(results))

    print(f"\n{sep}")
    print(f"  ABLATION STUDY -- {len(results)}-Way Comparison")
    print(f"  Test subjects (held-out): {HELD_OUT_SUBJECTS}")
    print(sep)

    # Header row
    header = f"{'Class':<14} "
    for name, _, _ in results:
        header += f"| {name:^{col_w-2}} "
    print(header)

    subheader = f"{'':14} "
    for _ in results:
        subheader += f"| {'Prec':>5} {'Rec':>5} {'F1':>5} {'AUC':>5}  "
    print(subheader)
    print("-" * len(sep))

    # Per-class rows
    all_metrics = [(name, per_class_metrics(t, p)) for name, t, p in results]

    for ci, cls in enumerate(CLASS_NAMES):
        row = f"{cls:<14} "
        for name, metrics in all_metrics:
            p_val, r_val, f_val, a_val = metrics[ci]
            auc_str = f"{a_val:5.3f}" if not np.isnan(a_val) else "  nan"
            row += f"| {p_val:5.3f} {r_val:5.3f} {f_val:5.3f} {auc_str}  "
        print(row)

    print("-" * len(sep))

    # Macro avg + exact match rows
    macro_row = f"{'Macro Avg':<14} "
    exact_row = f"{'Exact Match':<14} "
    for name, targets, probs in results:
        preds = (probs > THRESHOLD).astype(int)
        mp = precision_score(targets, preds, average='macro', zero_division=0)
        mr = recall_score   (targets, preds, average='macro', zero_division=0)
        mf = f1_score       (targets, preds, average='macro', zero_division=0)
        exact = 100 * ((preds == targets).all(axis=1)).mean()
        macro_row += f"| {mp:5.3f} {mr:5.3f} {mf:5.3f} {'--':>5}  "
        exact_row += f"|  {exact:5.1f}%{' ':>{col_w-10}} "
    print(macro_row)
    print(exact_row)
    print(sep + "\n")
    print("Note: Drowsy AUC/metrics may show nan if no drowsy samples exist in test subjects.")
    print("      This reflects test set composition, not model capability.\n")


def load_model(model_class, path, label, **kwargs):
    if not os.path.exists(path):
        print(f"WARNING: {path} not found -- skipping {label}.")
        return None
    m = model_class(**kwargs)
    m.load_state_dict(torch.load(path, map_location=device))
    m.to(device)
    params = sum(p.numel() for p in m.parameters() if p.requires_grad)
    print(f"{label}: {params:,} parameters")
    return m


def main():
    # ── Load test dataset ─────────────────────────────────────────────────────
    test_dataset = VideoDataset(DATA_PATH, augment=False,
                                split='test', held_out_subjects=HELD_OUT_SUBJECTS)
    if len(test_dataset) == 0:
        print(f"ERROR: No test videos found for subjects {HELD_OUT_SUBJECTS}.")
        sys.exit(1)
    print(f"\nTest set: {len(test_dataset)} videos | Subjects: {HELD_OUT_SUBJECTS}\n")

    # ── Load all 5 model variants ─────────────────────────────────────────────
    # Q1: Backbone ablation — does SNN beat CNN?
    snn_full   = load_model(DriverMonitoringModel,    "model.pth",                 "Full SNN+GRU+Attn",  num_classes=5)
    cnn_model  = load_model(BaselineDriverMonitoringModel, "model_baseline_cnn.pth", "CNN+GRU+Attn",     num_classes=5)

    # Q2: Component ablation — does each SNN component contribute?
    no_attn    = load_model(SNNBiGRUNoAttention,      "model_snn_no_attention.pth","SNN+GRU (no Attn)", num_classes=5)
    snn_only   = load_model(SNNOnlyNoGRU,             "model_snn_only.pth",        "SNN only (no GRU)", num_classes=5)

    # Q3: Stream ablation — does the body camera help?
    single_s   = load_model(SingleStreamSNN,          "model_single_stream.pth",   "SNN Face-only",     num_classes=5)
    print()

    results = []

    if snn_full:
        print("Evaluating Full SNN + GRU + Attention (Ours)...")
        t, p = evaluate_model(snn_full, test_dataset, is_snn=True, single_stream=False)
        results.append(("SNN+GRU+Attn (Ours)", t, p))

    if no_attn:
        print("Evaluating SNN + GRU (no Attention)...")
        t, p = evaluate_model(no_attn, test_dataset, is_snn=True, single_stream=False)
        results.append(("SNN+GRU (no Attn)", t, p))

    if snn_only:
        print("Evaluating SNN Only (no GRU, no Attention)...")
        t, p = evaluate_model(snn_only, test_dataset, is_snn=True, single_stream=False)
        results.append(("SNN only (no GRU)", t, p))

    if single_s:
        print("Evaluating Single-Stream SNN (face only)...")
        t, p = evaluate_model(single_s, test_dataset, is_snn=True, single_stream=True)
        results.append(("SNN Face-only", t, p))

    if cnn_model:
        print("Evaluating CNN + GRU + Attention Baseline...")
        t, p = evaluate_model(cnn_model, test_dataset, is_snn=False, single_stream=False)
        results.append(("CNN+GRU+Attn (Base)", t, p))

    if results:
        print_3way_table(results)
    else:
        print("No models available to compare. Train at least one model first.")


if __name__ == "__main__":
    main()


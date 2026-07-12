import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, multilabel_confusion_matrix

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.model_full import DriverMonitoringModel
from dataset.video_dataset import VideoDataset
from dataset.combined import build_combined_datasets
from spikingjelly.activation_based import functional

def main():
    print("="*65)
    print("  Generating Research Paper Graphs")
    print("="*65)
    
    DATA_PATH        = os.path.expanduser("~/Downloads")
    # Set to your Drive&Act root to include it in graphs, or None to skip.
    DRIVEANDACT_PATH = os.path.expanduser("~/Downloads")  # folders: inner_mirror, a_column_driver, activities_3s
    MDAD_PATH        = os.path.expanduser("~/Downloads/neuromorphic-drive-monitoring/data/3mdad")

    if not os.path.exists("model.pth"):
        print("ERROR: model.pth not found. Please train the model first.")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    state_dict = torch.load("model.pth", map_location=device)
    num_subjects = state_dict.get("subject_classifier.2.weight", torch.zeros(1, 1)).shape[0]
    model = DriverMonitoringModel(num_subjects=max(num_subjects, 1))
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load FULL combined dataset (no split) for graph generation
    dataset, _, _ = build_combined_datasets(
        dmd_path          = DATA_PATH,
        driveandact_path  = DRIVEANDACT_PATH,
        mdad_path         = MDAD_PATH,
        held_out_subjects = set(),  # no holdout — use all data
        augment_train     = False,
    )
    
    if len(dataset) == 0:
        print("No videos found in dataset. Cannot generate graphs.")
        sys.exit(1)

    # 1. Run Evaluation
    print("\nRunning full evaluation loop...")
    all_targets = []
    all_probs = []

    for i in range(len(dataset)):
        # dataset[i] is a 3-tuple: (x, label, subject_idx)
        x_face, x_body = dataset[i][0]
        y = dataset[i][1]
        
        x_face = x_face.unsqueeze(0).to(device)
        x_body = x_body.unsqueeze(0).to(device)

        with torch.no_grad():
            functional.reset_net(model)
            out = model(x_face, x_body)
            # DriverMonitoringModel returns (behavior_logits, subject_logits)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits)[0]
            
        all_targets.append(y.numpy())
        all_probs.append(probs.cpu().numpy())
        
        if (i+1) % 10 == 0 or (i+1) == len(dataset):
            print(f"Processed {i+1}/{len(dataset)} videos")

    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    class_names = ["Drowsy", "Distracted", "Glasses", "Sunglasses", "Night"]
    colors = ["#e63946", "#f4a261", "#2a9d8f", "#264653", "#8338ec"]
    
    # Setup styling
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_context("paper", font_scale=1.4)
    os.makedirs("graphs", exist_ok=True)
    
    print("\nGenerating Class Distribution...")
    # 2. Class Distribution
    plt.figure(figsize=(8, 6))
    counts = all_targets.sum(axis=0)
    sns.barplot(x=class_names, y=counts, palette=colors)
    plt.title("Dataset Class Distribution (Positive Samples)", fontsize=16, fontweight='bold')
    plt.ylabel("Number of Samples")
    plt.tight_layout()
    plt.savefig("graphs/class_distribution.png", dpi=300)
    plt.close()

    print("Generating ROC Curves...")
    # 3. ROC Curves
    plt.figure(figsize=(8, 8))
    for i in range(5):
        if counts[i] > 0:
            fpr, tpr, _ = roc_curve(all_targets[:, i], all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, color=colors[i], lw=2.5, label=f"{class_names[i]} (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontweight='bold')
    plt.ylabel('True Positive Rate', fontweight='bold')
    plt.title('Receiver Operating Characteristic (ROC)', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("graphs/roc_curve.png", dpi=300)
    plt.close()

    print("Generating Precision-Recall Curves...")
    # 4. Precision-Recall Curves
    plt.figure(figsize=(8, 8))
    for i in range(5):
        if counts[i] > 0:
            precision, recall, _ = precision_recall_curve(all_targets[:, i], all_probs[:, i])
            pr_auc = auc(recall, precision)
            plt.plot(recall, precision, color=colors[i], lw=2.5, label=f"{class_names[i]} (AUC = {pr_auc:.2f})")
            
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontweight='bold')
    plt.ylabel('Precision', fontweight='bold')
    plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig("graphs/pr_curve.png", dpi=300)
    plt.close()

    print("Generating High-Res Confusion Matrices...")
    # 5. Publication-Ready Confusion Matrices
    all_preds = (all_probs > 0.5).astype(int)
    mcm = multilabel_confusion_matrix(all_targets, all_preds)
    
    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for i, matrix in enumerate(mcm):
        sns.heatmap(matrix, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                    xticklabels=['Negative', 'Positive'],
                    yticklabels=['Negative', 'Positive'],
                    annot_kws={"size": 16}, cbar=False)
        axes[i].set_title(class_names[i], fontsize=16, fontweight='bold', pad=15)
        axes[i].set_ylabel('True Label', fontweight='bold')
        axes[i].set_xlabel('Predicted Label', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig("graphs/confusion_matrices.png", dpi=300)
    plt.close()
    
    print("\n[SUCCESS] All graphs generated in the 'graphs/' directory!")

if __name__ == "__main__":
    main()

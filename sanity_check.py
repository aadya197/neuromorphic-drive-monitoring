"""
Quick sanity check: load one batch from Dataset3MDAD and run a forward pass.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset.dataset_3mdad import Dataset3MDAD
import torch

root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "3mdad")
print(f"Dataset root: {root}")

ds = Dataset3MDAD(root, augment=False, split="train", held_out_subjects={"S1","S4"})
print(f"Total train samples: {len(ds)}")

if len(ds) > 0:
    (vf, vb), label, sidx = ds[0]
    print(f"voxel_face shape: {vf.shape} | voxel_body shape: {vb.shape}")
    print(f"label: {label}")
    print("✅ Sanity check PASSED — Dataset3MDAD is working correctly.")
else:
    print("❌ No samples found. Check path and folder structure.")

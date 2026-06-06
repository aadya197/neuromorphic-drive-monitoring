"""
Generates a small synthetic dataset for unit-testing the training pipeline
without requiring the full DMD dataset download.

NOTE: This is NOT a substitute for real training data. Labels are random and
the event distributions do not reflect real neuromorphic camera data. Use only
for integration testing (e.g., confirming model shapes and training loop logic).
"""
import os
import numpy as np

samples = []

for i in range(50):
    # Synthetic event array: N events x 4 columns [x, y, t, polarity]
    events = np.zeros((500, 4), dtype=np.float32)
    events[:, 0] = np.random.randint(0, 64, 500)    # x coords (64x64 grid)
    events[:, 1] = np.random.randint(0, 64, 500)    # y coords
    events[:, 2] = np.linspace(0, 1, 500)            # normalised timestamp
    events[:, 3] = np.random.choice([-1, 1], 500)   # polarity

    # 5-dimensional multi-label float tensor: [Drowsy, Distracted, Glasses, Sunglasses, Night]
    # Each value is independently 0.0 (absent) or 1.0 (present) — matches model output space.
    label = np.random.randint(0, 2, size=5).astype(np.float32)

    samples.append({
        "events": events,
        "label": label,   # shape (5,) float32 — compatible with BCEWithLogitsLoss
    })

os.makedirs("data", exist_ok=True)
np.save("data/train_dummy.npy", samples, allow_pickle=True)
print(f"Dummy dataset created: {len(samples)} samples at data/train_dummy.npy")
print(f"Label shape: {samples[0]['label'].shape} | dtype: {samples[0]['label'].dtype}")
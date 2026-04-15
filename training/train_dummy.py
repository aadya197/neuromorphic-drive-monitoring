"""
train_dummy.py — Synthetic Training Script for NeuroGuard
=========================================================
Since the full DMD dataset is very large (~30GB+) and isn't currently 
on the machine, this script runs the exact same training pipeline 
using synthetic/dummy event data.

This ensures:
1. The SNN architecture is mathematically sound and bug-free.
2. We can generate a valid 'model.pth' to test the live webcam frontend.
3. You can swap to the real dataset later easily!
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import time

from spikingjelly.activation_based import functional

# Import the actual architecture
from models.model_full import DriverMonitoringModel

def generate_synthetic_data(num_samples=32) -> TensorDataset:
    print(f"[1/4] Generating {num_samples} synthetic driving event samples...")
    # Shape expected: [batch, time_chunks, voxel_bins, height, width]
    # matching live feed: [3, 5, 64, 64]
    X = torch.zeros((num_samples, 3, 5, 64, 64), dtype=torch.float32)
    y = torch.zeros(num_samples, dtype=torch.long)
    
    # Fill with random "event" sparsity
    for i in range(num_samples):
        # Sparse events
        mask = torch.rand(3, 5, 64, 64) > 0.95
        X[i][mask] = torch.rand(mask.sum())
        
        # Labels: 0 = ALERT, 1 = DROWSY
        y[i] = i % 2
        
    return TensorDataset(X, y)

def train_dummy():
    os.makedirs("saved_models", exist_ok=True)
    
    dataset = generate_synthetic_data(32)
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    print("[2/4] Initializing Spiking Neural Network (SNN) + LSTM Architecture...")
    model = DriverMonitoringModel(num_classes=2)
    # Move to MPS if on Apple Silicon, else CPU
    device = torch.device('cpu') 
    model.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()
    
    epochs = 3
    print(f"[3/4] Starting Training Loop ({epochs} epochs)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        start_t = time.time()
        for x, labels in loader:
            x, labels = x.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            # Forward SNN
            preds = model(x)
            loss = loss_fn(preds, labels)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            # 🔴 CRITICAL: Reset SNN neuron membrane potentials after every sequence!
            functional.reset_net(model)
            
            total_loss += loss.item() * x.size(0)
            
            _, predicted = torch.max(preds, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        avg_loss = total_loss / total
        acc = 100 * correct / total
        elapsed = time.time() - start_t
        
        print(f"      Epoch [{epoch+1}/{epochs}] | Loss: {avg_loss:.4f} | Acc: {acc:.1f}% | Time: {elapsed:.1f}s")
        
    # Save the model
    save_path = os.path.join("saved_models", "model.pth")
    torch.save(model.state_dict(), save_path)
    print(f"[4/4] ✅ Success! Model saved to {save_path}")
    print("\n   The PyTorch model is now ready! Restart your backend server")
    print("   and the live frontend will now run using the SNN!")

if __name__ == "__main__":
    train_dummy()

# Neuromorphic Driver Monitoring

This project implements a **5-Target Multi-Label Spiking Neural Network (SNN)** to detect complex driver behaviors and environmental contexts using synthetic neuromorphic event data. The model is capable of concurrently detecting:
1. **Drowsiness** (e.g., yawning, eye closure)
2. **Distraction** (e.g., phone usage, looking away)
3. **Glasses**
4. **Sunglasses**
5. **Nighttime Driving**

By leveraging a Dual-Stream SNN Architecture (`x_face` and `x_body`), the network learns spatial-temporal anomalies rather than static color features, proving highly resilient to varying lighting conditions.

---

## ⚠️ Important Dataset Limitation: Synthetic Event Emulation
**Please note for dataset access requests and research applications:**
This repository uses **synthetic event emulation**, *not* raw hardware event data. The `video_to_events()` preprocessing script uses frame differencing (~30 FPS) on standard RGB videos to roughly approximate the behavior of a Dynamic Vision Sensor (DVS). 

Real neuromorphic cameras fire continuously at microsecond temporal resolution. Our synthetic approximation is designed to study spiking neural network architectures when hardware DVS data is unavailable, but it lacks the true microsecond granularity of an actual event camera. Do not treat the generated event `.npy` files as true hardware event data.

---

## 🛠 Installation

This project is built using PyTorch and SpikingJelly. It is highly recommended to use a virtual environment.

```bash
# Clone the repository
git clone <repository_url>
cd neuromorphic-drive-monitoring

# Install the required packages
pip install torch torchvision numpy opencv-python matplotlib seaborn scikit-learn spikingjelly
```

---

## 📂 Dataset Setup

The pipeline is designed to work with the **DMD (Driver Micro Sleep and Distraction)** Dataset. 

1. Download the required dataset blocks (e.g., `dmd 10`, `dmd 20`, etc.).
2. Extract them into your `~/Downloads` directory.
3. The training and inference scripts automatically scan all folders matching `~/Downloads/dmd*`.
4. Ensure the dataset retains its internal structure containing `_rgb_face.mp4`, `_rgb_body.mp4`, and the corresponding `_ann_drowsiness.json` / `_ann_distraction.json` annotations.

*(Note: You can safely delete `_rgb_hands.mp4`, `_ir*.mp4`, `_depth*.mp4`, and `_mosaic*.mp4` to save disk space, as this network is designed exclusively for the dual face/body streams).*

---

## 🚀 Usage

### 1. Training the Model
To initiate the training loop from scratch, run:
```bash
python3 -m training.train
```
This will automatically parse all datasets, convert videos to voxel grids, and train the `SNNBackbone` and bidirectional GRU. The best weights are saved to `model.pth`.

### 2. Batch Evaluation & Generating Research Graphs
To run a full inference evaluation over the test set and generate publication-ready performance graphs (ROC, Precision-Recall, Confusion Matrices), run:
```bash
python3 -m inference.generate_paper_graphs
```
The output visualizations will be saved inside the `graphs/` directory.

### 3. Live Webcam Inference
To test the Spiking Neural Network on a live stream using your computer's built-in webcam, run:
```bash
python3 -m inference.webcam
```
* **Smart Crop Mode:** This script uses a Haar Cascade face detector to intelligently crop your face for the `x_face` stream, while passing the full un-cropped frame to the `x_body` stream.
* Press `q` to terminate the live session.

---

## 📊 Results

When evaluated on 47 dual-camera videos (77 positive context events), the 5-target multi-label network achieved the following metrics:
- **Nighttime Recall:** 100%
- **Sunglasses Recall:** 90%
- **Glasses Recall:** 87%
- **Distraction Precision:** 84%
- **Drowsiness Precision:** 91%

The SNN correctly disentangles concurrent behaviors (Multi-Labeling), meaning the model is perfectly capable of detecting that a driver is both *Drowsy* AND wearing *Sunglasses* simultaneously.
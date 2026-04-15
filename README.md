# ⬡ NeuroGuard — Neuromorphic Driver Monitoring System

> Real-time driver drowsiness & distraction detection using **Event Cameras** + **Spiking Neural Networks**

---

## 📁 Project Structure

```
neuromorphic-drive-monitoring/
├── backend/                   ← Flask API server
│   ├── app.py                 ← REST endpoints (/api/predict, /api/status)
│   ├── detector.py            ← Core engine: frames→events→voxel→SNN
│   ├── requirements.txt       ← Backend dependencies
│   └── __init__.py
│
├── frontend/                  ← Web UI
│   ├── index.html             ← Main landing page
│   ├── live.html              ← Live webcam detection page
│   ├── style.css              ← Global design system
│   ├── live.css               ← Live page styles
│   ├── app.js                 ← Landing page animations
│   └── live.js                ← Webcam + API + real-time rendering
│
├── models/                    ← PyTorch model definitions
│   ├── model_full.py          ← DriverMonitoringModel (SNN + LSTM + Attention)
│   ├── snn_backbone.py        ← Spiking CNN backbone (LIF neurons)
│   ├── attention_module.py    ← Temporal soft-attention
│   └── __init__.py
│
├── training/                  ← Training pipeline
│   ├── train.py               ← 20-epoch training loop
│   └── __init__.py
│
├── inference/                 ← Standalone inference
│   └── predict.py             ← Single-clip prediction script
│
├── dataset/                   ← Data handling
│   ├── video_dataset.py       ← VideoDataset (loads .mp4/.avi)
│   ├── event_dataset.py       ← EventDataset (loads .npy events)
│   ├── event_from_video.py    ← Simulate events from video frames
│   ├── preprocessing.py       ← events_to_voxel() converter
│   └── __init__.py
│
├── utils/                     ← Utilities
│   ├── generate_dummy_data.py
│   ├── precompute_events.py
│   ├── visualization.py
│   └── __init__.py
│
├── saved_models/              ← Trained weights go here
│   └── model.pth              ← (put your trained model here)
│
├── requirements.txt           ← Full project dependencies
├── run_backend.sh             ← One-click backend starter
└── README.md
```

---

## 🧠 Model Architecture

```
Webcam Frame
     ↓
Frame Differencing → Simulated Events (x, y, t, p)
     ↓
Voxel Grid Encoding  [5 bins × 64 × 64]
     ↓
Spiking CNN Backbone  (Conv→BN→LIF→Pool ×2)
     ↓
LSTM  (input=8192, hidden=256)
     ↓
Temporal Attention  (Linear→Softmax weighted sum)
     ↓
Classifier  (Linear 256→2)
     ↓
ALERT 🙂  /  DROWSY 😴
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd neuromorphic-drive-monitoring
pip install -r requirements.txt
```

### 2. Train the Model (if not already trained)

```bash
# Put DMD dataset at ~/Desktop/dmd/
# Pre-compute events
python -m utils.precompute_events

# Train for 20 epochs
python -m training.train
```

Model saves to `model.pth` → move it:
```bash
mv model.pth saved_models/model.pth
```

### 3. Start the Backend API

```bash
bash run_backend.sh
# OR manually:
python -m backend.app
```

API runs at: `http://localhost:5050`

### 4. Open the Frontend

Open in your browser (no server needed):
```
frontend/index.html      ← Landing page
frontend/live.html       ← Live webcam detection
```

---

## 🌐 API Reference

| Endpoint | Method | Description |
|---|---|---|
| `/api/status` | GET | Health check + model info |
| `/api/predict` | POST | Send base64 frame, get detection result |
| `/api/reset` | POST | Clear frame buffer & session stats |

### POST `/api/predict`

**Request:**
```json
{ "frame": "data:image/jpeg;base64,..." }
```

**Response:**
```json
{
  "state":       "ALERT",
  "label":       "DRIVER ALERT 🙂",
  "confidence":  97.3,
  "perclos":     8.2,
  "blink_rate":  18.0,
  "head_droop":  2.0,
  "face_det":    true,
  "face_bbox":   [x, y, w, h],
  "mode":        "SNN",
  "model_loaded": true,
  "alert":       false
}
```

---

## 🔬 Technologies

| Component | Technology |
|---|---|
| Spiking Neural Network | PyTorch + SpikingJelly |
| Temporal Modeling | LSTM + Soft Attention |
| Event Simulation | OpenCV frame differencing |
| Face Detection (fallback) | Haar Cascades (OpenCV) |
| Backend API | Flask + Flask-CORS |
| Frontend | Vanilla HTML + CSS + JS |
| Dataset | DMD (Vicomtech) |

---

## 📊 Detection States

| State | Trigger | Action |
|---|---|---|
| 🟢 **ALERT** | Eyes open, PERCLOS < 20% | Normal |
| 🟡 **TIRED** | PERCLOS 20–60% or head droop | Warning log |
| 🔴 **DROWSY** | PERCLOS > 60%, eyes closing | Alarm banner |
| 🔍 **NO FACE** | No face detected in frame | Prompt reposition |

---

## 📄 License

MIT License — built for research and educational purposes.
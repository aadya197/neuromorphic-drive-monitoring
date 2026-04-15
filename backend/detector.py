"""
detector.py — Core Detection Engine for NeuroGuard
========================================================
Converts live webcam frames into simulated neuromorphic events,
passes them through the trained SNN model, and returns predictions.

Falls back to OpenCV + heuristic-based detection if model not loaded.
Torch and SpikingJelly are optional — system runs without them.
"""

import os
import sys
import time
import base64
import logging
import numpy as np
import cv2
from collections import deque

# Add parent directory to path so we can import the model
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

# Try importing torch (optional)
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("NeuroGuard")

# ───────────────────────────────────────────
# Constants
# ───────────────────────────────────────────
MODEL_PATH   = os.path.join(ROOT, "saved_models", "model.pth")
ALT_MODEL    = os.path.join(ROOT, "model.pth")   # also check project root
VOXEL_BINS   = 5
VOXEL_H      = 64
VOXEL_W      = 64
NUM_CLASSES  = 2
FRAME_BUFFER = 9   # Number of frames to collect before inference
EVENT_THRESH = 10  # Pixel difference threshold for event generation

# ───────────────────────────────────────────
# Class labels (extend if model has more classes)
# ───────────────────────────────────────────
LABELS = {
    0: "ALERT",
    1: "DROWSY",
}

# ───────────────────────────────────────────
# OpenCV Haar cascades (absolute paths)
# ───────────────────────────────────────────
CV2_DATA = cv2.data.haarcascades
FACE_CASCADE = cv2.CascadeClassifier(os.path.join(CV2_DATA, "haarcascade_frontalface_default.xml"))
EYE_CASCADE  = cv2.CascadeClassifier(os.path.join(CV2_DATA, "haarcascade_eye.xml"))


# ───────────────────────────────────────────
# Helper: frames → simulated events → voxel
# ───────────────────────────────────────────
def frames_to_events(frames: list, threshold: int = EVENT_THRESH) -> np.ndarray:
    """Convert a list of grayscale frames into a list of (x,y,t,p) events."""
    events = []
    for t, (prev, curr) in enumerate(zip(frames[:-1], frames[1:])):
        diff = curr.astype(np.int16) - prev.astype(np.int16)
        ys_p, xs_p = np.where(diff > threshold)
        ys_n, xs_n = np.where(diff < -threshold)
        for x, y in zip(xs_p, ys_p):
            events.append([int(x), int(y), t, 1])
        for x, y in zip(xs_n, ys_n):
            events.append([int(x), int(y), t, -1])
    return np.array(events, dtype=np.int16) if events else np.zeros((1, 4), dtype=np.int16)


def events_to_voxel(events: np.ndarray,
                    num_bins: int = VOXEL_BINS,
                    height: int = VOXEL_H,
                    width: int = VOXEL_W):
    """Convert event array to voxel grid tensor [num_bins, H, W]."""
    voxel = np.zeros((num_bins, height, width), dtype=np.float32)
    if len(events) < 4:
        if TORCH_AVAILABLE:
            return torch.tensor(voxel)
        return voxel

    timestamps = events[:, 2].astype(float)
    t_min, t_max = timestamps.min(), timestamps.max()
    delta_t = (t_max - t_min) if (t_max - t_min) != 0 else 1.0

    x_max = float(events[:, 0].max()) or 1.0
    y_max = float(events[:, 1].max()) or 1.0

    for x, y, t, p in events:
        xs = int((x / x_max) * (width - 1))
        ys = int((y / y_max) * (height - 1))
        bi = int(((t - t_min) / delta_t) * (num_bins - 1))
        bi = min(bi, num_bins - 1)
        voxel[bi, ys, xs] += float(p)

    max_v = voxel.max()
    if max_v != 0:
        voxel /= max_v
    if TORCH_AVAILABLE:
        return torch.tensor(voxel, dtype=torch.float32)
    return voxel


# ───────────────────────────────────────────
# OpenCV heuristic eye-closure detector
# ───────────────────────────────────────────
def heuristic_detect(frame_bgr: np.ndarray, prev_gray=None) -> dict:
    """
    Fallback detector using Haar cascades.
    Returns a dict with state info.
    """
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    faces = FACE_CASCADE.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    result = {
        "state": "NO_FACE",
        "label": "No Face Detected",
        "confidence": 0.0,
        "perclos": 0.0,
        "eyes_open": 0,
        "face_detected": False,
        "head_droop": 0.0,
        "bbox": None,
    }

    if len(faces) == 0:
        return result

    # Use largest face
    fx, fy, fw, fh = max(faces, key=lambda r: r[2] * r[3])
    result["face_detected"] = True
    result["bbox"] = [int(fx), int(fy), int(fw), int(fh)]

    # Look for eyes within face region
    roi = gray[fy:fy + fh, fx:fx + fw]
    upper_roi = roi[:fh // 2, :]  # Eyes are in upper half
    eyes = EYE_CASCADE.detectMultiScale(upper_roi, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
    eyes_found = len(eyes)
    result["eyes_open"] = eyes_found

    # PERCLOS approximation: fewer eye detections → more closed
    perclos = max(0.0, 1.0 - min(eyes_found / 2.0, 1.0))

    # Head droop: if face centroid moves down significantly
    face_center_y = fy + fh // 2
    head_droop_pct = max(0.0, (face_center_y / h) - 0.5) * 2.0  # 0–1

    result["perclos"]    = round(perclos * 100, 1)
    result["head_droop"] = round(head_droop_pct * 45, 1)  # degrees approx

    # Classify
    if perclos > 0.7:
        result["state"]      = "DROWSY"
        result["label"]      = "DROWSY 😴"
        result["confidence"] = round(75 + perclos * 20, 1)
    elif perclos > 0.3 or head_droop_pct > 0.2:
        result["state"]      = "TIRED"
        result["label"]      = "FATIGUE DETECTED 😪"
        result["confidence"] = round(60 + perclos * 30, 1)
    else:
        result["state"]      = "ALERT"
        result["label"]      = "DRIVER ALERT 🙂"
        result["confidence"] = round(80 + (1 - perclos) * 18, 1)

    return result


# ───────────────────────────────────────────
# Main Detector Class
# ───────────────────────────────────────────
class NeuroGuardDetector:
    def __init__(self):
        self.model = None
        self.device = None  # set in _load_model if torch available
        self.frame_buffer = deque(maxlen=FRAME_BUFFER + 1)
        self.prev_gray = None
        self.model_loaded = False

        # Blink tracking
        self.blink_count = 0
        self.blink_window_start = time.time()
        self.last_eye_state = "open"
        self.perclos_history = deque(maxlen=30)

        self._load_model()

    def _load_model(self):
        """Try to load the trained SNN model."""
        if not TORCH_AVAILABLE:
            log.warning("⚠️  PyTorch not installed. Using heuristic mode (OpenCV face detection).")
            log.info("   Install PyTorch to enable SNN mode: pip install torch")
            return

        path = MODEL_PATH if os.path.exists(MODEL_PATH) else (ALT_MODEL if os.path.exists(ALT_MODEL) else None)

        if path is None:
            log.warning("⚠️  No model.pth found. Using heuristic mode (OpenCV face detection).")
            log.info(f"   To enable SNN mode, save your model to: {MODEL_PATH}")
            return

        try:
            from models.model_full import DriverMonitoringModel
            self.model = DriverMonitoringModel(num_classes=NUM_CLASSES)
            self.device = torch.device("cpu")
            state_dict = torch.load(path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            self.model_loaded = True
            log.info(f"✅  SNN Model loaded from: {path}")
        except Exception as e:
            log.error(f"❌  Failed to load model: {e}")
            log.warning("   Falling back to heuristic detection.")
            self.model = None

    def _run_snn_inference(self, gray_frames: list) -> dict:
        """Run frames through the full SNN pipeline."""
        if not TORCH_AVAILABLE or self.model is None:
            return None
        try:
            from spikingjelly.activation_based import functional
            events = frames_to_events(gray_frames)
            if len(events) < 10:
                return None

            # Build 3 temporal chunks → [3, 5, H, W]
            chunk_size = max(1, len(events) // 3)
            voxels = []
            for i in range(3):
                chunk = events[i * chunk_size: (i + 1) * chunk_size]
                v = events_to_voxel(chunk)
                voxels.append(v)

            x = torch.stack(voxels).unsqueeze(0)  # [1, 3, 5, 64, 64]

            with torch.no_grad():
                out = self.model(x)
                functional.reset_net(self.model)
                probs = F.softmax(out, dim=1).squeeze().tolist()

            pred_cls = int(np.argmax(probs))
            confidence = round(probs[pred_cls] * 100, 1)
            label = LABELS.get(pred_cls, "UNKNOWN")

            return {
                "state":      label,
                "label":      label,
                "confidence": confidence,
                "probs":      probs,
                "model_mode": "SNN",
            }
        except Exception as e:
            log.error(f"SNN inference error: {e}")
            return None

    def process_frame(self, frame_bgr: np.ndarray) -> dict:
        """
        Process a single BGR webcam frame and return detection result.
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        gray_resized = cv2.resize(gray, (VOXEL_W * 4, VOXEL_H * 4))  # 256×256 for event gen

        # Always run heuristic for face/eye metrics
        heuristic = heuristic_detect(frame_bgr, self.prev_gray)
        self.prev_gray = gray

        # Track PERCLOS
        eyes_open_ratio = heuristic["eyes_open"] / 2.0
        self.perclos_history.append(1.0 - min(eyes_open_ratio, 1.0))
        perclos_avg = round(np.mean(self.perclos_history) * 100, 1) if self.perclos_history else 0.0

        # Blink rate (min per minute)
        now = time.time()
        if self.last_eye_state == "open" and heuristic["eyes_open"] == 0:
            self.blink_count += 1
        self.last_eye_state = "open" if heuristic["eyes_open"] > 0 else "closed"

        elapsed = now - self.blink_window_start
        blink_rate = round((self.blink_count / elapsed) * 60, 1) if elapsed > 0 else 0.0
        if elapsed > 60:  # reset window every minute
            self.blink_count = 0
            self.blink_window_start = now

        # Accumulate frames for SNN
        self.frame_buffer.append(gray_resized)
        snn_result = None

        if self.model_loaded and len(self.frame_buffer) >= FRAME_BUFFER:
            frames_list = list(self.frame_buffer)
            snn_result = self._run_snn_inference(frames_list)

        # Build final result
        if snn_result:
            state      = snn_result["state"]
            label      = snn_result["label"]
            confidence = snn_result["confidence"]
            mode       = "SNN"
        elif heuristic["face_detected"]:
            state      = heuristic["state"]
            label      = heuristic["label"]
            confidence = heuristic["confidence"]
            mode       = "Heuristic"
        else:
            state      = "NO_FACE"
            label      = "No Face Detected"
            confidence = 0.0
            mode       = "Heuristic"

        return {
            "state":       state,
            "label":       label,
            "confidence":  confidence,
            "perclos":     perclos_avg,
            "blink_rate":  blink_rate,
            "head_droop":  heuristic.get("head_droop", 0.0),
            "eyes_open":   heuristic.get("eyes_open", 0),
            "face_bbox":   heuristic.get("bbox"),
            "face_det":    heuristic.get("face_detected", False),
            "mode":        mode,
            "model_loaded": self.model_loaded,
        }

    def decode_frame(self, b64_data: str) -> np.ndarray:
        """Decode a base64-encoded JPEG/PNG frame from the browser."""
        # Strip data URL prefix if present
        if "," in b64_data:
            b64_data = b64_data.split(",", 1)[1]
        img_bytes = base64.b64decode(b64_data)
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        return frame


# Singleton
_detector_instance = None

def get_detector() -> NeuroGuardDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = NeuroGuardDetector()
    return _detector_instance

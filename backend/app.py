"""
app.py — NeuroGuard Flask Backend
==========================================
REST API server that receives webcam frames from the browser frontend
and returns real-time drowsiness / distraction detection results.

Run:
    cd backend && python app.py
"""

import os
import sys
import logging
from flask import Flask, request, jsonify
from flask_cors import CORS

# ── Import detector ──
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)
from backend.detector import get_detector

log = logging.getLogger("NeuroGuard")
logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

# ── App ──
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": "*"}})

# Eager-load detector on startup
detector = get_detector()


# ─────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────

@app.route("/api/status", methods=["GET"])
def status():
    """Health check + model info."""
    return jsonify({
        "status":       "running",
        "model_loaded": detector.model_loaded,
        "mode":         "SNN" if detector.model_loaded else "Heuristic (OpenCV)",
        "version":      "1.0.0",
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    Receive a base64-encoded webcam frame and return detection result.

    Expected JSON body:
        { "frame": "data:image/jpeg;base64,..." }

    Response JSON:
        {
            "state":        "ALERT" | "DROWSY" | "TIRED" | "NO_FACE",
            "label":        "Driver Alert 🙂",
            "confidence":   97.3,
            "perclos":      8.2,
            "blink_rate":   18.0,
            "head_droop":   2.0,
            "face_det":     true,
            "face_bbox":    [x, y, w, h] | null,
            "mode":         "SNN" | "Heuristic",
            "model_loaded": true | false,
            "alert":        false,
        }
    """
    data = request.get_json(silent=True)
    if not data or "frame" not in data:
        return jsonify({"error": "Missing 'frame' in request body"}), 400

    try:
        frame = detector.decode_frame(data["frame"])
        if frame is None:
            return jsonify({"error": "Could not decode frame"}), 400

        result = detector.process_frame(frame)

        # Flag for UI alert banner
        result["alert"] = result["state"] in ("DROWSY", "TIRED")

        return jsonify(result)

    except Exception as e:
        log.exception("Error during prediction")
        return jsonify({"error": str(e)}), 500


@app.route("/api/reset", methods=["POST"])
def reset():
    """Clear the frame buffer (call when starting a new session)."""
    detector.frame_buffer.clear()
    detector.blink_count = 0
    detector.perclos_history.clear()
    return jsonify({"status": "reset"})


# ─────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    log.info(f"🚀  NeuroGuard API running at http://localhost:{port}")
    log.info(f"   Model: {'✅ SNN loaded' if detector.model_loaded else '⚠️  Heuristic mode'}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)

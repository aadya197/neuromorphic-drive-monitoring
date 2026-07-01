"""
Neuromorphic Driver Monitoring — Web Dashboard
Runs inference on real 3MDAD dataset samples and shows results live.
Falls back to webcam if camera permission is granted.
"""
import sys, os, time, threading, random, glob
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2
import numpy as np
import torch
import collections
from flask import Flask, Response, render_template_string, jsonify
from spikingjelly.activation_based import functional

from models.model_full import DriverMonitoringModel
from dataset.preprocessing import events_to_voxel
from dataset.dataset_3mdad import load_tiff_stream

ROOT = os.path.dirname(os.path.abspath(__file__))

# ── Load model ────────────────────────────────────────────────────────────────
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
model = DriverMonitoringModel(num_classes=5)
model_path = os.path.join(ROOT, 'model.pth')
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    print("✅ Loaded model.pth")
model.to(device)
model.eval()

LABELS     = ["Drowsy", "Distracted", "Glasses", "Sunglasses", "Night"]
ICONS      = ["😴", "📱", "👓", "🕶️", "🌙"]

# ── Collect dataset samples ───────────────────────────────────────────────────
MDAD_ROOT = os.path.join(ROOT, "data", "3mdad")

def collect_samples():
    """Return list of (face_dir, body_dir, action_label) from dataset."""
    samples = []
    configs = [
        ("Day-RGB1",  "Day-RGB2",  "RGB1", "RGB2",  False),
        ("Night-IR1", "Night-IR2", "IR1",  "IR2",   True),
    ]
    from dataset.dataset_3mdad import DISTRACTED_ACTIONS, DROWSY_ACTIONS
    for fo, bo, fi, bi, night in configs:
        fr = os.path.join(MDAD_ROOT, fo, fi)
        br = os.path.join(MDAD_ROOT, bo, bi)
        if not os.path.isdir(fr): continue
        for subj in sorted(os.listdir(fr))[:10]:   # first 10 subjects
            sf = os.path.join(fr, subj)
            sb = os.path.join(br, subj)
            if not os.path.isdir(sf): continue
            for ac in sorted(os.listdir(sf)):
                if not ac.startswith("AC"): continue
                fd = os.path.join(sf, ac)
                bd = os.path.join(sb, ac) if os.path.isdir(sb) else fd
                # human label
                ac_up = ac.upper()
                if ac_up in DROWSY_ACTIONS:        label = "Fatigue / Drowsy 😴"
                elif ac_up == "AC1":               label = "Safe Driving ✅"
                elif ac_up in DISTRACTED_ACTIONS:  label = "Distracted 📱"
                else:                              label = ac
                samples.append((fd, bd, label, night))
    random.shuffle(samples)
    return samples

SAMPLES = collect_samples()
print(f"✅ Collected {len(SAMPLES)} dataset clips for demo")

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "preds":        [0.0] * 5,
    "label":        "Loading…",
    "night":        False,
    "frame_b64":    "",
    "sample_idx":   0,
    "total":        len(SAMPLES),
}
lock = threading.Lock()

# ── Demo inference thread ─────────────────────────────────────────────────────
def demo_thread():
    idx = 0
    while True:
        if not SAMPLES:
            time.sleep(1)
            continue

        face_dir, body_dir, human_label, is_night = SAMPLES[idx % len(SAMPLES)]
        idx += 1

        # Load voxels from tiff sequence
        vf = load_tiff_stream(face_dir).unsqueeze(0).to(device)  # (1,5,5,64,64)
        vb = load_tiff_stream(body_dir).unsqueeze(0).to(device)

        with torch.no_grad():
            functional.reset_net(model)
            out = model(vf, vb)
            logits = out[0] if isinstance(out, tuple) else out
            probs = torch.sigmoid(logits)[0].cpu().numpy().tolist()

        # Build a representative preview image from the tiff folder
        tiffs = sorted([f for f in os.listdir(face_dir)
                        if f.lower().endswith(('.tiff','.tif'))])
        preview = np.zeros((300, 400, 3), dtype=np.uint8)
        if tiffs:
            mid = tiffs[len(tiffs)//2]
            img = cv2.imread(os.path.join(face_dir, mid), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (400, 300))
                preview = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                # Apply tint for night
                if is_night:
                    tint = np.zeros_like(preview)
                    tint[:,:,0] = 30  # slight blue tint
                    preview = cv2.addWeighted(preview, 0.85, tint, 0.15, 0)

        # Draw label on preview
        cv2.rectangle(preview, (0, 260), (400, 300), (0,0,0), -1)
        cv2.putText(preview, human_label[:35], (10, 287),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 150), 2)

        # Encode to JPEG base64
        _, buf = cv2.imencode('.jpg', preview, [cv2.IMWRITE_JPEG_QUALITY, 85])
        b64 = "data:image/jpeg;base64," + __import__('base64').b64encode(buf).decode()

        with lock:
            state["preds"]      = probs
            state["label"]      = human_label
            state["night"]      = is_night
            state["frame_b64"]  = b64
            state["sample_idx"] = idx % len(SAMPLES)

        time.sleep(3.5)   # show each clip for 3.5 seconds

# ── Webcam thread (if available) ──────────────────────────────────────────────
webcam_ok = False
def webcam_thread():
    global webcam_ok
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return
    webcam_ok = True
    print("✅ Webcam active!")

    vq_face = collections.deque(maxlen=5)
    vq_body = collections.deque(maxlen=5)
    cf, cb  = [], []
    ret, prev = cap.read()
    if not ret: return
    pb = cv2.cvtColor(cv2.resize(prev,(64,64)), cv2.COLOR_BGR2GRAY)
    pf = pb.copy()
    last_bbox = None
    t = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) > 0:
            x,y,w,h = faces[0]; m=int(w*.2)
            last_bbox=(max(0,x-m),max(0,y-m),
                       min(frame.shape[1],x+w+m),min(frame.shape[0],y+h+m))
        if last_bbox:
            x1,y1,x2,y2=last_bbox
            face_img=frame[y1:y2,x1:x2]
            cv2.rectangle(display,(x1,y1),(x2,y2),(0,255,255),2)
        else:
            hf,wf=frame.shape[:2]
            face_img=frame[hf//4:3*hf//4,wf//4:3*wf//4]

        cb_ = cv2.cvtColor(cv2.resize(frame,(64,64)),cv2.COLOR_BGR2GRAY)
        cf_ = cv2.cvtColor(cv2.resize(face_img if face_img.size>0 else frame,(64,64)),cv2.COLOR_BGR2GRAY)

        def evs(c,p,ti):
            d=c.astype(np.int16)-p.astype(np.int16)
            e=[]
            for xx,yy in zip(*np.where(d>5)[::-1]): e.append([xx,yy,ti, 1])
            for xx,yy in zip(*np.where(d<-5)[::-1]): e.append([xx,yy,ti,-1])
            return e

        cb.extend(evs(cb_,pb,t)); cf.extend(evs(cf_,pf,t))
        pb,pf=cb_,cf_; t+=1

        if t%6==0:
            def vox(e):
                if not e: return torch.zeros((5,64,64))
                a=np.array(e,dtype=np.float32)
                v=events_to_voxel(a); mx=v.max()
                return v/mx if mx>0 else v
            vq_face.append(vox(cf)); vq_body.append(vox(cb))
            cf,cb=[],[]
            if len(vq_face)==5:
                vf=torch.stack(list(vq_face)).unsqueeze(0).to(device)
                vb=torch.stack(list(vq_body)).unsqueeze(0).to(device)
                with torch.no_grad():
                    functional.reset_net(model)
                    out=model(vf,vb)
                    logits=out[0] if isinstance(out,tuple) else out
                    probs=torch.sigmoid(logits)[0].cpu().numpy().tolist()
                _,buf=cv2.imencode('.jpg',display,[cv2.IMWRITE_JPEG_QUALITY,80])
                b64="data:image/jpeg;base64,"+__import__('base64').b64encode(buf).decode()
                with lock:
                    state["preds"]=probs
                    state["label"]="Live Webcam"
                    state["frame_b64"]=b64
        time.sleep(0.033)
    cap.release()

# ── Flask app ─────────────────────────────────────────────────────────────────
app = Flask(__name__)

HTML = r"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Neuromorphic Driver Monitoring</title>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap" rel="stylesheet">
<style>
  :root{--bg:#060c1a;--card:#0d1528;--border:#1a2540;--text:#e2e8f0;--muted:#64748b;--accent:#818cf8;--green:#34d399;--red:#f87171;--orange:#fb923c;}
  *{box-sizing:border-box;margin:0;padding:0;}
  body{font-family:'Inter',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;}
  body::before{content:'';position:fixed;inset:0;z-index:0;
    background-image:radial-gradient(circle at 20% 20%,rgba(99,102,241,.08) 0%,transparent 50%),
                     radial-gradient(circle at 80% 80%,rgba(139,92,246,.08) 0%,transparent 50%);
    pointer-events:none;}

  .wrap{position:relative;z-index:1;max-width:1280px;margin:0 auto;padding:20px;}

  /* Header */
  header{display:flex;align-items:center;gap:14px;padding:18px 24px;
    background:linear-gradient(135deg,#0f1a3e,#0d1528);
    border:1px solid #1e3a8a33;border-radius:20px;margin-bottom:20px;
    box-shadow:0 0 60px rgba(99,102,241,.12);}
  .logo{width:52px;height:52px;border-radius:14px;
    background:linear-gradient(135deg,#4f46e5,#7c3aed);
    display:flex;align-items:center;justify-content:center;font-size:26px;flex-shrink:0;
    box-shadow:0 0 20px rgba(99,102,241,.5);}
  header h1{font-size:1.4rem;font-weight:800;letter-spacing:-.02em;}
  header p{font-size:.78rem;color:var(--muted);margin-top:3px;}
  .pill{margin-left:auto;padding:5px 14px;border-radius:999px;font-size:.72rem;font-weight:700;letter-spacing:.06em;
    border:1px solid;display:flex;align-items:center;gap:6px;}
  .pill.live{border-color:#22c55e;color:#86efac;background:rgba(34,197,94,.1);}
  .pill.demo{border-color:#f59e0b;color:#fcd34d;background:rgba(245,158,11,.1);}
  .blink{width:7px;height:7px;border-radius:50%;animation:blink 1.2s ease infinite;}
  .blink.green{background:#22c55e;box-shadow:0 0 6px #22c55e;}
  .blink.amber{background:#f59e0b;box-shadow:0 0 6px #f59e0b;}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:.3}}

  /* Grid */
  .grid{display:grid;grid-template-columns:1fr 380px;gap:18px;}

  /* Card */
  .card{background:var(--card);border:1px solid var(--border);border-radius:18px;overflow:hidden;}
  .card-hdr{padding:12px 18px;border-bottom:1px solid var(--border);
    font-size:.7rem;font-weight:700;letter-spacing:.1em;color:var(--muted);text-transform:uppercase;
    display:flex;align-items:center;justify-content:space-between;}
  .chip{padding:3px 10px;border-radius:999px;font-size:.65rem;font-weight:700;}
  .chip.green{background:rgba(52,211,153,.15);color:#6ee7b7;border:1px solid #34d399;}
  .chip.amber{background:rgba(251,191,36,.12);color:#fcd34d;border:1px solid #f59e0b;}

  /* Frame */
  #previewImg{width:100%;display:block;aspect-ratio:4/3;object-fit:cover;background:#000;}
  .frame-meta{padding:10px 16px;font-size:.75rem;color:var(--muted);
    border-top:1px solid var(--border);display:flex;justify-content:space-between;align-items:center;}
  #clipLabel{color:var(--text);font-weight:600;}

  /* Predictions */
  .pred-list{padding:14px;display:flex;flex-direction:column;gap:12px;}
  .pred-row{}
  .pred-top{display:flex;justify-content:space-between;align-items:center;margin-bottom:5px;}
  .pred-name{font-size:.88rem;font-weight:600;display:flex;align-items:center;gap:5px;}
  .pred-pct{font-size:.82rem;font-weight:700;color:var(--accent);}
  .bar-bg{height:7px;border-radius:4px;background:rgba(255,255,255,.06);overflow:hidden;}
  .bar-fg{height:100%;border-radius:4px;transition:width .5s cubic-bezier(.4,0,.2,1),background .4s ease;}
  .status-tag{padding:1px 7px;border-radius:999px;font-size:.6rem;font-weight:700;margin-left:6px;}
  .tag-alert{background:rgba(248,113,113,.2);color:#fca5a5;border:1px solid var(--red);}
  .tag-ok{background:rgba(52,211,153,.1);color:#6ee7b7;border:1px solid var(--green);}

  /* Stats */
  .stats{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin-top:18px;}
  .stat{background:var(--card);border:1px solid var(--border);border-radius:14px;padding:16px;text-align:center;}
  .stat-val{font-size:1.7rem;font-weight:900;
    background:linear-gradient(90deg,#818cf8,#a78bfa);
    -webkit-background-clip:text;-webkit-text-fill-color:transparent;}
  .stat-lbl{font-size:.72rem;color:var(--muted);margin-top:3px;}

  /* Progress bar at top */
  #progress{position:fixed;top:0;left:0;height:3px;background:linear-gradient(90deg,#6366f1,#8b5cf6);transition:width .4s ease;z-index:999;}

  @media(max-width:820px){.grid{grid-template-columns:1fr}.stats{grid-template-columns:repeat(2,1fr)}}
</style>
</head>
<body>
<div id="progress" style="width:0%"></div>

<div class="wrap">
  <header>
    <div class="logo">🧠</div>
    <div>
      <h1>Neuromorphic Driver Monitoring</h1>
      <p>Spiking Neural Network · Event Camera · 5-Class Multi-Label Detection</p>
    </div>
    <div class="pill demo" id="modePill">
      <div class="blink amber"></div>
      <span id="modeLabel">DATASET DEMO</span>
    </div>
  </header>

  <div class="grid">
    <!-- Left: preview -->
    <div class="card">
      <div class="card-hdr">
        <span>📷 Input Frame (3MDAD Dataset)</span>
        <div class="chip amber" id="sampleChip">Sample 1 / —</div>
      </div>
      <img id="previewImg" src="" alt="Loading...">
      <div class="frame-meta">
        <span id="clipLabel">Loading dataset...</span>
        <span id="modeTag" style="color:var(--accent);font-weight:600">SNN Inference Active</span>
      </div>
    </div>

    <!-- Right: predictions -->
    <div class="card">
      <div class="card-hdr">
        <span>🎯 Model Predictions</span>
        <div class="chip green">Live</div>
      </div>
      <div class="pred-list" id="predList"></div>
    </div>
  </div>

  <!-- Stats -->
  <div class="stats">
    <div class="stat"><div class="stat-val">87%</div><div class="stat-lbl">Train Accuracy</div></div>
    <div class="stat"><div class="stat-val">84%</div><div class="stat-lbl">Val Accuracy</div></div>
    <div class="stat"><div class="stat-val">318K</div><div class="stat-lbl">Parameters</div></div>
    <div class="stat"><div class="stat-val" id="alertCount">0</div><div class="stat-lbl">Alerts Detected</div></div>
  </div>
</div>

<script>
const LABELS = ["Drowsy","Distracted","Glasses","Sunglasses","Night"];
const ICONS  = ["😴","📱","👓","🕶️","🌙"];
const GRAD   = [
  "linear-gradient(90deg,#ef4444,#f97316)",
  "linear-gradient(90deg,#f97316,#fbbf24)",
  "linear-gradient(90deg,#8b5cf6,#a78bfa)",
  "linear-gradient(90deg,#3b82f6,#6366f1)",
  "linear-gradient(90deg,#475569,#64748b)"
];

let alertTotal = 0;
let built = false;

function buildPreds() {
  if (built) return;
  built = true;
  const el = document.getElementById('predList');
  LABELS.forEach((l,i) => {
    el.innerHTML += `
    <div class="pred-row">
      <div class="pred-top">
        <span class="pred-name">${ICONS[i]} ${l}<span class="status-tag tag-ok" id="tag${i}">OK</span></span>
        <span class="pred-pct" id="pct${i}">0%</span>
      </div>
      <div class="bar-bg"><div class="bar-fg" id="bar${i}" style="width:0%;background:${GRAD[i]}"></div></div>
    </div>`;
  });
}

async function poll() {
  try {
    const r = await fetch('/state');
    const d = await r.json();

    // Update image
    if (d.frame_b64) {
      document.getElementById('previewImg').src = d.frame_b64;
    }
    document.getElementById('clipLabel').textContent = d.label;
    document.getElementById('sampleChip').textContent = `Sample ${d.sample_idx} / ${d.total}`;

    // Progress bar
    document.getElementById('progress').style.width = (d.sample_idx / d.total * 100) + '%';

    // Preds
    let alerts = 0;
    d.preds.forEach((p,i) => {
      const pct = Math.round(p * 100);
      document.getElementById(`pct${i}`).textContent = pct + '%';
      document.getElementById(`bar${i}`).style.width = pct + '%';
      const tag = document.getElementById(`tag${i}`);
      if (p > 0.5) {
        tag.textContent = 'ALERT'; tag.className = 'status-tag tag-alert';
        alerts++;
      } else {
        tag.textContent = 'OK'; tag.className = 'status-tag tag-ok';
      }
    });
    if (alerts > 0) alertTotal += alerts;
    document.getElementById('alertCount').textContent = alertTotal;

    // Mode pill
    if (d.webcam) {
      document.getElementById('modePill').className = 'pill live';
      document.getElementById('modeLabel').textContent = 'LIVE WEBCAM';
      document.querySelector('.blink').className = 'blink green';
      document.getElementById('sampleChip').textContent = 'Webcam Active';
    }

  } catch(e) {}
}

buildPreds();
setInterval(poll, 400);
poll();
</script>
</body>
</html>
"""

@app.route('/')
def index():
    return render_template_string(HTML)

@app.route('/state')
def get_state():
    with lock:
        s = dict(state)
    s["webcam"] = webcam_ok
    return jsonify(s)

if __name__ == '__main__':
    threading.Thread(target=demo_thread,   daemon=True).start()
    threading.Thread(target=webcam_thread, daemon=True).start()
    print("\n🚀 Dashboard → http://localhost:5050\n")
    app.run(host='0.0.0.0', port=5050, debug=False, threaded=True)

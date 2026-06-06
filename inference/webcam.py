import cv2
import numpy as np
import torch
import collections
from spikingjelly.activation_based import functional

from models.model_full import DriverMonitoringModel
from dataset.preprocessing import events_to_voxel

def run_webcam():
    print("[Webcam] Loading 5-target SNN model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = DriverMonitoringModel(num_classes=5)
    
    try:
        model.load_state_dict(torch.load('model.pth', map_location=device))
        print("[Webcam] Loaded trained model.pth")
    except Exception as e:
        print(f"[Webcam] Warning: model.pth not found or failed to load. Using untrained weights. Error: {e}")
    
    model.to(device)
    model.eval()

    # Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[Webcam] Error: Cannot open webcam.")
        return

    # Queues to hold temporal bins (num_chunks = 5)
    voxel_q_face = collections.deque(maxlen=5)
    voxel_q_body = collections.deque(maxlen=5)
    
    chunk_events_face = []
    chunk_events_body = []
    
    ret, prev_frame = cap.read()
    if not ret: return
    
    prev_gray_body = cv2.cvtColor(cv2.resize(prev_frame, (64, 64)), cv2.COLOR_BGR2GRAY)
    prev_gray_face = prev_gray_body.copy()
    
    last_face_bbox = None
    t = 0
    frames_per_chunk = 6 # Collect events over 6 frames to form 1 temporal bin (voxel)
    
    labels = ["Drowsy", "Distracted", "Glasses", "Sunglasses", "Night"]
    preds = [0.0] * 5
    
    print("[Webcam] Starting live stream. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret: break
        
        display_frame = frame.copy()
        
        # 1. Detect Face for Smart Crop
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) > 0:
            x, y, w, h = faces[0]
            # Add a 20% margin around the face
            margin = int(w * 0.2)
            x1 = max(0, x - margin)
            y1 = max(0, y - margin)
            x2 = min(frame.shape[1], x + w + margin)
            y2 = min(frame.shape[0], y + h + margin)
            last_face_bbox = (x1, y1, x2, y2)
        
        if last_face_bbox is not None:
            x1, y1, x2, y2 = last_face_bbox
            face_img = frame[y1:y2, x1:x2]
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 255, 0), 2)
        else:
            # Fallback to center crop if no face ever found
            h_f, w_f, _ = frame.shape
            face_img = frame[h_f//4:3*h_f//4, w_f//4:3*w_f//4]
            
        # 2. Resize to 64x64 for the SNN
        curr_gray_body = cv2.cvtColor(cv2.resize(frame, (64, 64)), cv2.COLOR_BGR2GRAY)
        curr_gray_face = cv2.cvtColor(cv2.resize(face_img, (64, 64)), cv2.COLOR_BGR2GRAY)
        
        # 3. Compute frame differences to generate "Events"
        threshold = 5
        def get_events(curr, prev, t_idx):
            diff = curr.astype(np.int16) - prev.astype(np.int16)
            pos = diff > threshold
            neg = diff < -threshold
            evs = []
            ys, xs = np.where(pos)
            for xx, yy in zip(xs, ys): evs.append([xx, yy, t_idx, 1])
            ys, xs = np.where(neg)
            for xx, yy in zip(xs, ys): evs.append([xx, yy, t_idx, -1])
            return evs

        chunk_events_body.extend(get_events(curr_gray_body, prev_gray_body, t))
        chunk_events_face.extend(get_events(curr_gray_face, prev_gray_face, t))
        
        prev_gray_body = curr_gray_body
        prev_gray_face = curr_gray_face
        t += 1
        
        # 4. Once we collect enough frames, convert events to a Voxel
        if t % frames_per_chunk == 0:
            def process_chunk(evs):
                if len(evs) == 0:
                    return torch.zeros((5, 64, 64))
                evs_arr = np.array(evs, dtype=np.float32)
                voxel = events_to_voxel(evs_arr)
                voxel_max = voxel.max()
                if voxel_max > 0:
                    voxel = voxel / voxel_max
                return voxel
                
            voxel_q_face.append(process_chunk(chunk_events_face))
            voxel_q_body.append(process_chunk(chunk_events_body))
            
            chunk_events_face = []
            chunk_events_body = []
            
            # 5. If we have 5 temporal bins, run the SNN!
            if len(voxel_q_face) == 5:
                vf = torch.stack(list(voxel_q_face)).unsqueeze(0).to(device)
                vb = torch.stack(list(voxel_q_body)).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    # Reset SNN membrane potentials between inference windows
                    # to prevent state leaking across temporal batches.
                    functional.reset_net(model)
                    logits = model(vf, vb)
                    probs = torch.sigmoid(logits)[0].cpu().numpy()
                    preds = probs
                    
        # 6. Overlay predictions on the screen
        y0 = 40
        for i, (name, prob) in enumerate(zip(labels, preds)):
            is_active = prob > 0.5
            color = (0, 0, 255) if is_active else (0, 255, 0)
            text = f"{name}: {prob*100:.1f}%"
            # Add a slight background for readability
            cv2.rectangle(display_frame, (10, y0 + i*35 - 25), (300, y0 + i*35 + 10), (0,0,0), -1)
            cv2.putText(display_frame, text, (15, y0 + i*35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        cv2.imshow("Neuromorphic Driver Monitoring (5-Target SNN)", display_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_webcam()

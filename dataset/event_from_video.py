import cv2
import numpy as np


def video_to_events(video_path, threshold=10, max_frames=30, max_events=20000):
    cap = cv2.VideoCapture(video_path)

    ret, prev = cap.read()
    if not ret:
        cap.release()
        return np.zeros((1, 4), dtype=np.int16)

    prev = cv2.cvtColor(prev, cv2.COLOR_BGR2GRAY)

    events = []
    t: int = 0
    frame_count: int = 0

    while True:
        ret, frame = cap.read()

        if not ret or frame_count >= max_frames:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # intensity difference
        diff = gray.astype(np.int16) - prev.astype(np.int16)

        # event generation
        pos = diff > threshold
        neg = diff < -int(threshold)

        # positive events
        ys, xs = np.where(pos)
        for x, y in zip(xs, ys):
            events.append([int(x), int(y), int(t), 1])

        # negative events
        ys, xs = np.where(neg)
        for x, y in zip(xs, ys):
            events.append([int(x), int(y), int(t), -1])

        prev = gray
        t += 1  # pyre-ignore
        frame_count += 1  # pyre-ignore

        # 🔥 limit event explosion (CRITICAL)
        if len(events) > max_events:
            break

    cap.release()

    # 🔥 convert early (important for speed + memory)
    if len(events) == 0:
        return np.zeros((1, 4), dtype=np.int16)

    events = np.array(events, dtype=np.int16)

    # 🔥 downsample if still large
    if len(events) > max_events:
        step = len(events) // max_events
        events = events[::step]

    return events
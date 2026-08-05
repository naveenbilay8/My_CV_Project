import cv2
import torch
import threading
from rfdetr import RFDETRSmall
from PIL import Image
import numpy as np
import time

VIDEO_PATHS = [
    r"test/1.mp4",
    r"test/i1.mp4",
    r"test/2.mp4",
    r"test/5.mp4"
]
MODEL_PATH = "checkpoints/checkpoint_best_ema.pth"
CONF_THRESH = 0.50
CLASS_NAMES = ["emr", "emr", "nem", "emr"]

model = RFDETRSmall(pretrain_weights=MODEL_PATH, num_classes=len(CLASS_NAMES))
model.model.model.eval()

feed_caps = [cv2.VideoCapture(vp) for vp in VIDEO_PATHS]
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)

TARGET_FPS = 30
FRAME_INTERVAL = 1.0 / TARGET_FPS

frame_buffer = [None] * 4
feed_has_ambulance = [False] * 4
feed_density = [0] * 4
lock = threading.Lock()
running = True

# Track last 5 emergency detections per feed
emergency_history = [[] for _ in range(4)]

# ---------------- Detection Thread ----------------
def detection_thread():
    global feed_has_ambulance, feed_density, emergency_history

    while running:
        with lock:
            local_frames = frame_buffer.copy()

        counts = [0] * 4
        new_emergency = [False] * 4

        for idx, frame in enumerate(local_frames):
            if frame is None:
                continue
            pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            with torch.no_grad():
                detections = model.predict(pil_frame, threshold=CONF_THRESH)

            vcount = 0
            has_emr = False
            for score, label in zip(detections.confidence, detections.class_id):
                if score < CONF_THRESH:
                    continue
                label_name = CLASS_NAMES[label] if label < len(CLASS_NAMES) else str(label)
                vcount += 1
                if "emr" in label_name.lower():
                    has_emr = True

            counts[idx] = vcount
            new_emergency[idx] = has_emr

        confirmed = [False] * 4
        for i in range(4):
            emergency_history[i].append(new_emergency[i])
            if len(emergency_history[i]) > 3:
                emergency_history[i].pop(0)
            confirmed[i] = all(emergency_history[i]) and len(emergency_history[i]) == 3

        with lock:
            feed_has_ambulance = confirmed
            feed_density = counts

        time.sleep(0.05)

thread = threading.Thread(target=detection_thread, daemon=True)
thread.start()

last_frame_time = time.time()
current_green_idx = -1
green_start_time = time.time()
GREEN_DURATION = 5

# ---------------- Main Display Loop ----------------
while True:
    frames = []
    ret_any = False
    for idx, cap in enumerate(feed_caps):
        ret, frame = cap.read()
        if not ret:
            frame = np.zeros((240, 320, 3), dtype=np.uint8)
        else:
            ret_any = True
            frame = cv2.resize(frame, (320, 240))
        with lock:
            frame_buffer[idx] = frame.copy()
        frames.append(frame)

    if not ret_any:
        break

    with lock:
        ambulances = feed_has_ambulance.copy()
        density = feed_density.copy()

    signals = [COLOR_RED] * 4

    # ---------------- Scheduling Logic ----------------
    if any(ambulances):
        current_green_idx = ambulances.index(True)
        green_start_time = time.time()
        GREEN_DURATION = 5
    else:
        if current_green_idx == -1 or (time.time() - green_start_time) >= GREEN_DURATION:
            next_idx = (current_green_idx + 1) % 4
            d = density[next_idx]
            GREEN_DURATION = max(3, min(20, d * 1))
            current_green_idx = next_idx
            green_start_time = time.time()

    signals[current_green_idx] = COLOR_GREEN

    # ---------------- Visualization ----------------
    remaining_time = max(0, GREEN_DURATION - int(time.time() - green_start_time))
    for idx in range(4):
        f = frames[idx]
        cx, cy = 30, 40
        r = 12
        # Red light
        cv2.circle(f, (cx, cy), r, COLOR_RED, -1 if signals[idx] == COLOR_RED else 2)
        # Green light
        cv2.circle(f, (cx, cy + 35), r, COLOR_GREEN, -1 if signals[idx] == COLOR_GREEN else 2)

        cv2.putText(f, 'SIGNAL', (cx - 14, cy + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.putText(f, f"Vehicles: {density[idx]}", (10, 230),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
        if idx == current_green_idx:
            cv2.putText(f, f"Timer: {remaining_time}s", (200, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    # ✅ Fixed np.vstack syntax
    grid = np.vstack((
        np.hstack((frames[0], frames[1])),
        np.hstack((frames[2], frames[3]))
    ))

    cv2.imshow("Traffic Simulation (Red & Green Only)", grid)

    elapsed = time.time() - last_frame_time
    if elapsed < FRAME_INTERVAL:
        time.sleep(FRAME_INTERVAL - elapsed)
    last_frame_time = time.time()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

running = False
for cap in feed_caps:
    cap.release()
cv2.destroyAllWindows()

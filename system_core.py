# system_core.py
import os
import cv2
import time
import math
import collections
import threading
import numpy as np
import urllib.request
import torch

from one_euro_filter import OneEuroFilter
import native_input as win32
from gesture_lstm import TemporalGestureLSTM

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------------------------------------------------------------
# 1. State Machine Definitions
# -------------------------------------------------------------
class SystemState:
    HOVER = 0
    DRAGGING = 1
    SCROLLING = 2

CLASSES = ["TRACKING", "SWIPE_LEFT", "SWIPE_RIGHT", "PINCH", "NEUTRAL"]

# -------------------------------------------------------------
# 2. Threaded Asynchronous Camera Stream
# -------------------------------------------------------------
class VideoCaptureThread:
    def __init__(self, src=0):
        self.cap = cv2.VideoCapture(src)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            time.sleep(0.005)

    def read(self):
        with self.lock:
            return self.ret, self.frame.copy() if self.frame is not None else None

    def stop(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()

# -------------------------------------------------------------
# 3. Dynamic Cursor Velocity Ballistics (Medium-Slow Calibration)
# -------------------------------------------------------------
class CursorBallistics:
    def __init__(self, gain=0.35, threshold=0.035, max_mult=1.35):
        # Moderate gain and lowered multiplier for calm, controlled tracking
        self.gain = gain
        self.threshold = threshold
        self.max_mult = max_mult
        self.last_pos = None
        self.last_time = None

    def apply(self, norm_x, norm_y):
        now = time.time()
        if self.last_pos is None or self.last_time is None:
            self.last_pos = (norm_x, norm_y)
            self.last_time = now
            return norm_x, norm_y

        dt = now - self.last_time
        if dt < 0.008:
            return self.last_pos[0], self.last_pos[1]

        dx = norm_x - self.last_pos[0]
        dy = norm_y - self.last_pos[1]
        dist = math.hypot(dx, dy)
        velocity = dist / dt

        multiplier = 1.0
        if velocity > self.threshold:
            multiplier = 1.0 + self.gain * (velocity - self.threshold)
            multiplier = min(multiplier, self.max_mult)

        res_x = self.last_pos[0] + dx * multiplier
        res_y = self.last_pos[1] + dy * multiplier

        if math.isnan(res_x) or math.isinf(res_x):
            res_x = norm_x
        if math.isnan(res_y) or math.isinf(res_y):
            res_y = norm_y

        clamped_x = float(np.clip(res_x, 0.0, 1.0))
        clamped_y = float(np.clip(res_y, 0.0, 1.0))

        self.last_pos = (clamped_x, clamped_y)
        self.last_time = now
        return clamped_x, clamped_y

# -------------------------------------------------------------
# 4. Main Runtime Execution
# -------------------------------------------------------------
def run_advanced_core():
    MODEL_PATH = "hand_landmarker.task"
    if not os.path.exists(MODEL_PATH):
        print("Downloading hand_landmarker.task...")
        urllib.request.urlretrieve(
            "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
            MODEL_PATH
        )
        print("Download complete.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lstm_model = TemporalGestureLSTM().to(device)
    lstm_model.eval()

    shared_result = None
    result_lock = threading.Lock()

    def async_callback(result, output_image, timestamp_ms):
        nonlocal shared_result
        with result_lock:
            shared_result = result

    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.LIVE_STREAM,
        num_hands=1,
        min_hand_detection_confidence=0.75,
        min_tracking_confidence=0.75,
        result_callback=async_callback
    )
    landmarker = vision.HandLandmarker.create_from_options(options)
    video_stream = VideoCaptureThread(src=0)

    # Damping filters tuned for steady medium-slow glide
    filter_x = OneEuroFilter(t0=time.time(), x0=0.5, min_cutoff=0.65, beta=0.009)
    filter_y = OneEuroFilter(t0=time.time(), x0=0.5, min_cutoff=0.65, beta=0.009)
    ballistics = CursorBallistics(gain=0.35, threshold=0.035, max_mult=1.35)

    temporal_buffer = collections.deque(maxlen=30)
    state = SystemState.HOVER
    last_state_change = time.time()
    start_time = time.time()
    last_sent_ts = 0

    # Trackpad scroll parameters
    scroll_prev_y = None
    SCROLL_SENSITIVITY = 1200

    print("System active. Press 'q' to quit.")

    try:
        while True:
            ret, frame = video_stream.read()
            if not ret or frame is None:
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            now = time.time()
            ts_ms = int((now - start_time) * 1000)
            if ts_ms <= last_sent_ts:
                ts_ms = last_sent_ts + 1
            last_sent_ts = ts_ms

            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            landmarker.detect_async(mp_img, ts_ms)

            with result_lock:
                current_detection = shared_result

            gesture_label = "IDLE"

            if current_detection and current_detection.hand_landmarks:
                landmarks = current_detection.hand_landmarks[0]

                # Coordinate normalization (scale-invariant relative to wrist)
                wrist = np.array([landmarks[0].x, landmarks[0].y])
                scale = np.linalg.norm(np.array([landmarks[9].x, landmarks[9].y]) - wrist)
                scale = max(scale, 1e-4)

                frame_features = []
                for pt in landmarks:
                    frame_features.extend([(pt.x - wrist[0]) / scale, (pt.y - wrist[1]) / scale])

                temporal_buffer.append(frame_features)

                if len(temporal_buffer) == 30:
                    with torch.no_grad():
                        tensor_in = torch.tensor([list(temporal_buffer)], dtype=torch.float32).to(device)
                        logits = lstm_model(tensor_in)
                        pred = torch.argmax(logits, dim=-1).item()
                        gesture_label = CLASSES[pred]

                # Check finger extension states
                index_up = landmarks[8].y < landmarks[6].y
                middle_up = landmarks[12].y < landmarks[10].y
                ring_down = landmarks[16].y > landmarks[14].y
                pinky_down = landmarks[20].y > landmarks[18].y

                idx_norm_x, idx_norm_y = landmarks[8].x, landmarks[8].y

                # Balanced tracking bounds: [0.10, 0.90] lowers rapid travel
                mapped_x = float(np.interp(idx_norm_x, [0.10, 0.90], [0.0, 1.0]))
                mapped_y = float(np.interp(idx_norm_y, [0.10, 0.90], [0.0, 1.0]))

                acc_x, acc_y = ballistics.apply(mapped_x, mapped_y)
                filt_x = filter_x.filter(acc_x, now)
                filt_y = filter_y.filter(acc_y, now)

                if math.isnan(filt_x) or math.isinf(filt_x):
                    filt_x = 0.5
                if math.isnan(filt_y) or math.isinf(filt_y):
                    filt_y = 0.5

                filt_x = float(np.clip(filt_x, 0.0, 1.0))
                filt_y = float(np.clip(filt_y, 0.0, 1.0))

                pinch_dist = math.hypot(
                    (landmarks[8].x - landmarks[4].x) * w,
                    (landmarks[8].y - landmarks[4].y) * h
                )

                # -------------------------------------------------------------
                # State Machine: HOVER, DRAG, & PROPORTIONAL SCROLL
                # -------------------------------------------------------------
                # 1. SCROLL MODE: Index + Middle UP, Ring + Pinky DOWN (✌️ gesture)
                if index_up and middle_up and ring_down and pinky_down:
                    state = SystemState.SCROLLING
                    current_scroll_y = (landmarks[8].y + landmarks[12].y) / 2.0

                    if scroll_prev_y is not None:
                        delta_y = scroll_prev_y - current_scroll_y
                        if abs(delta_y) > 0.008:
                            scroll_amount = int(delta_y * SCROLL_SENSITIVITY)
                            if scroll_amount != 0:
                                win32.mouse_scroll(scroll_amount)

                    scroll_prev_y = current_scroll_y

                # 2. MOUSE NAVIGATION & DRAG MODES
                else:
                    scroll_prev_y = None  # Reset baseline

                    # Left Click & Drag (Pinch Thumb and Index)
                    if pinch_dist < 28:
                        if state != SystemState.DRAGGING and (now - last_state_change > 0.2):
                            win32.mouse_down()
                            state = SystemState.DRAGGING
                            last_state_change = now
                        win32.set_mouse_pos(filt_x, filt_y)

                    else:
                        if state == SystemState.DRAGGING and pinch_dist >= 35:
                            win32.mouse_up()
                            state = SystemState.HOVER
                            last_state_change = now

                        # Smooth medium-slow cursor navigation
                        if index_up:
                            state = SystemState.HOVER
                            win32.set_mouse_pos(filt_x, filt_y)

            state_names = {0: "HOVER", 1: "DRAGGING", 2: "SCROLLING"}
            cv2.putText(frame, f"State: {state_names[state]} | Temporal: {gesture_label}", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
            cv2.imshow("Advanced Vision Control Core", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        if state == SystemState.DRAGGING:
            win32.mouse_up()
        video_stream.stop()
        landmarker.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    run_advanced_core()
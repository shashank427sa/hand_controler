import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
import cv2
import math
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request
import os

st.set_page_config(page_title="AI Hand Gesture Controller", layout="wide")

st.title("✋ AI Vision Hand Controller")
st.markdown("Real-time computer vision gesture tracking running live in the browser.")

# Ensure model task file exists
MODEL_PATH = "hand_landmarker.task"
if not os.path.exists(MODEL_PATH):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        MODEL_PATH
    )

# Setup MediaPipe Hand Landmarker for static/video streaming frames
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_hands=1,
    min_hand_detection_confidence=0.7
)
detector = vision.HandLandmarker.create_from_options(options)

class HandGestureTransformer(VideoTransformerBase):
    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)
        h, w, _ = img.shape

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = detector.detect(mp_image)

        status = "IDLE"

        if result.hand_landmarks:
            lm = result.hand_landmarks[0]

            # Fingertip coordinates
            idx_x, idx_y = int(lm[8].x * w), int(lm[8].y * h)
            thb_x, thb_y = int(lm[4].x * w), int(lm[4].y * h)

            # Check extended fingers
            index_up = lm[8].y < lm[6].y
            middle_up = lm[12].y < lm[10].y
            ring_down = lm[16].y > lm[14].y

            # Pinch detection
            pinch_dist = math.hypot(idx_x - thb_x, idx_y - thb_y)

            if pinch_dist < 32:
                status = "PINCH / CLICK"
                cv2.circle(img, (idx_x, idx_y), 15, (0, 0, 255), cv2.FILLED)
            elif index_up and middle_up and ring_down:
                status = "SCROLL MODE (✌️)"
                cv2.circle(img, (idx_x, idx_y), 10, (255, 255, 0), cv2.FILLED)
                cv2.circle(img, (int(lm[12].x * w), int(lm[12].y * h)), 10, (255, 255, 0), cv2.FILLED)
            elif index_up:
                status = "TRACKING / HOVER"
                cv2.circle(img, (idx_x, idx_y), 10, (0, 255, 0), cv2.FILLED)

            # Landmark dots
            for pt in lm:
                cv2.circle(img, (int(pt.x * w), int(pt.y * h)), 3, (0, 255, 255), -1)

        # Telemetry HUD
        cv2.rectangle(img, (10, 10), (320, 60), (0, 0, 0), cv2.FILLED)
        cv2.putText(img, f"Gesture: {status}", (20, 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return img

# WebRTC STUN server configuration for browser camera access
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

webrtc_streamer(
    key="hand-control-demo",
    video_transformer_factory=HandGestureTransformer,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
)
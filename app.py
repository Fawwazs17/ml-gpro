"""
BIM Recognition System — Streamlit Deployment App
BICS 4340 Machine Learning | Group G

Run with:  streamlit run app.py
"""

import json
import tempfile
from pathlib import Path

import av
import cv2
import joblib
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ─── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BIM Sign Recognition",
    page_icon="🤟",
    layout="centered",
)

# ─── MLP Architecture (must match notebook) ─────────────────────────────────
class KeypointMLP(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


# ─── Load config & models (cached) ──────────────────────────────────────────
CONFIG_PATH = Path("models/config.json")

@st.cache_resource
def load_config():
    if not CONFIG_PATH.exists():
        return None
    with open(CONFIG_PATH) as f:
        return json.load(f)


@st.cache_resource
def load_models(config):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # MLP
    mlp = KeypointMLP(config["input_dim"], config["num_classes"])
    mlp.load_state_dict(torch.load(config["mlp_model_path"], map_location=device))
    mlp.to(device)
    mlp.eval()

    # YOLOv11
    yolo = YOLO(config["yolo_model_path"])

    # Scaler + LabelEncoder
    scaler = joblib.load(config["scaler_path"])
    le = joblib.load(config["encoder_path"])

    return mlp, yolo, scaler, le, device


# ─── MediaPipe extractor (Tasks API for mediapipe >= 0.10) ───────────────────

@st.cache_resource
def load_landmarker(model_path: str):
    base_opts = mp_python.BaseOptions(model_asset_path=model_path)
    options = mp_vision.HandLandmarkerOptions(
        base_options=base_opts, num_hands=1,
        min_hand_detection_confidence=0.3,
        min_hand_presence_confidence=0.3
    )
    return mp_vision.HandLandmarker.create_from_options(options)


def extract_keypoints(image_rgb: np.ndarray, landmarker):
    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
    results = landmarker.detect(mp_img)
    if not results.hand_landmarks:
        return None, None
    lm = results.hand_landmarks[0]
    kp = np.array([[l.x, l.y, l.z] for l in lm], dtype=np.float32).flatten()
    wrist = kp[:3].copy()
    kp_norm = kp.copy()
    for i in range(21):
        kp_norm[i * 3 : (i + 1) * 3] -= wrist
    return kp_norm, lm


def annotate_image(image_rgb: np.ndarray, lm_list) -> np.ndarray:
    annotated = image_rgb.copy()
    h, w = annotated.shape[:2]
    pts = [(int(l.x * w), int(l.y * h)) for l in lm_list]
    connections = [
        (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
    ]
    for a, b in connections:
        cv2.line(annotated, pts[a], pts[b], (255, 220, 0), 2)
    for pt in pts:
        cv2.circle(annotated, pt, 4, (0, 150, 255), -1)
    return annotated


# ─── Prediction function ─────────────────────────────────────────────────────
def predict(image_rgb: np.ndarray, config, mlp, yolo, scaler, le, device, landmarker):
    classes = config["classes"]
    num_classes = config["num_classes"]

    # ── YOLOv11 prediction
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        Image.fromarray(image_rgb).save(tmp.name)
        yolo_pred = yolo.predict(tmp.name, imgsz=224, verbose=False)
    yolo_probs_raw = yolo_pred[0].probs.data.cpu().numpy()
    yolo_names = [yolo_pred[0].names[i] for i in range(len(yolo_pred[0].names))]
    yolo_probs = np.zeros(num_classes)
    for i, cls in enumerate(le.classes_):
        if cls in yolo_names:
            yolo_probs[i] = yolo_probs_raw[yolo_names.index(cls)]

    # ── MediaPipe + MLP prediction
    kp, hand_lm = extract_keypoints(image_rgb, landmarker)
    mlp_probs = np.zeros(num_classes)
    if kp is not None:
        kp_scaled = scaler.transform(kp.reshape(1, -1))
        kp_tensor = torch.tensor(kp_scaled, dtype=torch.float32).to(device)
        with torch.no_grad():
            logits = mlp(kp_tensor)
            mlp_probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

    # ── Hybrid soft vote
    if kp is not None:
        hybrid_probs = 0.5 * mlp_probs + 0.5 * yolo_probs
    else:
        hybrid_probs = yolo_probs  # fallback if no hand detected

    top_idx = int(np.argmax(hybrid_probs))
    top_class = le.inverse_transform([top_idx])[0]
    confidence = float(hybrid_probs[top_idx])

    # Top-3
    top3_idx = np.argsort(hybrid_probs)[::-1][:3]
    top3 = [(le.inverse_transform([i])[0], float(hybrid_probs[i])) for i in top3_idx]

    return top_class, confidence, top3, hand_lm, mlp_probs, yolo_probs


# ─── UI ───────────────────────────────────────────────────────────────────────
st.title("🤟 Real-Time BIM Sign Recognition")
st.markdown(
    "**Bahasa Isyarat Malaysia (BIM)** recognition using a YOLOv11 + MediaPipe hybrid model.  \n"
    "Upload a hand sign image to get a prediction."
)
st.divider()

config = load_config()

if config is None:
    st.error(
        "⚠️ Models not found. Please run the `BIM_Recognition.ipynb` notebook first "
        "to train and save all models, then restart this app."
    )
    st.stop()

mlp, yolo, scaler, le, device = load_models(config)
landmarker_path = config.get("hand_landmarker_path", "models/hand_landmarker.task")
landmarker = load_landmarker(landmarker_path)
st.success(f"✅ Models loaded — {config['num_classes']} BIM classes | Device: {device}")

# ── Input mode tabs
tab1, tab2 = st.tabs(["📷 Webcam", "🖼️ Upload Image"])

def show_results(image_rgb):
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input")
        st.image(image_rgb, use_container_width=True)

    with st.spinner("Predicting..."):
        top_class, confidence, top3, hand_lm, mlp_probs, yolo_probs = predict(
            image_rgb, config, mlp, yolo, scaler, le, device, landmarker
        )

    if hand_lm:
        annotated = annotate_image(image_rgb, hand_lm)
        with col2:
            st.subheader("Hand Landmarks")
            st.image(annotated, use_container_width=True)
    else:
        with col2:
            st.subheader("Hand Landmarks")
            st.warning("No hand detected — using YOLOv11 only.")

    st.divider()
    st.markdown(f"## Prediction: **{top_class}**")
    st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")

    st.subheader("Top 3 Predictions")
    for rank, (cls, prob) in enumerate(top3, 1):
        st.progress(prob, text=f"#{rank} {cls} — {prob * 100:.1f}%")

    return top_class, mlp_probs, yolo_probs

# ── Shared state for real-time prediction label
if "rt_label" not in st.session_state:
    st.session_state.rt_label = "—"
if "rt_conf" not in st.session_state:
    st.session_state.rt_conf = 0.0

# ── Video frame callback (runs on every webcam frame)
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)  # un-mirror horizontally
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    try:
        # MediaPipe keypoints
        kp, lm_list = extract_keypoints(img_rgb, landmarker)

        # YOLO prediction
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, img)
            yolo_pred = yolo.predict(tmp.name, imgsz=224, verbose=False)
        yolo_probs_raw = yolo_pred[0].probs.data.cpu().numpy()
        yolo_names = [yolo_pred[0].names[i] for i in range(len(yolo_pred[0].names))]
        yolo_probs = np.zeros(config["num_classes"])
        for i, cls in enumerate(le.classes_):
            if cls in yolo_names:
                yolo_probs[i] = yolo_probs_raw[yolo_names.index(cls)]

        # MLP prediction
        mlp_probs = np.zeros(config["num_classes"])
        if kp is not None:
            kp_scaled = scaler.transform(kp.reshape(1, -1))
            kp_tensor = torch.tensor(kp_scaled, dtype=torch.float32).to(device)
            with torch.no_grad():
                mlp_probs = torch.softmax(mlp(kp_tensor), dim=1).cpu().numpy()[0]

        # Ensemble
        hybrid = 0.5 * mlp_probs + 0.5 * yolo_probs if kp is not None else yolo_probs
        top_idx = int(np.argmax(hybrid))
        top_class = le.inverse_transform([top_idx])[0]
        confidence = float(hybrid[top_idx])
        st.session_state.rt_label = top_class
        st.session_state.rt_conf = confidence

        # Draw landmarks on frame
        if lm_list:
            h, w = img.shape[:2]
            pts = [(int(l.x * w), int(l.y * h)) for l in lm_list]
            connections = [
                (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),
                (0,9),(9,10),(10,11),(11,12),(0,13),(13,14),(14,15),(15,16),
                (0,17),(17,18),(18,19),(19,20),(5,9),(9,13),(13,17)
            ]
            for a, b in connections:
                cv2.line(img, pts[a], pts[b], (255, 220, 0), 2)
            for pt in pts:
                cv2.circle(img, pt, 4, (0, 150, 255), -1)

        # Draw prediction overlay on frame
        label_text = f"{top_class}  {confidence*100:.0f}%"
        cv2.rectangle(img, (0, 0), (300, 50), (0, 0, 0), -1)
        cv2.putText(img, label_text, (10, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 100), 3)

    except Exception:
        pass

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# ── Tab 1: Real-time webcam
with tab1:
    st.markdown("Show your BIM hand sign to the camera — predictions update in real time.")
    webrtc_streamer(
        key="bim-realtime",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
        video_frame_callback=video_frame_callback,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
    st.markdown(f"### Prediction: **{st.session_state.rt_label}**")
    if st.session_state.rt_conf > 0:
        st.progress(st.session_state.rt_conf, text=f"Confidence: {st.session_state.rt_conf*100:.1f}%")

# ── Tab 2: Upload
with tab2:
    uploaded = st.file_uploader("Upload a BIM hand sign image", type=["jpg", "jpeg", "png"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        image_rgb = np.array(image)
        top_class, mlp_probs, yolo_probs = show_results(image_rgb)
        with st.expander("Model breakdown"):
            st.markdown("**MLP (MediaPipe keypoints):**")
            if np.any(mlp_probs > 0):
                st.write(f"`{le.inverse_transform([np.argmax(mlp_probs)])[0]}` ({mlp_probs.max()*100:.1f}%)")
            else:
                st.write("No hand detected.")
            st.markdown("**YOLOv11:**")
            st.write(f"`{le.inverse_transform([np.argmax(yolo_probs)])[0]}` ({yolo_probs.max()*100:.1f}%)")
            st.markdown("**Ensemble:** 50% MLP + 50% YOLOv11 soft vote")

st.caption(
        "BICS 4340 Machine Learning | Group G | "
        "Iman Zafri · Muhammad Aizam · Adam Fawwaz · Muhammad Iman Imtiyaz"
    )

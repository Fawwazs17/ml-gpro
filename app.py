"""
BIM Recognition System — Streamlit Deployment App
BICS 4340 Machine Learning | Group G

Run with:  streamlit run app.py
"""

import json
import tempfile
from pathlib import Path

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
landmarker = load_landmarker(config["hand_landmarker_path"])
st.success(f"✅ Models loaded — {config['num_classes']} BIM classes | Device: {device}")

# ── Upload
uploaded = st.file_uploader(
    "Upload a BIM hand sign image", type=["jpg", "jpeg", "png"]
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    image_rgb = np.array(image)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Input Image")
        st.image(image, use_column_width=True)

    with st.spinner("Predicting..."):
        top_class, confidence, top3, hand_lm, mlp_probs, yolo_probs = predict(
            image_rgb, config, mlp, yolo, scaler, le, device, landmarker
        )

    # Annotated image
    if hand_lm:
        annotated = annotate_image(image_rgb, hand_lm)
        with col2:
            st.subheader("Hand Landmarks (MediaPipe)")
            st.image(annotated, use_column_width=True)
    else:
        with col2:
            st.subheader("Hand Landmarks")
            st.warning("No hand detected by MediaPipe — using YOLOv11 only.")

    st.divider()

    # Result
    st.markdown(f"## Prediction: **{top_class}**")
    st.progress(confidence, text=f"Confidence: {confidence * 100:.1f}%")

    # Top 3
    st.subheader("Top 3 Predictions")
    for rank, (cls, prob) in enumerate(top3, 1):
        st.progress(prob, text=f"#{rank} {cls} — {prob * 100:.1f}%")

    # Model breakdown
    with st.expander("Model breakdown"):
        st.markdown("**MLP (MediaPipe keypoints) — top prediction:**")
        if np.any(mlp_probs > 0):
            mlp_top = le.inverse_transform([np.argmax(mlp_probs)])[0]
            st.write(f"`{mlp_top}` ({mlp_probs.max() * 100:.1f}%)")
        else:
            st.write("No hand detected — MLP not used.")

        st.markdown("**YOLOv11 — top prediction:**")
        yolo_top = le.inverse_transform([np.argmax(yolo_probs)])[0]
        st.write(f"`{yolo_top}` ({yolo_probs.max() * 100:.1f}%)")

        st.markdown("**Ensemble:** 50% MLP + 50% YOLOv11 soft vote")

    st.caption(
        "BICS 4340 Machine Learning | Group G | "
        "Iman Zafri · Muhammad Aizam · Adam Fawwaz · Muhammad Iman Imtiyaz"
    )

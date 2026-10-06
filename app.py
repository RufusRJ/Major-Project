import os
import sys
import json
import io
import time
import tempfile
import numpy as np
from http.server import HTTPServer, BaseHTTPRequestHandler
from PIL import Image
from email.parser import BytesParser
from email.policy import default

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as tvm
import timm
import cv2
import librosa
from transformers import WavLMModel, Wav2Vec2FeatureExtractor, AutoFeatureExtractor, AutoModelForAudioClassification

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "shared_config.json")

# Load Shared Config
if os.path.exists(CONFIG_PATH):
    with open(CONFIG_PATH, "r") as f:
        CFG = json.load(f)
else:
    CFG = {
        "image": {"img_size": 380, "backbone": "tf_efficientnet_b4_ns"},
        "video": {"num_frames": 16, "frame_size": 224, "cnn_feature_dim": 512, "lstm_hidden": 256},
        "audio": {"sample_rate": 16000, "max_audio_seconds": 4, "wavlm_checkpoint": "microsoft/wavlm-base-plus"}
    }

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[Init] Using Device: {device}")

# ==============================================================================
# Model Architecture Definitions (matching trained notebooks)
# ==============================================================================

class EfficientNetB4Detector(nn.Module):
    def __init__(self, backbone_name=CFG["image"]["backbone"], num_classes=2):
        super().__init__()
        self.backbone = timm.create_model(backbone_name, pretrained=False, num_classes=0)
        self.feature_dim = self.backbone.num_features
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def extract_features(self, x):
        return self.backbone(x)

    def forward(self, x):
        feats = self.extract_features(x)
        return self.classifier(feats)

class DirectMLLSTM(nn.Module):
    """Pure PyTorch LSTM cell — runs on DirectML GPU or CPU without fused C++ ops."""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)

    def forward(self, x):
        B, T, C = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        c = torch.zeros(B, self.hidden_size, device=x.device)
        for t in range(T):
            combined = torch.cat([x[:, t, :], h], dim=1)
            gates = self.gates(combined)
            i, f, g, o = gates.chunk(4, dim=1)
            i, f, o = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c + i * g
            h = o * torch.tanh(c)
        return h

class CNNLSTMDetector(nn.Module):
    def __init__(self, cnn_feature_dim=CFG["video"]["cnn_feature_dim"], lstm_hidden=CFG["video"]["lstm_hidden"], num_classes=2):
        super().__init__()
        resnet = tvm.resnet18(weights=None)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        for p in self.cnn.parameters():
            p.requires_grad = False
        self.cnn_out_dim = resnet.fc.in_features
        self.lstm = DirectMLLSTM(self.cnn_out_dim, lstm_hidden)
        self.feature_dim = lstm_hidden
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def extract_features(self, x):
        B, T, C, H, W = x.shape
        feats = []
        for t in range(T):
            f_t = self.cnn(x[:, t]).flatten(1)
            feats.append(f_t)
        frame_feats = torch.stack(feats, dim=1)
        return self.lstm(frame_feats)

    def forward(self, x):
        feats = self.extract_features(x)
        return self.classifier(feats)

class ModalityProjection(nn.Module):
    def __init__(self, in_dim, proj_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self, x):
        return self.net(x)

class MultimodalFusionNetwork(nn.Module):
    def __init__(self, img_dim=1792, vid_dim=256, aud_dim=768, proj_dim=256, num_classes=2):
        super().__init__()
        self.proj_img = ModalityProjection(img_dim, proj_dim)
        self.proj_vid = ModalityProjection(vid_dim, proj_dim)
        self.proj_aud = ModalityProjection(aud_dim, proj_dim)
        
        self.attn_gate = nn.Sequential(
            nn.Linear(proj_dim * 3, 64),
            nn.ReLU(),
            nn.Linear(64, 3),
            nn.Softmax(dim=-1)
        )
        
        fused_dim = proj_dim * 3
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )
        
    def extract_fusion_embedding(self, f_img, f_vid, f_aud):
        p_img = self.proj_img(f_img)
        p_vid = self.proj_vid(f_vid)
        p_aud = self.proj_aud(f_aud)
        
        concat_feat = torch.cat([p_img, p_vid, p_aud], dim=-1)
        attn_weights = self.attn_gate(concat_feat)
        
        p_img_w = p_img * attn_weights[:, 0:1]
        p_vid_w = p_vid * attn_weights[:, 1:2]
        p_aud_w = p_aud * attn_weights[:, 2:3]
        
        fused = torch.cat([p_img_w, p_vid_w, p_aud_w], dim=-1)
        return fused, attn_weights
        
    def forward(self, f_img, f_vid, f_aud):
        fused, attn_weights = self.extract_fusion_embedding(f_img, f_vid, f_aud)
        return self.classifier(fused)

# ==============================================================================
# Model Loading & Initialization
# ==============================================================================

print("[Init] Loading Image Model (EfficientNet-B4)...")
img_model = EfficientNetB4Detector().to(device)
img_ckpt_path = os.path.join(MODELS_DIR, "efficientnet_b4_image.pt")
if os.path.exists(img_ckpt_path):
    ckpt = torch.load(img_ckpt_path, map_location=device, weights_only=False)
    img_model.load_state_dict(ckpt["model_state_dict"])
    print("  -> Image model loaded successfully from models/efficientnet_b4_image.pt")
img_model.eval()

print("[Init] Loading Video Model (CNN-LSTM)...")
vid_model = CNNLSTMDetector().to(device)
vid_ckpt_path = os.path.join(MODELS_DIR, "cnn_lstm_video.pt")
if os.path.exists(vid_ckpt_path):
    ckpt = torch.load(vid_ckpt_path, map_location=device, weights_only=False)
    vid_model.load_state_dict(ckpt["model_state_dict"])
    print("  -> Video model loaded successfully from models/cnn_lstm_video.pt")
vid_model.eval()

print("[Init] Loading Audio Model (DavidCombei/wavLM-base-Deepfake_V2)...")
WAVLM_CKPT = "DavidCombei/wavLM-base-Deepfake_V2"
audio_processor = AutoFeatureExtractor.from_pretrained(WAVLM_CKPT)
aud_model = AutoModelForAudioClassification.from_pretrained(WAVLM_CKPT)

aud_ckpt_path = os.path.join(MODELS_DIR, "wavlm_audio.pt")
if os.path.exists(aud_ckpt_path):
    try:
        ckpt = torch.load(aud_ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt.get("model_state_dict", ckpt)
        aud_model.load_state_dict(sd, strict=False)
        print("  -> Audio model loaded successfully from models/wavlm_audio.pt")
    except Exception as e:
        print(f"  -> Audio model loaded directly from HuggingFace ({e})")
aud_model = aud_model.to(device)
aud_model.eval()

print("[Init] Loading Multimodal Fusion Model...")
fusion_model = MultimodalFusionNetwork().to(device)
fusion_ckpt_path = os.path.join(MODELS_DIR, "multimodal_fusion.pt")
if os.path.exists(fusion_ckpt_path):
    ckpt = torch.load(fusion_ckpt_path, map_location=device, weights_only=False)
    fusion_model.load_state_dict(ckpt["model_state_dict"])
    print("  -> Fusion model loaded successfully from models/multimodal_fusion.pt (Phase 5 Active!)")
fusion_model.eval()

# Image Preprocessing Transforms
IMG_SIZE = CFG["image"]["img_size"]
img_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Video Preprocessing Transforms
NUM_FRAMES = CFG["video"]["num_frames"]
FRAME_SIZE = CFG["video"]["frame_size"]
frame_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((FRAME_SIZE, FRAME_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Audio Preprocessing Constants
SAMPLE_RATE = CFG["audio"]["sample_rate"]
MAX_SAMPLES = SAMPLE_RATE * CFG["audio"]["max_audio_seconds"]

# ==============================================================================
# Helper Inference Functions
# ==============================================================================

def predict_image(image_bytes):
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception:
        raise ValueError("Invalid image file format. Please upload a valid JPG, PNG, or WEBP image file.")

    tensor = img_transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = img_model.extract_features(tensor)
        logits = img_model.classifier(features)
        probs = torch.softmax(logits, dim=1)[0]
    
    real_prob = float(probs[0].item())
    fake_prob = float(probs[1].item())
    feat_sample = features[0][:20].cpu().numpy().tolist()
    
    return {
        "modality": "Image (EfficientNet-B4)",
        "prediction": "DEEPFAKE" if fake_prob > 0.5 else "REAL",
        "confidence": round(max(real_prob, fake_prob) * 100, 2),
        "probabilities": {"real": round(real_prob, 4), "fake": round(fake_prob, 4)},
        "feature_dim": features.shape[1],
        "feature_sample": feat_sample
    }

def predict_video(video_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Invalid video file format. Unable to open file stream.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, first_frame = cap.read()
        if not ret or first_frame is None:
            cap.release()
            raise ValueError("Invalid video file format or unreadable video codec. Please upload a valid MP4, AVI, or MOV video file.")

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        if total_frames <= 0:
            total_frames = 30

        indices = np.linspace(0, max(total_frames - 1, 0), NUM_FRAMES).astype(int)
        frames = []
        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_transform(frame))
        cap.release()

        if len(frames) == 0:
            raise ValueError("No valid video frames could be extracted. Please upload a valid MP4/AVI/MOV video file.")
            
        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1])
            
        tensor = torch.stack(frames[:NUM_FRAMES]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            features = vid_model.extract_features(tensor)
            logits = vid_model.classifier(features)
            probs = torch.softmax(logits, dim=1)[0]
            
        real_prob = float(probs[0].item())
        fake_prob = float(probs[1].item())
        feat_sample = features[0][:20].cpu().numpy().tolist()
        
        return {
            "modality": "Video (CNN-LSTM)",
            "prediction": "DEEPFAKE" if fake_prob > 0.5 else "REAL",
            "confidence": round(max(real_prob, fake_prob) * 100, 2),
            "probabilities": {"real": round(real_prob, 4), "fake": round(fake_prob, 4)},
            "feature_dim": features.shape[1],
            "feature_sample": feat_sample
        }
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def predict_audio(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        try:
            wav, _ = librosa.load(tmp_path, sr=SAMPLE_RATE, mono=True)
        except Exception:
            raise ValueError("Invalid audio file format. Please upload a valid WAV, MP3, or FLAC audio file.")

        if len(wav) == 0:
            raise ValueError("Audio file is empty or unreadable.")

        if len(wav) > MAX_SAMPLES:
            wav = wav[:MAX_SAMPLES]
        else:
            wav = np.pad(wav, (0, MAX_SAMPLES - len(wav)))
        wav = wav.astype(np.float32)
        
        inputs = audio_processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        
        with torch.no_grad():
            outputs = aud_model(input_values, output_hidden_states=True)
            logits = outputs.logits
            probs = torch.softmax(logits, dim=1)[0]
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                features = outputs.hidden_states[-1].mean(dim=1)
            else:
                features = torch.zeros(1, 768, device=device)

        # DavidCombei mapping: Index 0 = FAKE, Index 1 = REAL
        fake_prob = float(probs[0].item())
        real_prob = float(probs[1].item())
        feat_sample = features[0][:20].cpu().numpy().tolist()

        return {
            "modality": "Audio (DavidCombei WavLM Deepfake V2)",
            "prediction": "DEEPFAKE" if fake_prob > 0.5 else "REAL",
            "confidence": round(max(real_prob, fake_prob) * 100, 2),
            "probabilities": {"real": round(real_prob, 4), "fake": round(fake_prob, 4)},
            "feature_dim": 768,
            "feature_sample": feat_sample
        }
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass

def predict_multimodal(video_bytes, audio_bytes=None):
    tmp_vid_path = None
    tmp_aud_path = None
    
    try:
        if not video_bytes or len(video_bytes) == 0:
            raise ValueError("Video file stream is required for Multimodal Feature Fusion analysis.")
            
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_v:
            tmp_v.write(video_bytes)
            tmp_vid_path = tmp_v.name

        # 1. Extract Video Frames & Features (F_video)
        cap = cv2.VideoCapture(tmp_vid_path)
        if not cap.isOpened():
            raise ValueError("Invalid video file format or unreadable stream.")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            total_frames = 30

        indices = np.linspace(0, max(total_frames - 1, 0), NUM_FRAMES).astype(int)
        raw_frames_pil = []
        frames = []

        for i in range(total_frames):
            ret, frame = cap.read()
            if not ret:
                break
            if i in indices:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(frame_rgb)
                raw_frames_pil.append(pil_img)
                frames.append(frame_transform(frame_rgb))
        cap.release()

        if len(frames) == 0:
            raise ValueError("No valid video frames could be extracted from video.")

        while len(frames) < NUM_FRAMES:
            frames.append(frames[-1])
            if len(raw_frames_pil) < NUM_FRAMES:
                raw_frames_pil.append(raw_frames_pil[-1])

        vid_tensor = torch.stack(frames[:NUM_FRAMES]).unsqueeze(0).to(device)
        with torch.no_grad():
            f_vid = vid_model.extract_features(vid_tensor)
            vid_logits = vid_model.classifier(f_vid)
            vid_probs = torch.softmax(vid_logits, dim=1)[0]
            vid_fake_p = float(vid_probs[1].item())

        # 2. Keyframe Image Representation (F_image extracted from middle video frame)
        img_pil = raw_frames_pil[len(raw_frames_pil) // 2]
        img_tensor = img_transform(img_pil).unsqueeze(0).to(device)
        with torch.no_grad():
            f_img = img_model.extract_features(img_tensor)

        # 3. Extract Audio Features (F_audio)
        wav = None
        if audio_bytes and len(audio_bytes) > 0:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp_a:
                tmp_a.write(audio_bytes)
                tmp_aud_path = tmp_a.name
            try:
                wav, _ = librosa.load(tmp_aud_path, sr=SAMPLE_RATE, mono=True)
            except Exception:
                wav = None

        if wav is None:
            try:
                wav, _ = librosa.load(tmp_vid_path, sr=SAMPLE_RATE, mono=True)
            except Exception:
                wav = np.zeros(MAX_SAMPLES, dtype=np.float32)

        if len(wav) == 0:
            wav = np.zeros(MAX_SAMPLES, dtype=np.float32)
        elif len(wav) > MAX_SAMPLES:
            wav = wav[:MAX_SAMPLES]
        else:
            wav = np.pad(wav, (0, MAX_SAMPLES - len(wav)))
        wav = wav.astype(np.float32)

        inputs = audio_processor(wav, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            outputs = aud_model(input_values, output_hidden_states=True)
            aud_logits = outputs.logits
            aud_probs = torch.softmax(aud_logits, dim=1)[0]
            aud_fake_p = float(aud_probs[0].item()) # Index 0 = FAKE in DavidCombei
            if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
                f_aud = outputs.hidden_states[-1].mean(dim=1)
            else:
                f_aud = torch.zeros(1, 768, device=device)

        # 4. Multimodal Feature Fusion Model Inference
        with torch.no_grad():
            fused_emb, attn_weights = fusion_model.extract_fusion_embedding(f_img, f_vid, f_aud)
            fusion_logits = fusion_model.classifier(fused_emb)
            fusion_probs = torch.softmax(fusion_logits, dim=1)[0]

        raw_real_prob = float(fusion_probs[0].item())
        raw_fake_prob = float(fusion_probs[1].item())

        # Forensic Security Gating Rule:
        # In media forensics, if ANY individual stream (Video or Audio) is detected as DEEPFAKE (> 50%),
        # the overall file is compromised and MUST be flagged as DEEPFAKE.
        if vid_fake_p > 0.5 or aud_fake_p > 0.5:
            fake_prob = max(raw_fake_prob, vid_fake_p if vid_fake_p > 0.5 else 0.0, aud_fake_p if aud_fake_p > 0.5 else 0.0)
            real_prob = 1.0 - fake_prob
        else:
            real_prob = raw_real_prob
            fake_prob = raw_fake_prob

        w_img = float(attn_weights[0, 0].item())
        w_vid = float(attn_weights[0, 1].item())
        w_aud = float(attn_weights[0, 2].item())

        w_video_comb = w_img + w_vid
        total_w = w_video_comb + w_aud + 1e-6
        pct_video = round((w_video_comb / total_w) * 100, 1)
        pct_audio = round((w_aud / total_w) * 100, 1)

        feat_sample = fused_emb[0][:20].cpu().numpy().tolist()

        return {
            "modality": "Multimodal Feature Fusion (Video + Audio)",
            "prediction": "DEEPFAKE" if fake_prob > 0.5 else "REAL",
            "confidence": round(max(real_prob, fake_prob) * 100, 2),
            "probabilities": {"real": round(real_prob, 4), "fake": round(fake_prob, 4)},
            "attention_weights": {
                "video": pct_video,
                "audio": pct_audio
            },
            "branch_predictions": {
                "video": {"prediction": "DEEPFAKE" if vid_fake_p > 0.5 else "REAL", "confidence": round(max(vid_fake_p, 1-vid_fake_p)*100, 1)},
                "audio": {"prediction": "DEEPFAKE" if aud_fake_p > 0.5 else "REAL", "confidence": round(max(aud_fake_p, 1-aud_fake_p)*100, 1)}
            },
            "feature_dim": fused_emb.shape[1],
            "feature_sample": feat_sample
        }
    finally:
        if tmp_vid_path and os.path.exists(tmp_vid_path):
            try:
                os.remove(tmp_vid_path)
            except Exception:
                pass
        if tmp_aud_path and os.path.exists(tmp_aud_path):
            try:
                os.remove(tmp_aud_path)
            except Exception:
                pass

def parse_multipart(file_bytes, content_type_header):
    try:
        raw_mime = f"Content-Type: {content_type_header}\r\n\r\n".encode("utf-8") + file_bytes
        msg = BytesParser(policy=default).parsebytes(raw_mime)
        files = {}
        for part in msg.iter_parts():
            cd = part.get("Content-Disposition", "")
            if "name=" in cd:
                name = part.get_param("name", header="content-disposition")
                filename = part.get_filename()
                payload = part.get_payload(decode=True)
                if name and payload:
                    files[name] = {"filename": filename, "content": payload}
        return files
    except Exception as e:
        print(f"[Warning] Multipart parsing error: {e}")
        return {}

# ==============================================================================
# HTML Front-End Template
# ==============================================================================

HTML_CONTENT = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Multimodal Deepfake Forensic Workbench</title>
    <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-dark: #090d16;
            --card-bg: rgba(18, 26, 44, 0.75);
            --border-color: rgba(255, 255, 255, 0.08);
            --accent-blue: #00d2ff;
            --accent-purple: #7000ff;
            --accent-pink: #ff007f;
            --text-main: #f0f4f8;
            --text-muted: #8a99ad;
            --real-green: #00e676;
            --fake-red: #ff1744;
        }

        * { box-sizing: border-box; margin: 0; padding: 0; font-family: 'Outfit', sans-serif; }
        body { background: var(--bg-dark); color: var(--text-main); min-height: 100vh; overflow-x: hidden; }

        .background-glow {
            position: fixed; top: 0; left: 0; width: 100vw; height: 100vh; pointer-events: none; z-index: 0;
            background: radial-gradient(circle at 20% 20%, rgba(112, 0, 255, 0.15) 0%, transparent 40%),
                        radial-gradient(circle at 80% 80%, rgba(0, 210, 255, 0.15) 0%, transparent 40%);
        }

        .app-container { max-width: 1280px; margin: 0 auto; padding: 2rem 1.5rem; position: relative; z-index: 1; }

        header {
            display: flex; justify-content: space-between; align-items: center; margin-bottom: 2.5rem;
            padding-bottom: 1.5rem; border-bottom: 1px solid var(--border-color);
        }
        .logo-title h1 { font-size: 1.8rem; font-weight: 700; background: linear-gradient(135deg, #00d2ff, #00e676); -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
        .logo-title p { color: var(--text-muted); font-size: 0.9rem; margin-top: 4px; }
        
        .status-badge {
            background: rgba(0, 230, 118, 0.1); border: 1px solid rgba(0, 230, 118, 0.3);
            color: var(--real-green); padding: 6px 14px; border-radius: 20px; font-size: 0.85rem; font-weight: 500;
            display: flex; align-items: center; gap: 8px;
        }
        .pulse-dot { width: 8px; height: 8px; background: var(--real-green); border-radius: 50%; box-shadow: 0 0 10px var(--real-green); animation: pulse 1.5s infinite; }
        @keyframes pulse { 0%, 100% { opacity: 1; transform: scale(1); } 50% { opacity: 0.4; transform: scale(1.2); } }

        /* Navigation Tabs */
        .modality-tabs { display: flex; gap: 0.8rem; margin-bottom: 2rem; background: rgba(255,255,255,0.03); padding: 6px; border-radius: 12px; border: 1px solid var(--border-color); flex-wrap: wrap; }
        .tab-btn {
            flex: 1; min-width: 180px; padding: 14px 16px; border: none; background: transparent; color: var(--text-muted);
            font-size: 0.95rem; font-weight: 600; cursor: pointer; border-radius: 8px; transition: all 0.3s ease;
            display: flex; align-items: center; justify-content: center; gap: 8px; text-align: center;
        }
        .tab-btn:hover { color: #fff; background: rgba(255,255,255,0.05); }
        .tab-btn.active {
            background: linear-gradient(135deg, rgba(0, 210, 255, 0.2), rgba(112, 0, 255, 0.2));
            color: #fff; border: 1px solid rgba(0, 210, 255, 0.4); box-shadow: 0 4px 20px rgba(0, 210, 255, 0.15);
        }
        .tab-btn.fusion-tab.active {
            background: linear-gradient(135deg, rgba(255, 0, 127, 0.25), rgba(112, 0, 255, 0.25));
            border-color: rgba(255, 0, 127, 0.5); box-shadow: 0 4px 20px rgba(255, 0, 127, 0.2);
        }

        /* Workspace Grid */
        .workspace-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
        @media (max-width: 900px) { .workspace-grid { grid-template-columns: 1fr; } }

        .card {
            background: var(--card-bg); backdrop-filter: blur(12px); border: 1px solid var(--border-color);
            border-radius: 16px; padding: 1.8rem; transition: transform 0.3s ease;
        }
        .card-header { font-size: 1.1rem; font-weight: 600; margin-bottom: 1.2rem; color: #fff; display: flex; align-items: center; gap: 10px; }

        /* Upload Dropzone */
        .dropzone {
            border: 2px dashed rgba(255,255,255,0.15); border-radius: 12px; padding: 2.2rem 1rem;
            text-align: center; cursor: pointer; transition: all 0.3s ease; background: rgba(0,0,0,0.2);
            position: relative; overflow: hidden; margin-bottom: 1rem;
        }
        .dropzone:hover, .dropzone.dragover { border-color: var(--accent-blue); background: rgba(0, 210, 255, 0.05); }
        .dropzone input { display: none; }
        .dropzone-icon { font-size: 2.5rem; margin-bottom: 8px; opacity: 0.8; }
        .dropzone-text { font-size: 0.95rem; color: var(--text-muted); }
        .dropzone-text span { color: var(--accent-blue); font-weight: 500; }

        /* Additional Multimodal Dropzones */
        .optional-inputs { display: none; margin-top: 1rem; }
        .sub-input-label { font-size: 0.85rem; font-weight: 600; color: var(--text-muted); margin-bottom: 6px; display: block; }

        /* Preview Container */
        .preview-container { margin-top: 1rem; display: none; text-align: center; }
        .preview-container img, .preview-container video { max-width: 100%; max-height: 220px; border-radius: 8px; border: 1px solid var(--border-color); }
        .preview-container audio { width: 100%; margin-top: 10px; }

        /* Action Button */
        .btn-analyze {
            width: 100%; margin-top: 1.5rem; padding: 14px; border: none; border-radius: 10px;
            background: linear-gradient(135deg, var(--accent-blue), var(--accent-purple));
            color: #fff; font-size: 1rem; font-weight: 600; cursor: pointer; transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(0, 210, 255, 0.3); display: flex; align-items: center; justify-content: center; gap: 10px;
        }
        .btn-analyze.btn-fusion { background: linear-gradient(135deg, var(--accent-pink), var(--accent-purple)); box-shadow: 0 4px 15px rgba(255, 0, 127, 0.3); }
        .btn-analyze:hover { transform: translateY(-2px); box-shadow: 0 6px 20px rgba(0, 210, 255, 0.4); }
        .btn-analyze:disabled { opacity: 0.5; cursor: not-allowed; }

        /* Result Panel */
        .result-placeholder { display: flex; flex-direction: column; align-items: center; justify-content: center; height: 100%; color: var(--text-muted); font-size: 0.95rem; text-align: center; padding: 3rem 1rem; }

        .result-content { display: none; }

        /* Prediction Banner */
        .prediction-banner {
            display: flex; align-items: center; justify-content: space-between; padding: 1.2rem; border-radius: 12px;
            margin-bottom: 1.5rem; border: 1px solid var(--border-color);
        }
        .prediction-banner.REAL { background: rgba(0, 230, 118, 0.1); border-color: rgba(0, 230, 118, 0.4); }
        .prediction-banner.DEEPFAKE { background: rgba(255, 23, 68, 0.1); border-color: rgba(255, 23, 68, 0.4); }

        .pred-label { font-size: 1.3rem; font-weight: 700; }
        .REAL .pred-label { color: var(--real-green); }
        .DEEPFAKE .pred-label { color: var(--fake-red); }

        .conf-badge { font-size: 1.1rem; font-weight: 600; color: #fff; }

        /* Attention Weight Breakdown */
        .attention-box { background: rgba(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 10px; padding: 14px; margin-bottom: 1.5rem; display: none; }
        .attn-item { margin-bottom: 10px; }
        .attn-item:last-child { margin-bottom: 0; }
        .attn-header { display: flex; justify-content: space-between; font-size: 0.85rem; color: var(--text-main); margin-bottom: 4px; }
        .attn-bar-bg { background: rgba(255,255,255,0.06); height: 8px; border-radius: 4px; overflow: hidden; }
        .attn-bar-fill { height: 100%; border-radius: 4px; transition: width 0.6s ease; }
        .vid-fill { background: linear-gradient(90deg, #7000ff, #00d2ff); }
        .aud-fill { background: linear-gradient(90deg, #ff007f, #7000ff); }

        /* Individual Branch Consensus Grid (2 Columns: Video + Audio) */
        .branch-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 12px; margin-bottom: 1.5rem; display: none; }
        .branch-card { background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: 8px; padding: 12px; text-align: center; }
        .branch-title { font-size: 0.8rem; color: var(--text-muted); margin-bottom: 6px; font-weight: 500; }
        .branch-pred { font-size: 1rem; font-weight: 700; }

        /* Embedding Box */
        .embedding-box { background: rgba(0,0,0,0.4); padding: 12px; border-radius: 8px; border: 1px solid var(--border-color); }
        .embedding-title { font-size: 0.85rem; font-weight: 600; color: var(--text-muted); margin-bottom: 8px; display: flex; justify-content: space-between; }
        .embedding-bars { display: flex; gap: 3px; height: 35px; align-items: flex-end; }
        .bar { flex: 1; background: var(--accent-blue); border-radius: 2px; transition: height 0.4s ease; min-height: 4px; }

        .spinner {
            width: 20px; height: 20px; border: 3px solid rgba(255,255,255,0.3); border-top-color: #fff;
            border-radius: 50%; animation: spin 0.8s linear infinite; display: none;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="background-glow"></div>

    <div class="app-container">
        <header>
            <div class="logo-title">
                <h1>Multimodal Deepfake Forensic Workbench</h1>
                <p>Modality Evaluation: Image (EfficientNet-B4) | Video (CNN-LSTM) | Audio (WavLM) | Feature Fusion (Phase 5)</p>
            </div>
            <div class="status-badge">
                <div class="pulse-dot"></div>
                PyTorch & Phase 5 Active
            </div>
        </header>

        <!-- Navigation Modality Tabs -->
        <div class="modality-tabs">
            <button class="tab-btn active" onclick="switchModality('image', event)">
                🖼️ Image Branch (EfficientNet)
            </button>
            <button class="tab-btn" onclick="switchModality('video', event)">
                📹 Video Branch (CNN-LSTM)
            </button>
            <button class="tab-btn" onclick="switchModality('audio', event)">
                🎙️ Audio Branch (WavLM)
            </button>
            <button class="tab-btn fusion-tab" onclick="switchModality('multimodal', event)">
                ⚡ Multimodal Fusion (Video + Audio)
            </button>
        </div>

        <!-- Main Workspace -->
        <div class="workspace-grid">
            <!-- Left: Input & Upload Card -->
            <div class="card">
                <div class="card-header">
                    <span id="modality-icon">🖼️</span>
                    <span id="modality-title">Image Branch Input</span>
                </div>

                <!-- Primary Dropzone -->
                <div class="dropzone" id="dropzone" onclick="document.getElementById('file-input').click()">
                    <div class="dropzone-icon" id="dz-icon">📸</div>
                    <div class="dropzone-text" id="dz-text">
                        Drag & Drop media here or <span>Browse File</span>
                    </div>
                    <div class="dropzone-text" style="font-size:0.8rem; margin-top:6px;" id="file-spec-text">
                        Supports: JPG, PNG, WEBP (Max 20MB)
                    </div>
                    <input type="file" id="file-input" onchange="handleFileSelect(event)">
                </div>

                <!-- Secondary Optional Audio Dropzone for Multimodal Fusion -->
                <div class="optional-inputs" id="optional-inputs">
                    <div style="border-top:1px dashed var(--border-color); margin: 1rem 0; padding-top: 1rem;">
                        <span class="sub-input-label">🎙️ Optional Custom Audio Track (WAV, MP3, FLAC)</span>
                        <div class="dropzone" style="padding:1rem;" onclick="document.getElementById('aud-file-input').click()">
                            <div class="dropzone-text" id="aud-file-name">Click to select separate Audio track (If left empty, audio is automatically extracted from Video)</div>
                            <input type="file" id="aud-file-input" accept="audio/*" onchange="handleSubFileSelect(event, 'aud')">
                        </div>
                    </div>
                </div>

                <div class="preview-container" id="preview-container">
                    <img id="img-preview" src="" style="display:none;">
                    <video id="vid-preview" controls style="display:none;"></video>
                    <audio id="aud-preview" controls style="display:none;"></audio>
                </div>

                <button class="btn-analyze" id="btn-analyze" onclick="runInference()" disabled>
                    <div class="spinner" id="spinner"></div>
                    <span id="btn-text">Run Model Detection</span>
                </button>
            </div>

            <!-- Right: Diagnostics & Embedding Card -->
            <div class="card">
                <div class="card-header">
                    📊 Model Diagnostics & Feature Embeddings
                </div>

                <div class="result-placeholder" id="result-placeholder">
                    <div style="font-size:2.5rem; margin-bottom:10px;">🔬</div>
                    Upload media to execute model inference and extract feature vectors.
                </div>

                <div class="result-content" id="result-content">
                    <!-- Banner -->
                    <div class="prediction-banner" id="pred-banner">
                        <div>
                            <div style="font-size:0.8rem; color:var(--text-muted); text-transform:uppercase; letter-spacing:1px;">Classification Result</div>
                            <div class="pred-label" id="pred-label">REAL</div>
                        </div>
                        <div class="conf-badge" id="conf-badge">94.2% Confident</div>
                    </div>

                    <!-- Multimodal Cross-Attention Weighting (Phase 5: Video + Audio) -->
                    <div class="attention-box" id="attention-box">
                        <div style="font-size:0.85rem; font-weight:600; color:var(--text-muted); margin-bottom:10px; display:flex; justify-content:space-between;">
                            <span>⚡ Adaptive Attention Weighting (Phase 5)</span>
                            <span style="color:var(--accent-pink)">Video + Audio Fusion</span>
                        </div>
                        <div class="attn-item">
                            <div class="attn-header"><span>📹 Video Stream Attention (Visual)</span><span id="attn-vid-val">50.0%</span></div>
                            <div class="attn-bar-bg"><div class="attn-bar-fill vid-fill" id="attn-vid-bar" style="width:50.0%"></div></div>
                        </div>
                        <div class="attn-item">
                            <div class="attn-header"><span>🎙️ Audio Track Attention (Acoustic)</span><span id="attn-aud-val">50.0%</span></div>
                            <div class="attn-bar-bg"><div class="attn-bar-fill aud-fill" id="attn-aud-bar" style="width:50.0%"></div></div>
                        </div>
                    </div>

                    <!-- Individual Branch Predictions (Video & Audio) -->
                    <div class="branch-grid" id="branch-grid">
                        <div class="branch-card">
                            <div class="branch-title">📹 Video Branch (CNN-LSTM)</div>
                            <div class="branch-pred" id="b-vid-pred">REAL</div>
                            <div class="branch-title" id="b-vid-conf" style="margin-top:4px;">94.5%</div>
                        </div>
                        <div class="branch-card">
                            <div class="branch-title">🎙️ Audio Branch (WavLM)</div>
                            <div class="branch-pred" id="b-aud-pred">REAL</div>
                            <div class="branch-title" id="b-aud-conf" style="margin-top:4px;">88.0%</div>
                        </div>
                    </div>

                    <!-- Embedding Sample -->
                    <div class="embedding-box">
                        <div class="embedding-title">
                            <span id="emb-dim-label">Extracted Feature Vector (F_image)</span>
                            <span style="color:var(--accent-blue);" id="emb-dim-val">1792 Dim</span>
                        </div>
                        <div class="embedding-bars" id="embedding-bars">
                            <!-- JS populated bars -->
                        </div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <script>
        let currentModality = 'image';
        let currentFile = null;
        let optionalAudioFile = null;

        const specs = {
            image: {
                icon: '🖼️', title: 'Image Branch Input (EfficientNet-B4)', dzIcon: '📸',
                accept: 'image/*', spec: 'Supports: JPG, PNG, WEBP (Max 20MB)',
                endpoint: '/api/predict/image',
                embLabel: 'Extracted Feature Vector (F_image)',
                dzText: 'Drag & Drop Image here or <span>Browse File</span>'
            },
            video: {
                icon: '📹', title: 'Video Branch Input (CNN-LSTM)', dzIcon: '🎥',
                accept: 'video/*', spec: 'Supports: MP4, AVI, MOV (Max 50MB)',
                endpoint: '/api/predict/video',
                embLabel: 'Extracted Feature Vector (F_video)',
                dzText: 'Drag & Drop Video here or <span>Browse File</span>'
            },
            audio: {
                icon: '🎙️', title: 'Audio Branch Input (WavLM)', dzIcon: '🎵',
                accept: 'audio/*', spec: 'Supports: WAV, MP3, FLAC (Max 20MB)',
                endpoint: '/api/predict/audio',
                embLabel: 'Extracted Feature Vector (F_audio)',
                dzText: 'Drag & Drop Audio here or <span>Browse File</span>'
            },
            multimodal: {
                icon: '⚡', title: 'Multimodal Feature Fusion (Video + Audio)', dzIcon: '🎬',
                accept: 'video/*', spec: 'Primary Stream: MP4, AVI, MOV (Max 50MB)',
                endpoint: '/api/predict/multimodal',
                embLabel: 'Phase 5 Fused Feature Vector (F_fused)',
                dzText: 'Drag & Drop Primary Video here or <span>Browse File</span>'
            }
        };

        function switchModality(mod, evt) {
            currentModality = mod;
            currentFile = null;
            optionalAudioFile = null;
            document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            if (evt && evt.currentTarget) evt.currentTarget.classList.add('active');

            const s = specs[mod];
            document.getElementById('modality-icon').innerText = s.icon;
            document.getElementById('modality-title').innerText = s.title;
            document.getElementById('dz-icon').innerText = s.dzIcon;
            document.getElementById('file-spec-text').innerText = s.spec;
            document.getElementById('file-input').accept = s.accept;
            document.getElementById('dz-text').innerHTML = s.dzText;

            const optContainer = document.getElementById('optional-inputs');
            const analyzeBtn = document.getElementById('btn-analyze');

            if (mod === 'multimodal') {
                optContainer.style.display = 'block';
                analyzeBtn.classList.add('btn-fusion');
                document.getElementById('btn-text').innerText = 'Execute Multimodal Feature Fusion';
            } else {
                optContainer.style.display = 'none';
                analyzeBtn.classList.remove('btn-fusion');
                document.getElementById('btn-text').innerText = 'Run Model Detection';
            }

            // Reset previews & results
            document.getElementById('preview-container').style.display = 'none';
            document.getElementById('img-preview').style.display = 'none';
            document.getElementById('vid-preview').style.display = 'none';
            document.getElementById('aud-preview').style.display = 'none';
            document.getElementById('btn-analyze').disabled = true;

            document.getElementById('result-placeholder').style.display = 'flex';
            document.getElementById('result-content').style.display = 'none';
        }

        function handleFileSelect(e) {
            const file = e.target.files[0];
            if (!file) return;
            currentFile = file;

            const previewContainer = document.getElementById('preview-container');
            const imgP = document.getElementById('img-preview');
            const vidP = document.getElementById('vid-preview');
            const audP = document.getElementById('aud-preview');

            imgP.style.display = 'none';
            vidP.style.display = 'none';
            audP.style.display = 'none';
            previewContainer.style.display = 'block';

            const url = URL.createObjectURL(file);

            if (currentModality === 'image') {
                imgP.src = url; imgP.style.display = 'block';
            } else if (currentModality === 'video' || currentModality === 'multimodal') {
                vidP.src = url; vidP.style.display = 'block';
            } else if (currentModality === 'audio') {
                audP.src = url; audP.style.display = 'block';
            }

            document.getElementById('btn-analyze').disabled = false;
        }

        function handleSubFileSelect(e, type) {
            const file = e.target.files[0];
            if (!file) return;
            if (type === 'aud') {
                optionalAudioFile = file;
                document.getElementById('aud-file-name').innerText = '✅ Audio file attached: ' + file.name;
            }
        }

        async function runInference() {
            if (!currentFile) return;

            const btn = document.getElementById('btn-analyze');
            const spinner = document.getElementById('spinner');
            const btnText = document.getElementById('btn-text');

            btn.disabled = true;
            spinner.style.display = 'inline-block';
            btnText.innerText = currentModality === 'multimodal' ? 'Fusing Features & Evaluating Model...' : 'Extracting Features & Evaluating Model...';

            try {
                const endpoint = specs[currentModality].endpoint;
                let response;

                if (currentModality === 'multimodal') {
                    const formData = new FormData();
                    formData.append('video', currentFile);
                    if (optionalAudioFile) formData.append('audio', optionalAudioFile);

                    response = await fetch(endpoint, {
                        method: 'POST',
                        body: formData
                    });
                } else {
                    response = await fetch(endpoint, {
                        method: 'POST',
                        headers: { 'Content-Type': currentFile.type || 'application/octet-stream' },
                        body: currentFile
                    });
                }

                const data = await response.json();
                if (data.error) throw new Error(data.error);
                displayResults(data);
            } catch (err) {
                alert('Inference error: ' + err.message);
            } finally {
                btn.disabled = false;
                spinner.style.display = 'none';
                btnText.innerText = currentModality === 'multimodal' ? 'Execute Multimodal Feature Fusion' : 'Run Model Detection';
            }
        }

        function displayResults(data) {
            document.getElementById('result-placeholder').style.display = 'none';
            document.getElementById('result-content').style.display = 'block';

            const banner = document.getElementById('pred-banner');
            const label = document.getElementById('pred-label');
            const conf = document.getElementById('conf-badge');

            banner.className = 'prediction-banner ' + data.prediction;
            label.innerText = data.prediction === 'DEEPFAKE' ? '⚠️ DEEPFAKE DETECTED' : '✅ REAL / AUTHENTIC';
            conf.innerText = data.confidence + '% Confidence';

            const s = specs[currentModality];
            document.getElementById('emb-dim-label').innerText = s.embLabel;
            document.getElementById('emb-dim-val').innerText = data.feature_dim + ' Dim';

            const attnBox = document.getElementById('attention-box');
            const branchGrid = document.getElementById('branch-grid');

            if (currentModality === 'multimodal' && data.attention_weights) {
                attnBox.style.display = 'block';
                branchGrid.style.display = 'grid';

                const aw = data.attention_weights;
                document.getElementById('attn-vid-val').innerText = aw.video + '%';
                document.getElementById('attn-vid-bar').style.width = aw.video + '%';

                document.getElementById('attn-aud-val').innerText = aw.audio + '%';
                document.getElementById('attn-aud-bar').style.width = aw.audio + '%';

                if (data.branch_predictions) {
                    const bp = data.branch_predictions;
                    const elVid = document.getElementById('b-vid-pred');
                    elVid.innerText = bp.video.prediction;
                    elVid.style.color = bp.video.prediction === 'DEEPFAKE' ? 'var(--fake-red)' : 'var(--real-green)';
                    document.getElementById('b-vid-conf').innerText = bp.video.confidence + '% Confident';

                    const elAud = document.getElementById('b-aud-pred');
                    elAud.innerText = bp.audio.prediction;
                    elAud.style.color = bp.audio.prediction === 'DEEPFAKE' ? 'var(--fake-red)' : 'var(--real-green)';
                    document.getElementById('b-aud-conf').innerText = bp.audio.confidence + '% Confident';
                }
            } else {
                attnBox.style.display = 'none';
                branchGrid.style.display = 'none';
            }

            // Render feature bars
            const barsContainer = document.getElementById('embedding-bars');
            barsContainer.innerHTML = '';
            const sample = data.feature_sample || [];
            const minV = Math.min(...sample, 0);
            const maxV = Math.max(...sample, 1e-5);

            sample.forEach(val => {
                const norm = Math.max(10, Math.min(100, ((val - minV) / (maxV - minV + 1e-6)) * 100));
                const bar = document.createElement('div');
                bar.className = 'bar';
                bar.style.height = norm + '%';
                bar.title = 'Value: ' + val.toFixed(4);
                barsContainer.appendChild(bar);
            });
        }
    </script>
</body>
</html>
"""

# ==============================================================================
# HTTP Request Handler
# ==============================================================================

class ForensicRequestHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        print(f"[HTTP] {self.command} {self.path} -> {args[0]}")

    def do_GET(self):
        if self.path in ["/", "/index.html"]:
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(HTML_CONTENT.encode("utf-8"))
        else:
            self.send_error(404, "File Not Found")

    def do_POST(self):
        content_length = int(self.headers.get('Content-Length', 0))
        file_bytes = self.rfile.read(content_length)

        try:
            if self.path == "/api/predict/image":
                result = predict_image(file_bytes)
            elif self.path == "/api/predict/video":
                result = predict_video(file_bytes)
            elif self.path == "/api/predict/audio":
                result = predict_audio(file_bytes)
            elif self.path == "/api/predict/multimodal":
                ct = self.headers.get("Content-Type", "")
                if "multipart/form-data" in ct:
                    parsed_files = parse_multipart(file_bytes, ct)
                    vid_b = parsed_files.get("video", {}).get("content") or parsed_files.get("file", {}).get("content")
                    aud_b = parsed_files.get("audio", {}).get("content")
                    result = predict_multimodal(vid_b, aud_b)
                else:
                    result = predict_multimodal(file_bytes)
            else:
                self.send_error(404, "Unknown Endpoint")
                return

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(result).encode("utf-8"))

        except Exception as e:
            print(f"[Error] Predict error: {e}")
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"error": str(e)}).encode("utf-8"))

def run_server(port=5000):
    server_address = ('', port)
    httpd = HTTPServer(server_address, ForensicRequestHandler)
    print(f"\n==========================================================")
    print(f"[Server] Multimodal Deepfake Forensic Workbench Running!")
    print(f"[Server] Open Web UI at: http://localhost:{port}")
    print(f"==========================================================\n")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n[Server] Shutting down...")
        httpd.server_close()

if __name__ == "__main__":
    run_server(5000)

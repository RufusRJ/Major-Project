# Multimodal Deepfake Detection System (Feature Fusion Project)

An end-to-end Deepfake Detection Framework utilizing independent modality feature extractors (Image, Video, Audio) and feature fusion neural networks for robust, explainable forensic analysis.

---

## 📌 Project Overview

This project implements a multi-branch architecture for detecting deepfakes across different digital media modalities:

1. **Image Branch**: Pretrained **EfficientNet-B4** for spatial texture, edge, and compression artifact detection ($F_{\text{image}}$ vector, 1,792 dimensions).
2. **Video Branch**: **ResNet-18 CNN + LSTM** for frame-level spatial feature extraction and temporal relationship modeling ($F_{\text{video}}$ vector, 256 dimensions).
3. **Audio Branch**: Pretrained **WavLM** (`microsoft/wavlm-base-plus`) for utterance-level acoustic feature extraction and voice cloning detection ($F_{\text{audio}}$ vector, 768 dimensions).
4. **Feature Fusion Network** *(Upcoming Phase 4–5)*: Concatenates $F_{\text{image}} + F_{\text{video}} + F_{\text{audio}}$ and trains a lightweight neural network to learn cross-modal manipulation relationships.
5. **Interactive Testing Front-End**: Modern PyTorch-backed web interface to test each model independently with real-time prediction and feature vector visualization.

---

## 📥 Dataset Download & Setup (Step-by-Step for Group / Collaborators)

This repository includes an automated Kaggle dataset setup script (`download_datasets.py`) that downloads real & fake image/video datasets and organizes them directly into the required `data/` layout.

### Datasets Used:
- **Images**: `prithivsakthiur/deepfake-vs-real-20k`
- **Videos**: `xdxd003/ff-c23` (FaceForensics++ c23)

### How Team Members Set Up Datasets:

1. **Install Kagglehub**:
   ```bash
   pip install kagglehub
   ```

2. **(Optional) Configure Kaggle Credentials**:
   If Kaggle requires API authentication:
   - Go to [Kaggle Profile Settings](https://www.kaggle.com/settings) $\to$ **Create New API Token**.
   - Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json` (on Windows: `C:\Users\<YourUser>\.kaggle\kaggle.json`).

3. **Run Automated Dataset Download Script**:
   ```bash
   python download_datasets.py
   ```
   This will automatically download the datasets and populate:
   - `data/image/real/` and `data/image/fake/`
   - `data/video/real/` and `data/video/fake/`

---

## 📁 Repository Structure

```
Major_Project/
├── data/                         <- Dataset directory structure
│   ├── image/ (real/ fake/)      <- Image samples (.jpg, .png)
│   ├── video/ (real/ fake/)      <- Video samples (.mp4, .avi)
│   └── audio/ (real/ fake/)      <- Audio samples (.wav, .flac)
├── checkpoints/                  <- Saved PyTorch fine-tuned model checkpoints
├── models/                       <- Fine-tuned PyTorch model files (.pt)
│   ├── efficientnet_b4_image.pt  (72.8 MB)
│   ├── cnn_lstm_video.pt         (48.1 MB)
│   └── wavlm_audio.pt            (378.0 MB)
├── features/                     <- Extracted feature tensors for fusion network
├── shared_config.json            <- Shared configuration across all notebooks
├── download_datasets.py          <- Automated Kaggle dataset downloader
├── 00_setup_environment.ipynb    <- Phase 0: Scaffolding & Shared Config
├── 01_image_efficientnet_b4.ipynb<- Phase 1: EfficientNet-B4 Image Branch
├── 02_video_cnn_lstm.ipynb       <- Phase 2: CNN-LSTM Video Branch
├── 03_audio_wavlm.ipynb          <- Phase 3: WavLM Audio Branch
├── run_all_notebooks.py          <- Automated executor for all 4 notebooks
├── update_notebooks.py           <- Path configuration utility script
├── app.py                        <- Web Application (Python HTTP Server + Front-End UI)
├── README.md                     <- This execution & usage guide
└── PROJECT_PHASES_AND_DOCS.txt   <- Complete project documentation, completed & pending phases
```

---

## 🛠️ Requirements & Environment Setup

### 1. Prerequisites
- Python 3.10 or 3.11
- PyTorch 2.4+ (CPU or CUDA GPU)

### 2. Installation Command
Run the following in your shell:
```bash
pip install torch torchvision timm "transformers<4.46" librosa soundfile opencv-python scikit-learn matplotlib tqdm pandas pillow kagglehub
```

---

## 🚀 How to Train & Execute Notebooks

Once datasets are downloaded (`python download_datasets.py`):

### Option A: Run Individual Jupyter Notebooks
1. **Environment Setup**: Run `00_setup_environment.ipynb`.
2. **Image Branch**: Run `01_image_efficientnet_b4.ipynb` to train EfficientNet-B4.
3. **Video Branch**: Run `02_video_cnn_lstm.ipynb` to train CNN-LSTM.
4. **Audio Branch**: Run `03_audio_wavlm.ipynb` to train WavLM.

### Option B: Automated Execution Script
Run all 4 notebooks sequentially from the terminal:
```bash
python run_all_notebooks.py
```

---

## 🌐 How to Launch & Use the Front-End Web App

Start the interactive testing workbench:
```bash
python app.py
```
Open browser at: **[http://localhost:5000](http://localhost:5000)**

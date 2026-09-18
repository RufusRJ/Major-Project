import os
import shutil
import kagglehub

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

IMAGE_REAL_DIR = os.path.join(PROJECT_ROOT, "data", "image", "real")
IMAGE_FAKE_DIR = os.path.join(PROJECT_ROOT, "data", "image", "fake")
VIDEO_REAL_DIR = os.path.join(PROJECT_ROOT, "data", "video", "real")
VIDEO_FAKE_DIR = os.path.join(PROJECT_ROOT, "data", "video", "fake")
AUDIO_REAL_DIR = os.path.join(PROJECT_ROOT, "data", "audio", "real")
AUDIO_FAKE_DIR = os.path.join(PROJECT_ROOT, "data", "audio", "fake")

for d in [IMAGE_REAL_DIR, IMAGE_FAKE_DIR, VIDEO_REAL_DIR, VIDEO_FAKE_DIR, AUDIO_REAL_DIR, AUDIO_FAKE_DIR]:
    os.makedirs(d, exist_ok=True)

def setup_image_dataset():
    print("\n==================================================")
    print("Downloading Image Dataset (lukaslechovic/ffhq-facefusion-10k)...")
    print("==================================================")
    try:
        path = kagglehub.dataset_download("lukaslechovic/ffhq-facefusion-10k")
        print(f"Path to dataset files: {path}")

        real_copied, fake_copied = 0, 0
        for root, dirs, files in os.walk(path):
            parts = [p.lower() for p in root.replace("\\", "/").split("/")]
            
            is_fake = "fake" in parts or "deepfake" in parts or "fusion" in parts or "synthesized" in parts
            is_real = "real" in parts or "original" in parts or "ffhq" in parts and not is_fake

            for file in files:
                if file.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
                    src_file = os.path.join(root, file)
                    if is_fake:
                        dst_file = os.path.join(IMAGE_FAKE_DIR, file)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
                            fake_copied += 1
                    else:
                        dst_file = os.path.join(IMAGE_REAL_DIR, file)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
                            real_copied += 1
        
        print(f"-> Organized Image Dataset:")
        print(f"   Real Images: {real_copied} in data/image/real/")
        print(f"   Fake Images: {fake_copied} in data/image/fake/")
    except Exception as e:
        print(f"[Error processing image dataset]: {e}")

def setup_audio_dataset():
    print("\n==================================================")
    print("Downloading Audio Dataset (jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech)...")
    print("==================================================")
    try:
        path = kagglehub.dataset_download("jayjoshi37/deepfake-audio-dataset-fake-vs-real-speech")
        print(f"Path to dataset files: {path}")

        real_copied, fake_copied = 0, 0
        for root, dirs, files in os.walk(path):
            parts = [p.lower() for p in root.replace("\\", "/").split("/")]
            
            is_fake = "fake" in parts or "spoof" in parts or "synthetic" in parts
            is_real = "real" in parts or "bonafide" in parts or "human" in parts

            for file in files:
                if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')):
                    src_file = os.path.join(root, file)
                    if is_fake:
                        dst_file = os.path.join(AUDIO_FAKE_DIR, file)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
                            fake_copied += 1
                    elif is_real:
                        dst_file = os.path.join(AUDIO_REAL_DIR, file)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
                            real_copied += 1
        
        print(f"-> Organized Audio Dataset:")
        print(f"   Real Audio: {real_copied} in data/audio/real/")
        print(f"   Fake Audio: {fake_copied} in data/audio/fake/")
    except Exception as e:
        print(f"[Error processing audio dataset]: {e}")

def setup_video_dataset():
    print("\n==================================================")
    print("Downloading Video Dataset (xdxd003/ff-c23)...")
    print("==================================================")
    try:
        path = kagglehub.dataset_download("xdxd003/ff-c23")
        print(f"Path to dataset files: {path}")

        real_copied, fake_copied = 0, 0
        for root, dirs, files in os.walk(path):
            parts = [p.lower() for p in root.replace("\\", "/").split("/")]
            
            is_real_vid = any(p in parts for p in ["original_sequences", "youtube", "original", "real", "actors"])
            is_fake_vid = not is_real_vid and any(p in parts for p in [
                "manipulated_sequences", "deepfakes", "deepfakedetection", 
                "faceswap", "face2face", "neuraltextures", "faceshifter", "fake"
            ])

            for file in files:
                if file.lower().endswith(('.mp4', '.avi', '.mov')):
                    src_file = os.path.join(root, file)
                    if is_real_vid:
                        dst_file = os.path.join(VIDEO_REAL_DIR, file)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
                            real_copied += 1
                    elif is_fake_vid:
                        dst_file = os.path.join(VIDEO_FAKE_DIR, file)
                        if not os.path.exists(dst_file):
                            shutil.copy2(src_file, dst_file)
                            fake_copied += 1
        
        print(f"-> Organized Video Dataset (FaceForensics++ c23):")
        print(f"   Real Videos: {real_copied} in data/video/real/")
        print(f"   Fake Videos: {fake_copied} in data/video/fake/")
    except Exception as e:
        print(f"[Error processing video dataset]: {e}")

if __name__ == "__main__":
    setup_image_dataset()
    setup_audio_dataset()
    setup_video_dataset()
    print("\nDataset setup script complete!")

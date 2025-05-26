# models/clip_video_encoder.py
import os
import torch
import clip
from PIL import Image


device = "cuda" if torch.cuda.is_available() else "cpu"
clip_model, preprocess = clip.load("ViT-B/32", device=device)


def extract_clip_video_embedding(frames_folder, max_frames=32):
    """
    Compute a global video embedding using CLIP by averaging over frame-level features.

    Parameters:
        frames_folder (str): Path to folder containing frames
        max_frames (int): Max number of frames to use for embedding

    Returns:
        torch.Tensor: averaged video-level embedding (shape: [512])
    """
    frame_files = sorted(os.listdir(frames_folder))[:max_frames]
    embeddings = []

    for filename in frame_files:
        frame_path = os.path.join(frames_folder, filename)
        try:
            image = Image.open(frame_path).convert("RGB")
            image_input = preprocess(image).unsqueeze(0).to(device)
            with torch.no_grad():
                image_features = clip_model.encode_image(image_input)
                embeddings.append(image_features.cpu())
        except Exception as e:
            print(f"⚠️ Failed to process frame {filename}: {e}")
            continue

    if not embeddings:
        return None

    video_embedding = torch.stack(embeddings).mean(dim=0).squeeze()
    return video_embedding

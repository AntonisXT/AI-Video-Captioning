# models/frame_selector.py
import os
import re
import torch
import clip
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from sentence_transformers import SentenceTransformer, util
from utils.text_utils import clean_caption
from utils.embedding_cache import EmbeddingCache


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class FrameSelector:
    """
    Selects the most representative frame caption for a video scene using:
    - Motion keyword detection
    - Semantic fusion of multiple captions
    - CLIP similarity with global video embedding (optional)
    """
    def __init__(self):
        self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.embedding_cache = EmbeddingCache(device=device)

        self.motion_keywords = {
            "driving", "moving", "running", "walking", "flying", "jumping", "dancing", "swimming",
            "riding", "rolling", "sliding", "traveling", "cruising", "accelerating", "walking down", "driving down"
        }

    def _generate_caption(self, image_path):
        """Generate a BLIP caption for the given image frame."""
        try:
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(image, return_tensors="pt").to(device)
            output = self.blip_model.generate(**inputs)
            caption = self.processor.decode(output[0], skip_special_tokens=True)
            return clean_caption(caption)
        except Exception as e:
            print(f"⚠️ BLIP captioning failed on {image_path}: {e}")
            return ""

    def _contains_motion(self, caption):
        """Check if the caption includes motion-related keywords."""
        return any(word in caption.lower() for word in self.motion_keywords)

    def _semantic_fusion(self, captions):
        """
        Select the most representative caption from a list using semantic similarity
        to the mean embedding and keyword richness as tie-breaker.
        """
        if len(set(captions)) == 1:
            return captions[0]

        embeddings = self.semantic_model.encode(captions, convert_to_tensor=True)
        mean_embedding = embeddings.mean(dim=0)
        similarities = util.cos_sim(embeddings, mean_embedding).squeeze()

        def score(caption, idx):
            unique_words = len(set(re.findall(r'\b\w+\b', caption.lower())))
            return similarities[idx].item() + 0.01 * unique_words

        scores = [score(c, i) for i, c in enumerate(captions)]
        return captions[int(torch.tensor(scores).argmax())]

    def select_best_caption(self, scene, frame_files, frames_folder, clip_video_embedding=None):
        """
        Select the best caption for a scene using motion cues, semantic fusion,
        and optional CLIP similarity guidance.
        """
        start, end = scene["start_frame"], scene["end_frame"]
        mid = (start + end) // 2
        frame_indices = list(set([start, mid, end]))

        candidates = []
        for idx in frame_indices:
            if 0 <= idx < len(frame_files):
                path = os.path.join(frames_folder, frame_files[idx])
                caption = self._generate_caption(path)
                candidates.append((path, caption))

        for _, caption in candidates:
            if self._contains_motion(caption):
                return caption

        fused_caption = self._semantic_fusion([c[1] for c in candidates])

        if clip_video_embedding is not None:
            try:
                tokens = clip.tokenize([fused_caption]).to(device)
                clip_text_emb = self.clip_model.encode_text(tokens)
                clip_score = self.embedding_cache.clip_similarity(clip_text_emb, clip_video_embedding)
                return fused_caption
            except Exception as e:
                print(f"⚠️ CLIP similarity failed: {e}")

        return max(candidates, key=lambda x: len(set(x[1].split())))[1]

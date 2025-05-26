# utils/embedding_cache.py
import torch
from sentence_transformers import SentenceTransformer
import clip


class EmbeddingCache:
    """
    Caches and batches sentence and CLIP text embeddings to avoid redundant computation.
    Improves runtime efficiency for downstream caption filtering, merging, and scoring.
    """
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.semantic_model = SentenceTransformer("all-MiniLM-L6-v2", device=self.device)
        self.clip_model, _ = clip.load("ViT-B/32", device=self.device)

        # Internal embedding caches
        self._semantic_cache = {}
        self._clip_text_cache = {}

    def encode_sentences(self, texts):
        """
        Encode a batch of sentences with SentenceTransformer.
        Avoids recomputing previously cached embeddings.

        Parameters:
            texts (List[str]): input sentences

        Returns:
            torch.Tensor: semantic embeddings tensor [N, D]
        """
        uncached = [t for t in texts if t not in self._semantic_cache]
        if uncached:
            new_embeddings = self.semantic_model.encode(uncached, convert_to_tensor=True)
            for t, emb in zip(uncached, new_embeddings):
                self._semantic_cache[t] = emb.detach().cpu()

        return torch.stack([self._semantic_cache[t] for t in texts]).to(self.device)

    def encode_clip_texts(self, texts):
        """
        Encode a batch of texts using CLIP's text encoder.
        Avoids recomputation using internal cache.

        Parameters:
            texts (List[str]): input captions or summaries

        Returns:
            torch.Tensor: CLIP embeddings tensor [N, 512]
        """
        uncached = [t for t in texts if t not in self._clip_text_cache]
        if uncached:
            tokenized = clip.tokenize(uncached).to(self.device)
            with torch.no_grad():
                new_embeddings = self.clip_model.encode_text(tokenized)
            for t, emb in zip(uncached, new_embeddings):
                self._clip_text_cache[t] = emb.detach().cpu()

        return torch.stack([self._clip_text_cache[t] for t in texts]).to(self.device)

    def clip_similarity(self, clip_embeddings, video_embedding):
        """
        Compute cosine similarity between a batch of CLIP embeddings and a video-level CLIP embedding.
        Automatically aligns devices.
        """
        video_embedding = video_embedding.to(clip_embeddings.device)
        return torch.nn.functional.cosine_similarity(clip_embeddings, video_embedding.unsqueeze(0)).squeeze()

    def clear(self):
        """
        Clear both semantic and CLIP text embedding caches.
        Useful between batch jobs or evaluations.
        """
        self._semantic_cache.clear()
        self._clip_text_cache.clear()

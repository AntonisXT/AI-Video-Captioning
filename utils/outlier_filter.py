# utils/outlier_filter.py
from sklearn.neighbors import LocalOutlierFactor
from sklearn.cluster import DBSCAN
from utils.embedding_cache import EmbeddingCache
import numpy as np
import torch


class OutlierFilter:
    """
    Detects and removes semantic outlier captions based on duration,
    Local Outlier Factor, and cosine similarity clustering.
    """
    def __init__(self):
        self.embedding_cache = EmbeddingCache()

    def filter_by_lof(self, captions, duration_threshold=1, lof_threshold=-1.4, sim_threshold=0.3, sim_protect_threshold=0.45):
        """
        Removes captions that are isolated in semantic space using LOF and cosine similarity.

        Parameters:
            captions (list of dict): Scene-level caption objects with timestamp info.

        Returns:
            list of dict: Filtered scene captions.
        """
        if len(captions) < 4:
            return captions

        texts = [c["caption"].strip().lower() for c in captions]
        embeddings = self.embedding_cache.encode_sentences(texts)

        num_points = len(embeddings)
        if num_points <= 6:
            n_neighbors = 2
        elif num_points <= 10:
            n_neighbors = 3
        else:
            n_neighbors = 5

        # Local Outlier Factor based on cosine distance
        lof = LocalOutlierFactor(n_neighbors=min(n_neighbors, num_points - 1), metric="cosine")
        lof.fit(embeddings.cpu().numpy())
        lof_scores = lof.negative_outlier_factor_

        mean_embedding = embeddings.mean(dim=0)

        filtered = []
        for i, cap in enumerate(captions):
            duration = cap["end_time"] - cap["start_time"]
            emb = embeddings[i]
            sim = torch.nn.functional.cosine_similarity(emb, mean_embedding, dim=0).item()
            score = lof_scores[i]

            is_outlier = (
                duration <= duration_threshold and
                sim < sim_protect_threshold and
                (score < lof_threshold or sim < sim_threshold)
            )

            if is_outlier:
                print(f"⚠️ Removed outlier: [{cap['start_time']}s - {cap['end_time']}s] score={score:.2f} sim={sim:.2f}: {cap['caption']}")
            else:
                filtered.append(cap)

        return filtered

    def filter_by_clustering(self, captions, eps=0.45, min_samples=2):
        """
        Uses DBSCAN to cluster captions and retain only the main thematic group.
        Falls back to duration-weighted semantic similarity for very short videos.

        Parameters:
            captions (list of dict): List of scene captions with timestamp info

        Returns:
            list of dict: Filtered captions (dominant theme only)
        """
        if not captions:
            return captions

        texts = [c["caption"].strip().lower() for c in captions]
        embeddings = self.embedding_cache.encode_sentences(texts)

        if len(captions) <= 3:
            # Fallback: semantic + duration weighted relevance
            mean_embedding = embeddings.mean(dim=0)
            sims = torch.nn.functional.cosine_similarity(embeddings, mean_embedding.unsqueeze(0)).squeeze()
            durations = torch.tensor([c["end_time"] - c["start_time"] for c in captions], dtype=torch.float32, device=sims.device)

            scores = sims + 0.05 * torch.log1p(durations)
            best_idx = int(torch.argmax(scores).item())
            return [captions[best_idx]]

        # DBSCAN clustering in cosine space
        embeddings_np = embeddings.cpu().numpy()
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric="cosine").fit(embeddings_np)
        labels = clustering.labels_

        valid_labels = labels[labels >= 0]
        if len(valid_labels) == 0:
            return captions

        main_cluster = np.argmax(np.bincount(valid_labels))
        filtered = [c for c, label in zip(captions, labels) if label == main_cluster]
        return filtered
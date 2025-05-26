# utils/scene_merger.py
from sentence_transformers import util
import torch
import re
from collections import Counter
from utils.embedding_cache import EmbeddingCache


class SceneMerger:
    """
    Merges temporally adjacent video scenes with semantically similar captions.
    Uses sentence similarity, motion keywords, and caption frequency to merge scenes meaningfully.
    """
    def __init__(self, similarity_threshold=0.7, short_scene_threshold=1):
        self.similarity_threshold = similarity_threshold
        self.short_scene_threshold = short_scene_threshold
        self.embedding_cache = EmbeddingCache()

        self.motion_keywords = {
            "driving", "moving", "running", "walking", "flying", "jumping", "dancing", "swimming",
            "riding", "rolling", "sliding", "traveling", "cruising", "accelerating"
        }

    def _contains_motion(self, caption):
        """Check if the caption contains any predefined motion keywords."""
        caption = caption.lower()
        return any(word in caption for word in self.motion_keywords)

    def _count_unique_keywords(self, caption):
        """Return the number of unique words in a caption (as a proxy for richness)."""
        words = re.findall(r'\b\w+\b', caption.lower())
        return len(set(words))

    def _select_best_caption(self, c1, c2, caption_freq):
        """
        Select which caption to keep when merging two scenes.
        Preference given to higher frequency, more keywords, and motion references.
        """
        freq1 = caption_freq.get(c1.lower(), 0)
        freq2 = caption_freq.get(c2.lower(), 0)

        if freq1 > freq2:
            return c1
        elif freq2 > freq1:
            return c2
        else:
            score1 = self._count_unique_keywords(c1) + (1 if self._contains_motion(c1) else 0)
            score2 = self._count_unique_keywords(c2) + (1 if self._contains_motion(c2) else 0)
            return c1 if score1 >= score2 else c2

    def merge(self, captions):
        """
        Merge adjacent scenes if they are semantically similar or very short and close in time.

        Parameters:
            captions (list of dict): each dict must have 'start_time', 'end_time', 'caption'

        Returns:
            list of dict: merged scene list with updated time spans and captions
        """
        if not captions:
            return []

        # Frequency count for each caption (lowercase)
        caption_freq = Counter(c["caption"].strip().lower() for c in captions)

        # Precompute all embeddings
        all_texts = [c["caption"] for c in captions]
        all_embeddings = self.embedding_cache.encode_sentences(all_texts)

        merged = [captions[0]]
        last_emb = all_embeddings[0]

        for i in range(1, len(captions)):
            current = captions[i]
            last = merged[-1]
            current_emb = all_embeddings[i]

            sim = util.cos_sim(last_emb, current_emb).item()
            duration = current["end_time"] - current["start_time"]
            time_gap = current["start_time"] - last["end_time"]

            should_merge = (
                sim >= self.similarity_threshold or
                (duration <= self.short_scene_threshold and sim >= 0.55 and time_gap <= 1)
            )

            if should_merge:
                last["end_time"] = current["end_time"]
                last["caption"] = self._select_best_caption(last["caption"], current["caption"], caption_freq)
                last_emb = self.embedding_cache.encode_sentences([last["caption"]])[0]
            else:
                merged.append(current)
                last_emb = current_emb

        for i, scene in enumerate(merged):
            scene["scene_id"] = i + 1

        return merged

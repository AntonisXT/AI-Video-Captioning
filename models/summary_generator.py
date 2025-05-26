# models/summary_generator.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils.text_utils import clean_caption
from utils.outlier_filter import OutlierFilter
from utils.embedding_cache import EmbeddingCache
import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SummaryGenerator:
    """
    Generates a video-level summary sentence using scene captions.
    Applies semantic filtering, CLIP-based scoring, and language modeling via FLAN-T5.
    """
    def __init__(self, model_name="google/flan-t5-base"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()

        self.outlier_filter = OutlierFilter()
        self.embedding_cache = EmbeddingCache(device=device)

    def _score_captions(self, captions, clip_video_embedding):
        """
        Compute a composite relevance score for each caption using:
        - Semantic centrality
        - CLIP alignment
        - Duration weighting
        """
        texts = [c["caption"].strip().lower() for c in captions]
        durations = [c["end_time"] - c["start_time"] for c in captions]

        embeddings = self.embedding_cache.encode_sentences(texts)
        semantic_scores = torch.nn.functional.cosine_similarity(embeddings, embeddings.mean(dim=0).unsqueeze(0))

        if clip_video_embedding is not None:
            clip_embeddings = self.embedding_cache.encode_clip_texts(texts)
            clip_scores = self.embedding_cache.clip_similarity(clip_embeddings, clip_video_embedding)
        else:
            clip_scores = torch.zeros(len(texts), dtype=torch.float32, device=device)

        durations_tensor = torch.tensor(durations, dtype=torch.float32, device=device)
        total_scores = 0.6 * semantic_scores + 0.3 * clip_scores + 0.1 * torch.log1p(durations_tensor)
        return texts, total_scores

    def _fallback_summary(self, captions, max_length):
        sorted_caps = sorted(captions, key=lambda x: x["start_time"])
        labels = ["First", "Next", "Finally"]

        if len(sorted_caps) == 2:
            prompts = [
                f"Summarize the video in one short and natural sentence:\n{cap['caption'].rstrip('.')}"
                for cap in sorted_caps
            ]
            outputs = []
            for prompt in prompts:
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
                out = self.model.generate(input_ids, max_length=max_length, min_length=5, num_beams=4)
                outputs.append(self.tokenizer.decode(out[0], skip_special_tokens=True).strip())
            return f"{outputs[0]} and then {outputs[1].lower()}"

        phrases = []
        for label, cap in zip(labels, sorted_caps):
            prompt = f"Summarize the scene in one natural sentence:\n{cap['caption'].rstrip('.')}"
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
            out = self.model.generate(input_ids, max_length=max_length, min_length=5, num_beams=4)
            phrases.append(f"{label}, {self.tokenizer.decode(out[0], skip_special_tokens=True).strip()}")
        return " ".join(phrases)

    @torch.no_grad()
    def generate(self, captions, clip_video_embedding=None, max_length=35):
        if not captions:
            return "a video"

        valid = [c for c in captions if isinstance(c, dict) and "caption" in c and c["caption"].strip()]
        if not valid:
            return "a video"

        texts = [c["caption"].strip().lower() for c in valid]
        embeddings = self.embedding_cache.encode_sentences(texts)
        sim_matrix = torch.nn.functional.cosine_similarity(embeddings.unsqueeze(1), embeddings.unsqueeze(0), dim=-1)
        mean_sim = sim_matrix.mean().item()

        if len(valid) <= 3 and mean_sim < 0.4:
            return self._fallback_summary(valid, max_length)

        filtered = self.outlier_filter.filter_by_clustering(valid)
        if not filtered:
            return "a video"
        if len(filtered) == 1 or len(set(c["caption"].lower() for c in filtered)) == 1:
            return clean_caption(filtered[0]["caption"])

        texts, scores = self._score_captions(filtered, clip_video_embedding)
        top_k = min(5, len(texts))
        top_indices = torch.topk(scores, k=top_k).indices.tolist()
        selected = list(dict.fromkeys([texts[i] for i in top_indices]))

        prompt_input = ". ".join([s.rstrip(".") for s in selected]) + "."
        prompt = (
            "Summarize the video in one short and natural sentence based on these scene descriptions:\n"
            f"{prompt_input}"
        )

        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        output = self.model.generate(input_ids, max_length=max_length, min_length=5, num_beams=4)
        return clean_caption(self.tokenizer.decode(output[0], skip_special_tokens=True).strip().lower())

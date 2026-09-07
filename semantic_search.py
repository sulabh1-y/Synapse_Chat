import os
import re
import json
import numpy as np
from typing import List, Dict, Any, Union, Optional
from sentence_transformers import SentenceTransformer

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "or", "is",
    "are", "was", "were", "it", "with", "from", "by", "this", "that", "i", "you",
    "he", "she", "we", "they", "my", "your", "his", "her", "our", "their", "me",
    "us", "him", "them", "what", "which", "who", "whom", "am", "be", "been", "being"
}

class SemanticSearchEngine:
    """
    Search Engine supporting Semantic Vector Search, Keyword Frequency Search, Hybrid Search,
    and metadata filtering (sender, date range, mandatory keyword).
    """
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        corpus_path: Optional[str] = "chat_corpus.json",
        messages: Optional[List[Dict[str, Any]]] = None
    ):
        self.model_name = model_name
        self.corpus_path = corpus_path
        self.corpus: List[Dict[str, Any]] = []
        self.embeddings: Optional[np.ndarray] = None
        
        # Load embedding model in offline mode if cached
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(self.model_name)

        if messages is not None:
            self.corpus = messages
        elif corpus_path and os.path.exists(corpus_path):
            with open(corpus_path, "r", encoding="utf-8") as f:
                self.corpus = json.load(f)
            
        if self.corpus:
            self.build_index()

    def build_index(self):
        """Encodes corpus into dense vectors and normalizes them for cosine similarity."""
        if not self.corpus:
            return

        texts = [msg.get("text", "") for msg in self.corpus]
        raw_embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = raw_embeddings / norms
        print(f"✅ Vector index built for {len(self.corpus)} messages! Shape: {self.embeddings.shape}")

    def get_unique_senders(self) -> List[str]:
        """Returns sorted list of unique senders in the corpus."""
        senders = set(msg.get("sender") for msg in self.corpus if msg.get("sender"))
        return sorted(list(senders))

    def _extract_query_terms(self, query: str) -> List[str]:
        """Tokenizes query and filters out stop words."""
        tokens = re.findall(r'\b[a-zA-Z0-9_-]+\b', query.lower())
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    def _find_matched_keywords(self, query_terms: List[str], text: str) -> List[str]:
        """Finds which query terms are present in text."""
        text_lower = text.lower()
        return [term for term in query_terms if re.search(r'\b' + re.escape(term) + r'\b', text_lower) or term in text_lower]

    def _compute_keyword_frequency(self, query_terms: List[str], text: str) -> int:
        """Computes total frequency of query words in message text."""
        text_lower = text.lower()
        return sum(len(re.findall(r'\b' + re.escape(term) + r'\b', text_lower)) for term in query_terms)

    def search(
        self,
        query: str,
        top_k: int = 5,
        mode: str = "semantic",
        sender: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        keyword_filter: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes search with filtering by sender, date range, and mandatory keyword.
        """
        if not self.corpus:
            return []

        clean_query = (query or "").strip()
        query_terms = self._extract_query_terms(clean_query) if clean_query else []
        if not query_terms and clean_query:
            query_terms = re.findall(r'\b[a-zA-Z0-9_-]+\b', clean_query.lower())

        # Step 1: Pre-filter candidate message indices based on metadata
        eligible_indices = []
        for idx, msg in enumerate(self.corpus):
            # Username Filter
            if sender and sender.strip() and sender.strip().lower() != "all":
                msg_sender = (msg.get("sender") or "").strip().lower()
                if msg_sender != sender.strip().lower():
                    continue

            # Date Range Filter
            msg_ts = msg.get("timestamp") or ""
            msg_date = msg_ts[:10] if len(msg_ts) >= 10 else ""
            
            if start_date and start_date.strip():
                if msg_date < start_date.strip():
                    continue
            if end_date and end_date.strip():
                if msg_date > end_date.strip():
                    continue

            # Mandatory Keyword Filter
            if keyword_filter and keyword_filter.strip():
                kw = keyword_filter.strip().lower()
                if kw not in (msg.get("text") or "").lower():
                    continue

            eligible_indices.append(idx)

        if not eligible_indices:
            return []

        candidates = []

        # If query is empty, simply return top filtered messages ordered by timestamp
        if not clean_query:
            for idx in eligible_indices[:top_k]:
                msg = self.corpus[idx]
                candidates.append({
                    "message": dict(msg),
                    "score": 1.0,
                    "search_type": mode,
                    "matched_keywords": []
                })
            return candidates

        # Step 2: Rank eligible messages using requested search mode
        if mode == "keyword":
            for idx in eligible_indices:
                msg = self.corpus[idx]
                text = msg.get("text", "")
                freq_score = self._compute_keyword_frequency(query_terms, text)
                matched_kw = self._find_matched_keywords(query_terms, text)
                
                if freq_score > 0 or (keyword_filter and keyword_filter.strip()):
                    candidates.append({
                        "message": dict(msg),
                        "score": float(max(freq_score, 1.0)),
                        "search_type": "keyword",
                        "matched_keywords": matched_kw
                    })

            candidates.sort(key=lambda x: x["score"], reverse=True)

        elif mode == "hybrid":
            query_vec = self.model.encode([clean_query], convert_to_numpy=True)[0]
            query_norm = np.linalg.norm(query_vec)
            if query_norm > 0:
                query_vec = query_vec / query_norm

            # Dot product with pre-normalized embeddings
            sim_scores = np.dot(self.embeddings[eligible_indices], query_vec)

            for i, idx in enumerate(eligible_indices):
                msg = self.corpus[idx]
                text = msg.get("text", "")
                cos_sim = float(sim_scores[i])
                freq_score = self._compute_keyword_frequency(query_terms, text)
                matched_kw = self._find_matched_keywords(query_terms, text)

                norm_freq = min(1.0, freq_score / 3.0)
                final_score = round(0.8 * cos_sim + 0.2 * norm_freq, 4)

                candidates.append({
                    "message": dict(msg),
                    "score": final_score,
                    "similarity_percentage": f"{round(cos_sim * 100, 2)}%",
                    "search_type": "hybrid",
                    "matched_keywords": matched_kw
                })

            candidates.sort(key=lambda x: x["score"], reverse=True)

        else: # mode == "semantic"
            query_vec = self.model.encode([clean_query], convert_to_numpy=True)[0]
            query_norm = np.linalg.norm(query_vec)
            if query_norm > 0:
                query_vec = query_vec / query_norm

            sim_scores = np.dot(self.embeddings[eligible_indices], query_vec)

            for i, idx in enumerate(eligible_indices):
                msg = self.corpus[idx]
                text = msg.get("text", "")
                cos_sim = float(sim_scores[i])
                matched_kw = self._find_matched_keywords(query_terms, text)

                candidates.append({
                    "message": dict(msg),
                    "score": round(cos_sim, 4),
                    "similarity_percentage": f"{round(cos_sim * 100, 2)}%",
                    "search_type": "semantic",
                    "matched_keywords": matched_kw
                })

            candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:min(top_k, len(candidates))]

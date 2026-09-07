import os
import re
import numpy as np
from typing import List, Dict, Any, Optional
from sentence_transformers import SentenceTransformer
from models.chat_corpus import ChatCorpusLoader

STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "for", "to", "of", "and", "or", "is",
    "are", "was", "were", "it", "with", "from", "by", "this", "that", "i", "you",
    "he", "she", "we", "they", "my", "your", "his", "her", "our", "their", "me",
    "us", "him", "them", "what", "which", "who", "whom", "am", "be", "been", "being"
}

class SearchService:
    """
    Core Search Service containing all Vector Embedding & Keyword Search logic.
    Decoupled from Flask web framework to ensure testability and scalability.
    """
    def __init__(
        self,
        corpus_loader: ChatCorpusLoader,
        model_name: str = "all-MiniLM-L6-v2"
    ):
        self.corpus_loader = corpus_loader
        self.model_name = model_name
        self.embeddings: Optional[np.ndarray] = None

        # Initialize Sentence Transformer Model
        print(f"🧠 [SearchService] Loading ML Model: '{self.model_name}'...")
        os.environ["HF_HUB_OFFLINE"] = "1"
        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            self.model = SentenceTransformer(self.model_name)

        if len(self.corpus_loader) > 0:
            self.build_embeddings_index()

    def build_embeddings_index(self):
        """Encodes all message texts into L2-normalized 384-d dense vectors."""
        messages = self.corpus_loader.get_messages()
        if not messages:
            return

        texts = [msg.get("text", "") for msg in messages]
        print(f"⚡ [SearchService] Building vector embeddings index for {len(texts)} messages...")
        
        raw_embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        norms = np.linalg.norm(raw_embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.embeddings = raw_embeddings / norms
        print(f"✅ [SearchService] In-memory index built! Vector shape: {self.embeddings.shape}")

    def extract_query_terms(self, query: str) -> List[str]:
        """Tokenizes search query into clean terms (excluding stop words)."""
        tokens = re.findall(r'\b[a-zA-Z0-9_-]+\b', query.lower())
        return [t for t in tokens if t not in STOP_WORDS and len(t) > 1]

    def find_matched_keywords(self, query_terms: List[str], text: str) -> List[str]:
        """Identifies which query terms occur inside text."""
        text_lower = text.lower()
        return [term for term in query_terms if re.search(r'\b' + re.escape(term) + r'\b', text_lower) or term in text_lower]

    def compute_keyword_frequency(self, query_terms: List[str], text: str) -> int:
        """Calculates total occurrences of query terms in message text."""
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
        Executes filtered search across indexed chat messages.
        
        Modes supported:
          - 'semantic': Dense vector cosine similarity using all-MiniLM-L6-v2.
          - 'keyword': Exact keyword frequency ranking.
          - 'hybrid': Weighted combination of vector distance + keyword frequency.
        """
        messages = self.corpus_loader.get_messages()
        if not messages:
            return []

        clean_query = (query or "").strip()
        query_terms = self.extract_query_terms(clean_query) if clean_query else []

        # Step 1: Pre-filter candidate indices by metadata (Sender, Date, Mandatory Keyword)
        eligible_indices = []
        for idx, msg in enumerate(messages):
            if sender and sender.strip() and sender.strip().lower() != "all":
                if (msg.get("sender") or "").strip().lower() != sender.strip().lower():
                    continue

            msg_ts = msg.get("timestamp") or ""
            msg_date = msg_ts[:10] if len(msg_ts) >= 10 else ""
            
            if start_date and start_date.strip() and msg_date < start_date.strip():
                continue
            if end_date and end_date.strip() and msg_date > end_date.strip():
                continue

            if keyword_filter and keyword_filter.strip():
                if keyword_filter.strip().lower() not in (msg.get("text") or "").lower():
                    continue

            eligible_indices.append(idx)

        if not eligible_indices:
            return []

        candidates = []

        # Step 2: Handle empty search query (return top metadata-filtered items)
        if not clean_query:
            for idx in eligible_indices[:top_k]:
                candidates.append({
                    "message": dict(messages[idx]),
                    "score": 1.0,
                    "search_type": mode,
                    "matched_keywords": []
                })
            return candidates

        # Step 3: Compute Relevance Scores based on requested Search Mode
        if mode == "keyword":
            for idx in eligible_indices:
                msg = messages[idx]
                text = msg.get("text", "")
                freq_score = self.compute_keyword_frequency(query_terms, text)
                matched_kw = self.find_matched_keywords(query_terms, text)
                
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

            sim_scores = np.dot(self.embeddings[eligible_indices], query_vec)

            for i, idx in enumerate(eligible_indices):
                msg = messages[idx]
                cos_sim = float(sim_scores[i])
                freq_score = self.compute_keyword_frequency(query_terms, msg.get("text", ""))
                matched_kw = self.find_matched_keywords(query_terms, msg.get("text", ""))

                final_score = round(0.8 * cos_sim + 0.2 * min(1.0, freq_score / 3.0), 4)
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
                msg = messages[idx]
                cos_sim = float(sim_scores[i])
                matched_kw = self.find_matched_keywords(query_terms, msg.get("text", ""))
                
                candidates.append({
                    "message": dict(msg),
                    "score": round(cos_sim, 4),
                    "similarity_percentage": f"{round(cos_sim * 100, 2)}%",
                    "search_type": "semantic",
                    "matched_keywords": matched_kw
                })

            candidates.sort(key=lambda x: x["score"], reverse=True)

        return candidates[:min(top_k, len(candidates))]

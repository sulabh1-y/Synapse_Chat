import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder

class SynapseChatEngine:
    """
    Synapse_Chat Semantic Search & Retrieval Engine.
    Combines Multilingual Dense Vector Embeddings, Metadata Intent Routing,
    Context-Windowing, and Cross-Encoder Hybrid Reranking.
    """
    def __init__(
        self,
        corpus_path: str = "chat_corpus.json",
        db_dir: str = "./chroma_db",
        model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        reranker_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        collection_name: str = "synapse_chat_collection"
    ):
        self.corpus_path = corpus_path
        self.db_dir = db_dir
        self.model_name = model_name
        self.reranker_name = reranker_name
        self.collection_name = collection_name
        
        self.corpus: List[Dict[str, Any]] = []
        self.corpus_by_id: Dict[int, Dict[str, Any]] = {}
        self.load_corpus()

        # Initialize Primary Multilingual Embedding Model
        print(f"🧠 [Synapse_Chat] Loading Embedding Model: {self.model_name}...")
        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            os.environ["HF_HUB_OFFLINE"] = "1"
            self.model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", local_files_only=True)

        # Initialize Cross-Encoder Reranker
        print(f"⚡ [Synapse_Chat] Loading Reranker Engine...")
        try:
            self.reranker = CrossEncoder(self.reranker_name, local_files_only=True, max_length=512)
        except Exception:
            self.reranker = None
            
        self.chroma_client = chromadb.PersistentClient(path=self.db_dir)
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def load_corpus(self):
        """Loads corpus JSON into memory."""
        if os.path.exists(self.corpus_path):
            with open(self.corpus_path, "r", encoding="utf-8") as f:
                self.corpus = json.load(f)
            self.corpus_by_id = {msg["message_id"]: msg for msg in self.corpus}
            print(f"💬 [Synapse_Chat] Loaded {len(self.corpus)} messages from {self.corpus_path}")
        else:
            print(f"⚠️ Corpus file {self.corpus_path} not found.")

    def index_corpus(self, force_reindex: bool = False, batch_size: int = 256):
        """
        Indexes corpus messages into ChromaDB vector database using Context-Aware Embeddings.
        """
        existing_count = self.collection.count()
        current_dim = getattr(self.model, "get_sentence_embedding_dimension", lambda: None)()
        
        # Verify vector dimension compatibility
        if existing_count > 0 and current_dim and not force_reindex:
            try:
                sample = self.collection.get(limit=1, include=["embeddings"])
                if sample and sample["embeddings"] and len(sample["embeddings"]) > 0:
                    existing_dim = len(sample["embeddings"][0])
                    if existing_dim != current_dim:
                        print(f"⚠️ Vector DB dimension mismatch ({existing_dim} vs {current_dim}). Auto re-indexing collection...")
                        force_reindex = True
            except Exception:
                pass

        if existing_count >= len(self.corpus) and not force_reindex:
            print(f"✅ [Synapse_Chat] Vector DB already indexed with {existing_count} items. Ready for search!")
            return

        print("🚀 [Synapse_Chat] Building High-Precision Context-Aware Vector Index...")
        try:
            self.chroma_client.delete_collection(self.collection_name)
        except Exception:
            pass
            
        self.collection = self.chroma_client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"}
        )

        total_msgs = len(self.corpus)
        for i in range(0, total_msgs, batch_size):
            batch = self.corpus[i : i + batch_size]
            
            documents = []
            metadatas = []
            ids = []
            
            for idx_in_batch, msg in enumerate(batch):
                global_idx = i + idx_in_batch
                
                prev_text = f" Prev: {self.corpus[global_idx-1]['sender']}: {self.corpus[global_idx-1]['text']}" if global_idx > 0 else ""
                next_text = f" Next: {self.corpus[global_idx+1]['sender']}: {self.corpus[global_idx+1]['text']}" if global_idx < total_msgs - 1 else ""
                
                doc_text = f"{msg['sender']}: {msg['text']}{prev_text}{next_text}"
                documents.append(doc_text)
                
                dt = datetime.fromisoformat(msg["timestamp"])
                metadatas.append({
                    "message_id": msg["message_id"],
                    "sender": msg["sender"],
                    "timestamp": msg["timestamp"],
                    "year": dt.year,
                    "month": dt.month,
                    "month_name": dt.strftime("%B").lower(),
                    "iso_date": dt.strftime("%Y-%m-%d")
                })
                ids.append(str(msg["message_id"]))

            embeddings = self.model.encode(documents, show_progress_bar=False).tolist()
            
            self.collection.add(
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
                ids=ids
            )
            print(f"📦 Indexed batch {i} to {i + len(batch)} / {total_msgs}")
            
        print("🎉 [Synapse_Chat] Context-Aware Vector Indexing complete!")

    def parse_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Parses query into metadata filters (Sender, Month, Year) and clean semantic query text.
        """
        query_lower = query.lower().strip()
        sender_filter = None
        month_filter = None
        year_filter = None
        
        known_senders = ["aarav", "priya", "rohan", "sneha", "vikram", "ananya", "kabir", "neha"]
        
        for s in known_senders:
            patterns = [
                rf"\bfrom {s}\b", rf"\bby {s}\b", rf"\bsent by {s}\b",
                rf"\b{s} said\b", rf"\b{s} wrote\b", rf"\b{s}:"
            ]
            if any(re.search(p, query_lower) for p in patterns):
                sender_filter = s.capitalize()
                break
        

        months = ["january", "february", "march", "april", "may", "june", "july", "august", "september", "october", "november", "december"]
        for m in months:
            if m in query_lower:
                month_filter = m
                break
                
        year_match = re.search(r'\b(202[5-9])\b', query_lower)
        if year_match:
            year_filter = int(year_match.group(1))

        clean_query = query
        if sender_filter:
            clean_query = re.sub(rf"\b(from|by|sent by|said by)?\s*{sender_filter}\b", "", clean_query, flags=re.IGNORECASE)
        if month_filter:
            clean_query = re.sub(rf"\b(in|during|for)?\s*{month_filter}\b", "", clean_query, flags=re.IGNORECASE)
        if year_filter:
            clean_query = re.sub(rf"\b{year_filter}\b", "", clean_query)
            
        clean_query = re.sub(r'\s+', ' ', clean_query).strip()
        if not clean_query:
            clean_query = query

        return {
            "raw_query": query,
            "clean_query": clean_query,
            "sender_filter": sender_filter,
            "month_filter": month_filter,
            "year_filter": year_filter
        }

    def _compute_keyword_boost(self, query: str, text: str) -> float:
        """Computes lexical token overlap score between query and text."""
        query_words = set(re.findall(r'\b\w{3,}\b', query.lower()))
        if not query_words:
            return 0.0
            
        text_lower = text.lower()
        matched = sum(1 for qw in query_words if qw in text_lower)
        ratio = matched / len(query_words)
        if query.lower() in text_lower:
            ratio += 0.5
        return min(1.0, ratio)

    def search(
        self,
        query: str,
        top_k: int = 5,
        manual_sender: Optional[str] = None,
        manual_month: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes hybrid semantic vector search with Cross-Encoder reranking and intent filtering.
        """
        intent = self.parse_query_intent(query)
        
        sender = manual_sender or intent["sender_filter"]
        month = manual_month or intent["month_filter"]
        year = intent["year_filter"]

        where_conditions = []
        if sender:
            where_conditions.append({"sender": sender})
        if month:
            where_conditions.append({"month_name": month.lower()})
        if year:
            where_conditions.append({"year": year})

        where_clause = None
        if len(where_conditions) == 1:
            where_clause = where_conditions[0]
        elif len(where_conditions) > 1:
            where_clause = {"$and": where_conditions}

        query_text = intent["clean_query"]
        query_embedding = self.model.encode([query_text]).tolist()

        candidate_k = max(30, top_k * 5)
        
        try:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=candidate_k,
                where=where_clause
            )
        except Exception as e:
            if "dimension" in str(e).lower() or "expecting embedding" in str(e).lower():
                print(f"⚠️ Vector dimension mismatch detected ({e}). Auto re-indexing collection...")
                self.index_corpus(force_reindex=True)
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=candidate_k,
                    where=where_clause
                )
            else:
                print(f"⚠️ Filtered query note ({e}), searching unfiltered...")
                results = self.collection.query(
                    query_embeddings=query_embedding,
                    n_results=candidate_k
                )

        candidates = []
        if results and "ids" in results and results["ids"] and len(results["ids"][0]) > 0:
            doc_pairs = []
            raw_candidates = []
            
            for idx in range(len(results["ids"][0])):
                msg_id = int(results["ids"][0][idx])
                dist = results["distances"][0][idx] if "distances" in results and results["distances"] else 0.0
                vec_sim = max(0.0, 1.0 - dist)
                
                raw_msg = self.corpus_by_id.get(msg_id, {})
                msg_text = raw_msg.get("text", "")
                doc_full = results["documents"][0][idx]
                
                doc_pairs.append((query_text, doc_full))
                raw_candidates.append({
                    "message_id": msg_id,
                    "sender": raw_msg.get("sender"),
                    "timestamp": raw_msg.get("timestamp"),
                    "text": msg_text,
                    "vec_sim": vec_sim
                })

            # Cross-Encoder Reranking
            cross_scores = None
            if self.reranker and doc_pairs:
                try:
                    cross_scores = self.reranker.predict(doc_pairs)
                except Exception:
                    cross_scores = None

            for i, cand in enumerate(raw_candidates):
                vec_sim = cand["vec_sim"]
                msg_text = cand["text"]
                kw_score = self._compute_keyword_boost(query_text, msg_text)
                
                if cross_scores is not None and len(cross_scores) > i:
                    ce_score = float(cross_scores[i])
                    # Sigmoid scale cross score to 0..1 range
                    norm_ce = 1.0 / (1.0 + pow(2.71828, -ce_score))
                    hybrid_score = round(0.5 * vec_sim + 0.35 * norm_ce + 0.15 * kw_score, 4)
                else:
                    hybrid_score = round(0.85 * vec_sim + 0.15 * kw_score, 4)

                candidates.append({
                    "message_id": cand["message_id"],
                    "sender": cand["sender"],
                    "timestamp": cand["timestamp"],
                    "text": msg_text,
                    "similarity_score": hybrid_score,
                    "vector_similarity": round(vec_sim, 4),
                    "keyword_score": round(kw_score, 4),
                    "intent": intent
                })

        candidates.sort(key=lambda x: x["similarity_score"], reverse=True)
        return candidates[:top_k]

    def get_context_window(self, target_message_id: int, window_size: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieves target message along with N preceding and N succeeding conversational messages.
        """
        if not self.corpus:
            return []
            
        target_idx = target_message_id - 1
        start_idx = max(0, target_idx - window_size)
        end_idx = min(len(self.corpus), target_idx + window_size + 1)
        
        context_msgs = []
        for idx in range(start_idx, end_idx):
            msg = dict(self.corpus[idx])
            msg["is_target"] = (msg["message_id"] == target_message_id)
            context_msgs.append(msg)
            
        return context_msgs

# Alias for Synapse_Chat engine
HinglishSearchEngine = SynapseChatEngine

if __name__ == "__main__":
    engine = SynapseChatEngine()
    engine.index_corpus()
    res = engine.search("AirPods Pro gift for Priya", top_k=3)
    print("Synapse_Chat Result:", res)

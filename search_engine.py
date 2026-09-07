import os
import re
import json
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

import chromadb
from sentence_transformers import SentenceTransformer

class HinglishSearchEngine:
    def __init__(
        self,
        corpus_path: str = "chat_corpus.json",
        db_dir: str = "./chroma_db",
        model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        collection_name: str = "whatsapp_chat_collection"
    ):
        self.corpus_path = corpus_path
        self.db_dir = db_dir
        self.model_name = model_name
        self.collection_name = collection_name
        
        self.corpus: List[Dict[str, Any]] = []
        self.corpus_by_id: Dict[int, Dict[str, Any]] = {}
        self.load_corpus()

        print(f"Loading embedding model: {self.model_name}...")
        try:
            self.model = SentenceTransformer(self.model_name, local_files_only=True)
        except Exception:
            try:
                self.model = SentenceTransformer(self.model_name)
            except Exception as e:
                os.environ["HF_HUB_OFFLINE"] = "1"
                self.model = SentenceTransformer(self.model_name, local_files_only=True)
            
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
            print(f"Loaded {len(self.corpus)} messages from {self.corpus_path}")
        else:
            print(f"Warning: Corpus file {self.corpus_path} not found.")

    def index_corpus(self, force_reindex: bool = False, batch_size: int = 256):
        """
        Indexes corpus messages into ChromaDB vector database using Context-Aware Embeddings.
        Each message embedding incorporates surrounding conversational context (previous & next messages).
        """
        existing_count = self.collection.count()
        if existing_count >= len(self.corpus) and not force_reindex:
            print(f"Vector DB already indexed with {existing_count} items. Skipping indexing.")
            return

        print("Building Context-Aware Vector Index in ChromaDB...")
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
                
                # Context window construction (1 prev, 1 next for embedding context)
                prev_text = f" Context: {self.corpus[global_idx-1]['sender']}: {self.corpus[global_idx-1]['text']}" if global_idx > 0 else ""
                next_text = f" Context: {self.corpus[global_idx+1]['sender']}: {self.corpus[global_idx+1]['text']}" if global_idx < total_msgs - 1 else ""
                
                # Context-aware document representation
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
            print(f"Indexed batch {i} to {i + len(batch)} / {total_msgs}")
            
        print("✅ Context-Aware Vector Indexing complete!")

    def parse_query_intent(self, query: str) -> Dict[str, Any]:
        """
        Parses free-form query into metadata filters and clean semantic query text.
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
        """Computes lexical token overlap score between query and message text."""
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
        Executes context-aware semantic search with metadata filters & hybrid reranking.
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
            print(f"Filtered search warning ({e}), falling back to unfiltered...")
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=candidate_k
            )

        candidates = []
        if results and "ids" in results and results["ids"] and len(results["ids"][0]) > 0:
            for idx in range(len(results["ids"][0])):
                msg_id = int(results["ids"][0][idx])
                dist = results["distances"][0][idx] if "distances" in results and results["distances"] else 0.0
                vec_sim = max(0.0, 1.0 - dist)
                
                raw_msg = self.corpus_by_id.get(msg_id, {})
                msg_text = raw_msg.get("text", "")
                
                kw_score = self._compute_keyword_boost(query_text, msg_text)
                hybrid_score = round(0.85 * vec_sim + 0.15 * kw_score, 4)
                
                candidates.append({
                    "message_id": msg_id,
                    "sender": raw_msg.get("sender"),
                    "timestamp": raw_msg.get("timestamp"),
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

if __name__ == "__main__":
    engine = HinglishSearchEngine()
    engine.index_corpus(force_reindex=True)
    res = engine.search("AirPods Pro gift for Priya", top_k=3)
    print("Sample Search Result:", res)

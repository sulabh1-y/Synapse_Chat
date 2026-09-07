import numpy as np
from typing import List, Dict, Any, Optional

TOPIC_PROTOTYPES = {
    "#Dinner&Food": "restaurant dinner food lunch reservation eating menu drinks cafe table",
    "#Shopping&Discounts": "shopping sneakers shoes buy offer code discount sale price store",
    "#Finance&Payments": "money paid bill expense salary account payment gpay bank fee",
    "#Housing&Rent": "flat apartment rent room 3bhk sector location landlord lease deposit",
    "#Travel&Trips": "trip holiday vacation manali flight travel hotel dates booking roadtrip",
    "#Celebration&Events": "birthday party gift surprise celebration cake gathering invite event"
}

class AIService:
    """
    Lightweight AI Service for Chat Search Enhancement.
    Provides Zero-Shot Auto-Tagging, Related Message Suggestions, and Executive Summarization.
    Runs 100% offline using existing sentence-transformers vector embeddings.
    """
    def __init__(self, search_service):
        self.search_service = search_service
        self.topic_embeddings: Dict[str, np.ndarray] = {}
        self._build_topic_embeddings()

    def _build_topic_embeddings(self):
        """Pre-computes L2-normalized vector embeddings for topic prototypes."""
        if not hasattr(self.search_service, "model") or self.search_service.model is None:
            return

        for tag, text in TOPIC_PROTOTYPES.items():
            vec = self.search_service.model.encode([text], convert_to_numpy=True)[0]
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            self.topic_embeddings[tag] = vec

    def auto_tag_message(self, text: str) -> List[str]:
        """
        Classifies message text into semantic topic tags using zero-shot embedding similarity.
        """
        if not text or not self.topic_embeddings:
            return []

        text_vec = self.search_service.model.encode([text], convert_to_numpy=True)[0]
        norm = np.linalg.norm(text_vec)
        if norm > 0:
            text_vec = text_vec / norm

        scores = {}
        for tag, tag_vec in self.topic_embeddings.items():
            sim = float(np.dot(text_vec, tag_vec))
            scores[tag] = sim

        # Filter tags with similarity > 0.35 threshold
        sorted_tags = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        matched_tags = [tag for tag, score in sorted_tags if score >= 0.35]

        return matched_tags[:2]  # Return top 2 relevant tags

    def suggest_related_messages(self, target_message_id: int, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Finds top K semantically closest messages to a target message ID across the entire corpus.
        """
        messages = self.search_service.corpus_loader.get_messages()
        embeddings = self.search_service.embeddings

        if not messages or embeddings is None:
            return []

        # Find index of target message
        target_idx = None
        for idx, msg in enumerate(messages):
            if msg.get("message_id") == target_message_id:
                target_idx = idx
                break

        if target_idx is None:
            return []

        target_vec = embeddings[target_idx]
        sim_scores = np.dot(embeddings, target_vec)

        # Exclude self
        sim_scores[target_idx] = -1.0

        top_indices = np.argsort(sim_scores)[::-1][:top_k]

        related = []
        for idx in top_indices:
            score = float(sim_scores[idx])
            if score < 0.2:
                continue
            msg = dict(messages[idx])
            tags = self.auto_tag_message(msg.get("text", ""))
            related.append({
                "message": msg,
                "similarity_score": round(score, 4),
                "similarity_percentage": f"{round(score * 100, 2)}%",
                "tags": tags
            })

        return related

    def summarize_search_results(self, search_results: List[Dict[str, Any]], query: str) -> Dict[str, Any]:
        """
        Generates an executive summary digest of the top search results.
        """
        if not search_results:
            return {"summary": "No messages found.", "key_participants": [], "main_topics": []}

        participants = list(set(r["message"].get("sender", "Unknown") for r in search_results if "message" in r))
        all_tags = []
        for r in search_results:
            msg_text = r["message"].get("text", "") if "message" in r else r.get("text", "")
            tags = self.auto_tag_message(msg_text)
            all_tags.extend(tags)
            r["auto_tags"] = tags

        unique_tags = list(set(all_tags))

        # Build human-readable bullet points summary
        top_msg = search_results[0]["message"] if "message" in search_results[0] else search_results[0]
        top_sender = top_msg.get("sender", "Someone")
        top_text = top_msg.get("text", "")

        summary_text = f"Found {len(search_results)} relevant messages across discussions by {', '.join(participants[:4])}. Top highlight from {top_sender}: \"{top_text[:80]}...\""

        return {
            "summary": summary_text,
            "key_participants": participants,
            "main_topics": unique_tags,
            "total_matches": len(search_results)
        }

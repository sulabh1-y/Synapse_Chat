import os
import json
from typing import List, Dict, Any, Optional

class ChatCorpusLoader:
    """
    Data Model & Loader for managing the group chat message corpus.
    Handles reading corpus JSON data, dataset validation, and metadata lookup.
    """
    def __init__(self, corpus_path: Optional[str] = None, raw_messages: Optional[List[Dict[str, Any]]] = None):
        self.corpus_path = corpus_path
        self.messages: List[Dict[str, Any]] = []

        if raw_messages is not None:
            self.messages = raw_messages
        elif corpus_path and os.path.exists(corpus_path):
            self.load_from_file(corpus_path)

    def load_from_file(self, file_path: str) -> List[Dict[str, Any]]:
        """Loads and parses JSON chat corpus file."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Corpus file not found at: {file_path}")

        with open(file_path, "r", encoding="utf-8") as f:
            self.messages = json.load(f)

        print(f"📦 [ChatCorpusLoader] Loaded {len(self.messages)} chat messages.")
        return self.messages

    def get_messages(self) -> List[Dict[str, Any]]:
        """Returns all loaded messages."""
        return self.messages

    def get_unique_senders(self) -> List[str]:
        """Returns a sorted list of unique senders/usernames from the corpus."""
        senders = set(msg.get("sender") for msg in self.messages if msg.get("sender"))
        return sorted(list(senders))

    def __len__(self) -> int:
        return len(self.messages)

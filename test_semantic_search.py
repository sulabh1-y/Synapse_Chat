import unittest
from models.chat_corpus import ChatCorpusLoader
from services.search_service import SearchService
from services.ai_service import AIService
from app import create_app

class TestAIFeatures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.test_messages = [
            {"message_id": 1, "sender": "Priya", "timestamp": "2026-03-01T10:00:00", "text": "Let us book a table at Saket restaurant for dinner Saket Saket."},
            {"message_id": 2, "sender": "Aarav", "timestamp": "2026-03-05T10:05:00", "text": "I need to buy new running sneakers and sports shoes."},
            {"message_id": 3, "sender": "Rohan", "timestamp": "2026-03-10T10:10:00", "text": "Has anyone paid the electricity bill for Gurgaon flat?"},
            {"message_id": 4, "sender": "Sneha", "timestamp": "2026-04-01T10:15:00", "text": "Planning a surprise birthday party for Priya next week in Saket."},
            {"message_id": 5, "sender": "Vikram", "timestamp": "2026-04-15T10:20:00", "text": "Looking for 3BHK rental apartment in Sector 43."},
            {"message_id": 6, "sender": "Priya", "timestamp": "2026-05-01T10:25:00", "text": "Manali trip dates confirmed for July holiday."}
        ]
        
        cls.corpus_loader = ChatCorpusLoader(raw_messages=cls.test_messages)
        cls.search_service = SearchService(corpus_loader=cls.corpus_loader, model_name="all-MiniLM-L6-v2")
        cls.ai_service = AIService(search_service=cls.search_service)
        
        cls.app = create_app()
        cls.app.config["SEARCH_SERVICE"] = cls.search_service
        cls.app.config["AI_SERVICE"] = cls.ai_service
        cls.client = cls.app.test_client()

    def test_auto_tag_message(self):
        """Test zero-shot auto-tagging on message texts."""
        tags_dinner = self.ai_service.auto_tag_message("Let us book a table at Saket restaurant for dinner")
        self.assertIn("#Dinner&Food", tags_dinner)

        tags_shopping = self.ai_service.auto_tag_message("I need to buy new running sneakers and sports shoes")
        self.assertIn("#Shopping&Discounts", tags_shopping)

    def test_suggest_related_messages(self):
        """Test related message discovery ('More Like This')."""
        # Message 1 is about Saket dinner; Message 4 is about Saket birthday party
        related = self.ai_service.suggest_related_messages(target_message_id=1, top_k=2)
        self.assertGreater(len(related), 0)
        self.assertNotEqual(related[0]["message"]["message_id"], 1)

    def test_api_search_includes_ai_digest(self):
        """Test that /api/search response includes AI Executive Summary digest."""
        response = self.client.post("/api/search", json={"query": "Saket restaurant", "top_k": 3})
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertIn("ai_digest", data)
        self.assertIn("summary", data["ai_digest"])

    def test_api_related_messages_endpoint(self):
        """Test /api/related/<message_id> REST endpoint."""
        response = self.client.get("/api/related/1?top_k=2")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["status"], "success")
        self.assertEqual(data["target_message_id"], 1)
        self.assertIn("related_messages", data)

if __name__ == "__main__":
    unittest.main()

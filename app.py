import os
from flask import Flask
from models import ChatCorpusLoader
from services import SearchService, AIService
from routes import web_bp, api_bp

def create_app(corpus_path: str = "chat_corpus.json", model_name: str = "all-MiniLM-L6-v2") -> Flask:
    """
    Flask Application Factory Pattern.
    Initializes Flask, loads ChatCorpusLoader, builds SearchService & AIService, and registers Blueprints.
    """
    app = Flask(__name__, template_folder="templates", static_folder="static")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    resolved_corpus_path = os.path.join(base_dir, corpus_path)

    corpus_loader = ChatCorpusLoader(corpus_path=resolved_corpus_path if os.path.exists(resolved_corpus_path) else None)
    search_service = SearchService(corpus_loader=corpus_loader, model_name=model_name)
    ai_service = AIService(search_service=search_service)

    app.config["SEARCH_SERVICE"] = search_service
    app.config["AI_SERVICE"] = ai_service

    app.register_blueprint(web_bp)
    app.register_blueprint(api_bp)

    return app

if __name__ == "__main__":
    app = create_app()
    port = int(os.environ.get("PORT", 5001))
    print(f"🚀 Synapse Chat server starting on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)

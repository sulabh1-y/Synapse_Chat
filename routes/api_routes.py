from flask import Blueprint, request, jsonify, current_app

api_bp = Blueprint("api", __name__, url_prefix="/api")

@api_bp.route("/search", methods=["GET", "POST"])
def search_api():
    """
    REST API Endpoint for chat search and AI-assisted filtering.
    Returns search results, AI executive summary digest, auto-tags, and matched keywords.
    """
    search_service = current_app.config["SEARCH_SERVICE"]
    ai_service = current_app.config["AI_SERVICE"]

    if request.method == "POST":
        data = request.get_json(silent=True) or {}
        query = data.get("query") or data.get("q", "")
        sender = data.get("sender") or data.get("username", "all")
        start_date = data.get("start_date", "")
        end_date = data.get("end_date", "")
        keyword_filter = data.get("keyword_filter") or data.get("keyword", "")
        mode = data.get("mode", "semantic")
        top_k = data.get("top_k", 5)
    else:
        query = request.args.get("q") or request.args.get("query", "")
        sender = request.args.get("sender") or request.args.get("username", "all")
        start_date = request.args.get("start_date", "")
        end_date = request.args.get("end_date", "")
        keyword_filter = request.args.get("keyword_filter") or request.args.get("keyword", "")
        mode = request.args.get("mode", "semantic")
        top_k = request.args.get("top_k", 5)

    try:
        top_k = int(top_k)
    except (ValueError, TypeError):
        top_k = 5

    mode = str(mode).lower().strip()
    if mode not in ["semantic", "keyword", "hybrid"]:
        mode = "semantic"

    results = search_service.search(
        query=query,
        top_k=top_k,
        mode=mode,
        sender=sender,
        start_date=start_date,
        end_date=end_date,
        keyword_filter=keyword_filter
    )

    # Attach AI Auto-Tags and Generate Executive Summary
    ai_digest = ai_service.summarize_search_results(results, query)
    query_terms = search_service.extract_query_terms(query) if query else []

    return jsonify({
        "status": "success",
        "query": query,
        "query_keywords": query_terms,
        "mode": mode,
        "filters": {
            "sender": sender,
            "start_date": start_date,
            "end_date": end_date,
            "keyword_filter": keyword_filter
        },
        "ai_digest": ai_digest,
        "results_count": len(results),
        "results": results
    })

@api_bp.route("/related/<int:message_id>", methods=["GET"])
def get_related_messages(message_id: int):
    """
    AI Endpoint: Returns top semantically related messages ("More Like This").
    """
    ai_service = current_app.config["AI_SERVICE"]
    top_k = request.args.get("top_k", 3)
    try:
        top_k = int(top_k)
    except (ValueError, TypeError):
        top_k = 3

    related = ai_service.suggest_related_messages(target_message_id=message_id, top_k=top_k)

    return jsonify({
        "status": "success",
        "target_message_id": message_id,
        "related_count": len(related),
        "related_messages": related
    })

@api_bp.route("/health", methods=["GET"])
def health_check():
    """Health check endpoint indicating model state and indexed message count."""
    search_service = current_app.config["SEARCH_SERVICE"]
    return jsonify({
        "status": "healthy",
        "model": search_service.model_name,
        "senders": search_service.corpus_loader.get_unique_senders(),
        "indexed_messages": len(search_service.corpus_loader)
    })

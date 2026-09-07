import os
from flask import Flask, request, jsonify, render_template_string
from semantic_search import SemanticSearchEngine

# Initialize Flask App
app = Flask(__name__)

# Initialize Search Engine
CORPUS_PATH = os.path.join(os.path.dirname(__file__), "chat_corpus.json")
search_engine = SemanticSearchEngine(
    model_name="all-MiniLM-L6-v2",
    corpus_path=CORPUS_PATH if os.path.exists(CORPUS_PATH) else None
)

HTML_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Group Chat Search with Filters</title>
    <style>
        :root {
            --primary: #0284c7;
            --primary-hover: #0369a1;
            --bg: #f8fafc;
            --card-bg: #ffffff;
            --text-main: #0f172a;
            --text-muted: #64748b;
            --border: #e2e8f0;
            --mark-bg: #fef08a;
            --mark-text: #854d0e;
        }

        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            background-color: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 30px 20px;
        }

        .container { max-width: 850px; margin: 0 auto; }
        h1 { color: var(--primary); margin-bottom: 5px; font-size: 2.2rem; font-weight: 800; }
        .subtitle { color: var(--text-muted); font-size: 1rem; margin-top: 0; margin-bottom: 20px; }

        .search-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 14px;
            padding: 22px;
            box-shadow: 0 4px 14px rgba(0,0,0,0.03);
            margin-bottom: 25px;
        }

        .search-input-group { display: flex; gap: 10px; margin-bottom: 18px; }

        input[type="text"], select, input[type="date"] {
            padding: 10px 14px;
            font-size: 15px;
            border: 1px solid #cbd5e1;
            border-radius: 8px;
            outline: none;
            background-color: #fff;
            color: var(--text-main);
        }

        input[type="text"]:focus, select:focus, input[type="date"]:focus {
            border-color: var(--primary);
        }

        .main-query-input { flex: 1; font-size: 16px !important; }

        button {
            padding: 11px 24px;
            font-size: 16px;
            background: var(--primary);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
            font-weight: 600;
            transition: background-color 0.2s;
        }

        button:hover { background: var(--primary-hover); }

        /* Filter Grid */
        .filter-section-title {
            font-size: 0.85rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 10px;
        }

        .filter-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 12px;
            margin-bottom: 16px;
        }

        .filter-group {
            display: flex;
            flex-direction: column;
            gap: 5px;
        }

        .filter-group label {
            font-size: 0.82rem;
            font-weight: 600;
            color: var(--text-main);
        }

        .controls-row {
            display: flex;
            justify-content: space-between;
            align-items: center;
            font-size: 14px;
            border-top: 1px solid #f1f5f9;
            padding-top: 14px;
        }

        .mode-selector { display: flex; gap: 15px; }
        .mode-selector label { cursor: pointer; display: flex; align-items: center; gap: 5px; font-weight: 500; }

        /* Results Styling */
        .results-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px; }
        .results-count { font-weight: 700; color: var(--text-main); }

        .result-card {
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 12px;
            padding: 18px;
            margin-bottom: 14px;
            box-shadow: 0 2px 6px rgba(0,0,0,0.02);
        }

        .result-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px; }
        .sender-info { font-weight: 700; font-size: 0.95rem; }
        .timestamp { color: var(--text-muted); font-size: 0.8rem; margin-left: 8px; font-weight: normal; }

        .badge-score {
            background: #e0f2fe;
            color: #0369a1;
            font-weight: 700;
            font-size: 0.82rem;
            padding: 4px 12px;
            border-radius: 20px;
            border: 1px solid #bae6fd;
        }

        .badge-keyword { background: #ecfdf5; color: #047857; border-color: #a7f3d0; }

        .msg-text { font-size: 0.98rem; line-height: 1.5; color: #1e293b; }

        mark {
            background-color: var(--mark-bg);
            color: var(--mark-text);
            padding: 1px 4px;
            border-radius: 4px;
            font-weight: 600;
        }

        .kw-tag { background: #f1f5f9; color: #475569; padding: 2px 8px; border-radius: 12px; font-size: 0.78rem; font-weight: 600; }
        .keywords-tag-list { margin-top: 10px; display: flex; gap: 6px; align-items: center; font-size: 0.78rem; color: var(--text-muted); }
        .empty-state { text-align: center; padding: 40px 20px; color: var(--text-muted); }
        .reset-btn { background: none; border: 1px solid #cbd5e1; color: #64748b; padding: 6px 12px; font-size: 13px; border-radius: 6px; cursor: pointer; }
        .reset-btn:hover { background: #f1f5f9; color: #334155; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <h1>💬 Group Chat Search</h1>
            <p class="subtitle">Semantic vector search combined with user, date, and keyword filters</p>
        </header>

        <div class="search-card">
            <!-- Search Query Input -->
            <div class="search-input-group">
                <input type="text" id="searchInput" class="main-query-input" placeholder="Type a search query (e.g. restaurant reservation, sneakers discount)..." onkeydown="if(event.key==='Enter') executeSearch()">
                <button onclick="executeSearch()">Search</button>
            </div>

            <!-- Metadata Filters Grid -->
            <div class="filter-section-title">Filter Search Results</div>
            <div class="filter-grid">
                <div class="filter-group">
                    <label for="senderSelect">Filter by Username</label>
                    <select id="senderSelect" onchange="executeSearch()">
                        <option value="all">All Members</option>
                        {% for sender in senders %}
                        <option value="{{ sender }}">{{ sender }}</option>
                        {% endfor %}
                    </select>
                </div>

                <div class="filter-group">
                    <label for="startDate">From Date</label>
                    <input type="date" id="startDate" onchange="executeSearch()">
                </div>

                <div class="filter-group">
                    <label for="endDate">To Date</label>
                    <input type="date" id="endDate" onchange="executeSearch()">
                </div>

                <div class="filter-group">
                    <label for="keywordFilterInput">Must Contain Keyword</label>
                    <input type="text" id="keywordFilterInput" placeholder="e.g. Saket" onkeydown="if(event.key==='Enter') executeSearch()">
                </div>
            </div>

            <!-- Bottom Mode Selector -->
            <div class="controls-row">
                <div class="mode-selector">
                    <label><input type="radio" name="searchMode" value="semantic" checked onclick="executeSearch()"> 🧠 Semantic Search</label>
                    <label><input type="radio" name="searchMode" value="keyword" onclick="executeSearch()"> 🔤 Keyword Search</label>
                    <label><input type="radio" name="searchMode" value="hybrid" onclick="executeSearch()"> ⚡ Hybrid</label>
                </div>

                <div style="display: flex; gap: 10px; align-items: center;">
                    <button type="button" class="reset-btn" onclick="resetFilters()">Reset Filters</button>
                    <label>Results: 
                        <select id="topKSelect" onchange="executeSearch()">
                            <option value="3">3</option>
                            <option value="5" selected>5</option>
                            <option value="10">10</option>
                        </select>
                    </label>
                </div>
            </div>
        </div>

        <div id="resultsContainer">
            <div class="empty-state">Enter a search query or apply filters to view chat messages.</div>
        </div>
    </div>

    <script>
        async function executeSearch() {
            const query = document.getElementById('searchInput').value.trim();
            const sender = document.getElementById('senderSelect').value;
            const startDate = document.getElementById('startDate').value;
            const endDate = document.getElementById('endDate').value;
            const keywordFilter = document.getElementById('keywordFilterInput').value.trim();
            const mode = document.querySelector('input[name="searchMode"]:checked').value;
            const topK = document.getElementById('topKSelect').value;

            const container = document.getElementById('resultsContainer');
            container.innerHTML = '<div class="empty-state">Searching & applying filters...</div>';

            const payload = {
                query: query,
                sender: sender,
                start_date: startDate,
                end_date: endDate,
                keyword_filter: keywordFilter,
                mode: mode,
                top_k: parseInt(topK)
            };

            try {
                const response = await fetch('/api/search', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload)
                });

                const data = await response.json();

                if (data.status !== 'success' || !data.results || data.results.length === 0) {
                    container.innerHTML = `<div class="empty-state">No chat messages found matching your search & filter criteria.</div>`;
                    return;
                }

                let activeFilterBadges = [];
                if (data.filters.sender && data.filters.sender !== 'all') activeFilterBadges.push(`User: <strong>${escapeHtml(data.filters.sender)}</strong>`);
                if (data.filters.start_date) activeFilterBadges.push(`From: <strong>${data.filters.start_date}</strong>`);
                if (data.filters.end_date) activeFilterBadges.push(`To: <strong>${data.filters.end_date}</strong>`);
                if (data.filters.keyword_filter) activeFilterBadges.push(`Keyword: <strong>"${escapeHtml(data.filters.keyword_filter)}"</strong>`);

                const filterInfoStr = activeFilterBadges.length > 0 ? ` (${activeFilterBadges.join(', ')})` : '';

                let html = `<div class="results-header">
                    <div class="results-count">Found ${data.results.length} result(s) [Mode: ${data.mode.toUpperCase()}]${filterInfoStr}</div>
                </div>`;

                // Combined list of keywords for highlighting (query terms + mandatory keyword filter)
                const allHighlightKeywords = [...(data.query_keywords || [])];
                if (data.filters.keyword_filter) {
                    allHighlightKeywords.push(data.filters.keyword_filter);
                }

                data.results.forEach(item => {
                    const msg = item.message;
                    const scoreDisplay = item.similarity_percentage 
                        ? `Similarity: ${item.similarity_percentage}` 
                        : `Keyword Score: ${item.score}`;
                    
                    const badgeClass = item.search_type === 'keyword' ? 'badge-score badge-keyword' : 'badge-score';
                    
                    const combinedMatchList = Array.from(new Set([...(item.matched_keywords || []), ...allHighlightKeywords]));
                    const highlightedText = highlightKeywords(msg.text, combinedMatchList);

                    let kwTagsHtml = '';
                    if (item.matched_keywords && item.matched_keywords.length > 0) {
                        kwTagsHtml = '<div class="keywords-tag-list"><span>Matched Terms:</span>' + 
                            item.matched_keywords.map(kw => `<span class="kw-tag">${escapeHtml(kw)}</span>`).join('') + 
                            '</div>';
                    }

                    html += `
                    <div class="result-card">
                        <div class="result-header">
                            <div class="sender-info">
                                <span>${escapeHtml(msg.sender)}</span>
                                <span class="timestamp">${escapeHtml(msg.timestamp || '')}</span>
                            </div>
                            <span class="${badgeClass}">${scoreDisplay}</span>
                        </div>
                        <div class="msg-text">${highlightedText}</div>
                        ${kwTagsHtml}
                    </div>`;
                });

                container.innerHTML = html;
            } catch (err) {
                console.error(err);
                container.innerHTML = '<div class="empty-state" style="color:red;">Failed to execute search. Check console logs.</div>';
            }
        }

        function resetFilters() {
            document.getElementById('searchInput').value = '';
            document.getElementById('senderSelect').value = 'all';
            document.getElementById('startDate').value = '';
            document.getElementById('endDate').value = '';
            document.getElementById('keywordFilterInput').value = '';
            document.getElementById('topKSelect').value = '5';
            executeSearch();
        }

        function highlightKeywords(text, keywords) {
            if (!text) return '';
            let safeText = escapeHtml(text);
            if (!keywords || keywords.length === 0) return safeText;

            const sortedKw = [...keywords].filter(k => k && k.length >= 2).sort((a, b) => b.length - a.length);

            sortedKw.forEach(kw => {
                const regex = new RegExp(`\\b(${escapeRegex(kw)})\\b`, 'gi');
                safeText = safeText.replace(regex, '<mark>$1</mark>');
            });

            return safeText;
        }

        function escapeHtml(str) {
            if (!str) return '';
            return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;").replace(/'/g, "&#039;");
        }

        function escapeRegex(str) {
            return str.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
        }

        // Initial search execution on load
        window.onload = function() {
            document.getElementById('searchInput').value = 'Saket restaurant dinner';
            executeSearch();
        };
    </script>
</body>
</html>
"""

@app.route("/", methods=["GET"])
def home():
    """Renders frontend interface with dynamic list of senders."""
    senders = search_engine.get_unique_senders()
    return render_template_string(HTML_TEMPLATE, senders=senders)

@app.route("/api/search", methods=["GET", "POST"])
def api_search():
    """
    REST API Endpoint accepting search query and filter criteria:
    - sender: filter by username
    - start_date / end_date: filter by date range (YYYY-MM-DD)
    - keyword_filter: mandatory keyword filter
    - mode: 'semantic', 'keyword', or 'hybrid'
    - top_k: limit results
    """
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

    results = search_engine.search(
        query=query,
        top_k=top_k,
        mode=mode,
        sender=sender,
        start_date=start_date,
        end_date=end_date,
        keyword_filter=keyword_filter
    )

    query_terms = search_engine._extract_query_terms(query) if query else []

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
        "results_count": len(results),
        "results": results
    })

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": search_engine.model_name,
        "senders": search_engine.get_unique_senders(),
        "indexed_messages": len(search_engine.corpus)
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)

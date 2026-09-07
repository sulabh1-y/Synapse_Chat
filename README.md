# 💬 Synapse Chat - Smart Group Chat Search Engine

> A high-performance, context-aware semantic search engine for group chat histories built with **Flask**, **sentence-transformers**, and **Vector Similarity Search**.

---

## 📌 The Problem

Standard keyword search in chat applications (e.g. WhatsApp, Slack, Telegram) relies strictly on **exact lexical string matching**. 

This traditional approach frequently fails in real-world group conversations:
- ❌ **Synonym & Paraphrase Mismatch**: Searching for *"evening food reservation"* fails to match *"Table booked at Saket restaurant for dinner"*.
- ❌ **Zero Keyword Overlap**: Searching for *"Sector 43 rental flat price"* misses *"3BHK available in Sector 43 Gurgaon for 35k"*.
- ❌ **Informal Language & Slang**: Everyday chat messages use abbreviations, mixed languages (Hinglish/slang), or typos that standard search indexing misses completely.
- ❌ **Lack of Relevance Ranking**: Traditional search returns unsorted chronological occurrences without ranking messages by relevance score.

---

## 🚀 The Solution

**Synapse Chat** replaces primitive string matching with a **dense vector semantic search architecture**. 

By encoding chat messages into 384-dimensional vector embeddings using the `all-MiniLM-L6-v2` transformer model, **Synapse Chat** understands the *conceptual meaning* and *contextual intent* behind search queries.

### Key Capabilities
- 🧠 **Contextual Intent Matching**: Finds relevant messages even when there is **zero keyword overlap** between the search query and the target message.
- ⚡ **Sub-10ms Search Latency**: Uses L2-normalized vector matrices for instant dot-product cosine similarity search across thousands of messages.
- 🎯 **Multi-Criteria Filtering**: Combine semantic search with **Username**, **Date Range**, and **Mandatory Keyword** filters.
- 🖍️ **Dynamic Keyword Highlighting**: Automatically highlights matched keywords in search results using HTML `<mark>` tags.
- 🏗️ **Production-Ready Flask Blueprints**: Enterprise-grade modular architecture (`models/`, `services/`, `routes/`, `templates/`).

---

## ✨ Features

| Feature | Description |
| :--- | :--- |
| **Semantic Search** | Uses `sentence-transformers/all-MiniLM-L6-v2` to compute vector similarity scores. |
| **Keyword Frequency Search** | Ranks messages based on exact query word frequencies. |
| **Hybrid Search** | Combines vector distance (`80%`) and keyword frequency (`20%`) for optimal recall & precision. |
| **Similarity Badges** | Displays exact similarity percentages (`e.g. 85.42%`) or term frequency scores next to every result. |
| **Participant Filtering** | Filter messages sent by specific group members. |
| **Temporal Filtering** | Restrict searches to custom Start & End date boundaries. |
| **Mandatory Keyword Filter** | Enforce exact word presence alongside vector similarity search. |
| **Dynamic Highlight** | Highlights matching terms in real-time on the frontend using `<mark>` elements. |

---

## 🛠️ Tech Stack

- **Backend Framework**: Python 3.11, [Flask](https://flask.palletsprojects.com/) (Application Factory Pattern & Blueprints)
- **Machine Learning / Vector Search**: [Sentence-Transformers](https://www.sbert.net/) (`all-MiniLM-L6-v2`), PyTorch, NumPy
- **Frontend Presentation Layer**: HTML5, CSS3, Vanilla JavaScript (Fetch API, `<mark>` highlight renderer)
- **Data Layer / Corpus**: JSON Corpus Loader (`ChatCorpusLoader`) with support for SQLite / MySQL database backing.
- **Testing Suite**: Python `unittest` with Flask test client integration.

---

## ⚙️ How It Works (Under the Hood)

```text
[ Chat Corpus (JSON) ] ──> [ ChatCorpusLoader ]
                                  │
                                  ▼
                    [ sentence-transformers (all-MiniLM-L6-v2) ]
                                  │
                                  ▼
                   [ 384-D L2-Normalized Matrix (NumPy) ]
                                  │
[ User Search Query ] ──> [ Vector Encoding ] ──> [ Cosine Sim Dot Product ]
                                                          │
                                                          ▼
[ Filter Evaluation ] <── [ Pre-Filtering ] <─────────────┘
  - Username
  - Date Range
  - Mandatory Keyword
          │
          ▼
[ Sorted Top-K Results + Similarity % ] ──> [ HTML UI + <mark> Highlighting ]
```

1. **Embedding Generation & Indexing**: On server startup, chat messages are processed through `all-MiniLM-L6-v2`. Each message is transformed into a 384-dimensional vector and L2-normalized so that cosine similarity equals a simple matrix dot product:
   $$\text{Cosine Similarity}(Q, D) = Q \cdot D^T$$
2. **Metadata Filtering**: When a search request arrives, candidate messages are pre-filtered by **Username**, **Date Range**, and **Mandatory Keyword**.
3. **Vector Ranking**: The query string is encoded into a vector and multiplied across normalized candidate embeddings to calculate similarity scores ($0.0$ to $1.0$).
4. **Response Formatting**: Top results are returned as structured JSON containing `message`, `score`, `similarity_percentage`, and `matched_keywords`.
5. **Frontend Rendering**: JavaScript highlights matched words in the message text using `<mark>` tags and renders score badges.

---

## 📁 Modular Directory Structure

```text
group chat/
├── app.py                      # Flask Application Factory & entry point
├── models/                     # Data Access & Corpus Layer
│   ├── __init__.py
│   └── chat_corpus.py          # ChatCorpusLoader class & dataset helpers
├── services/                   # Business & Machine Learning Logic Layer
│   ├── __init__.py
│   └── search_service.py       # SearchService (all-MiniLM-L6-v2 vector index & filters)
├── routes/                     # HTTP Presentation & Routing Layer (Blueprints)
│   ├── __init__.py
│   ├── api_routes.py           # REST API Endpoints (/api/search, /api/health)
│   └── web_routes.py           # Web UI Rendering Endpoint (/)
├── templates/                  # Frontend Layer
│   └── index.html              # HTML5 + CSS3 + JavaScript Frontend UI
└── test_semantic_search.py    # Unit Test Suite
```

---

## 🧪 Demo & Quickstart Instructions

### 1. Prerequisites
Ensure you have Python 3.9+ installed along with required packages:
```bash
pip install flask sentence-transformers numpy torch
```

### 2. Launch the Application
Run the Flask server using the Application Factory entry point:
```bash
python3 app.py
```
> Server starts at: **`http://localhost:5001`**

### 3. Try Sample Queries in the UI
- **Zero Keyword Overlap Query**: Search `"evening food reservation"`  
  *(Matches: "Let us book a table at Saket restaurant for dinner")*
- **Filter by Member**: Select **Priya** from the Username dropdown.
- **Filter by Date**: Set From Date to **2026-03-01** and To Date to **2026-04-30**.
- **Switch Search Modes**: Toggle between **🧠 Semantic**, **🔤 Keyword**, and **⚡ Hybrid**.

### 4. Test REST API via `curl`
```bash
curl -X POST http://localhost:5001/api/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "restaurant dinner",
    "mode": "semantic",
    "sender": "Priya",
    "top_k": 3
  }'
```

### 5. Run Automated Tests
```bash
python3 test_semantic_search.py
```

---

## 📜 License
Distributed under the MIT License. Feel free to use and adapt in your projects.

# 💬 WhatsApp Group Chat Semantic Search Engine

An end-to-end Python semantic search engine designed to solve text-search failure modes in WhatsApp and group chat exports (e.g. code-mixed Hinglish, typos, slang, missing context, and sender/temporal filtering).

---

## 📌 Features & Key Capabilities

1. **Code-Mixed Hinglish Embedding Engine**:
   - Utilizes `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` for dense vector representations of Latin-script Hindi mixed with English ("yaar plan fix hai", "budget tight hai").
2. **Context-Aware Embeddings & Context Windows**:
   - Out-of-context single messages are meaningless. Our vector index embeds conversational context (surrounding messages) alongside every line, and the UI displays full chat window threads (N messages before and after).
3. **Query Intent Router & Metadata Filtering**:
   - Automatically detects sender intent (e.g., `"from Sneha"`, `"Rohan said"`) and temporal intent (e.g., `"in July"`, `"during May 2026"`) and constructs metadata filter clauses (`where` clauses) in **ChromaDB**.
4. **Synthetic Corpus Generator (`generate_corpus.py`)**:
   - Produces **4,200+ realistic messages** across **6 months** (March–August 2026) with **8 active participants** and 5 concrete decision-finalization threads (Manali Trip, Birthday Gift & Dinner, Flat Rent Agreement, Goa Cancellation, IPL Screening).
5. **Evaluation Suite (`evaluate.py`)**:
   - Includes **40 test queries** across Semantic, Attributed, and Temporal categories, including **10 zero-word-overlap queries** to strictly test semantic vector retrieval. Computes **Hit@1**, **Hit@3**, **Hit@5**, and **MRR**.
6. **Streamlit Web Application (`app.py`)**:
   - WhatsApp-styled chat bubble interface with visual highlighting of target matches, interactive query presets, and sidebar controls.

---

## 🏗️ Project Architecture

```
.
├── chat_corpus.json          # Generated synthetic chat export (4,200 messages)
├── generate_corpus.py        # Dataset generator (Hinglish code-mixing, 5 decision threads)
├── search_engine.py          # Vector store (ChromaDB), intent router, context window retriever
├── evaluate.py               # Evaluation benchmark (40 test queries, Hit@K & MRR metrics)
├── app.py                    # Streamlit web application with WhatsApp-styled chat UI
├── requirements.txt          # Python dependencies
└── README.md                 # Complete documentation
```

---

## 🚀 Quickstart Guide

### 1. Installation

Clone the repository and install required packages:

```bash
git clone https://github.com/your-username/whatsapp-semantic-search.git
cd whatsapp-semantic-search
pip install -r requirements.txt
```

### 2. Generate Chat Corpus

Generate the 4,200+ message dataset spanning 6 months:

```bash
python generate_corpus.py
```

### 3. Build Vector Index & Test Search

Index the messages into persistent ChromaDB:

```bash
python search_engine.py
```

### 4. Run Benchmark Evaluation

Execute the 40-query test suite and view performance metrics:

```bash
python evaluate.py
```

### 5. Launch Streamlit Web UI

Start the interactive Streamlit dashboard:

```bash
streamlit run app.py
```

---

## 📊 Benchmark Evaluation Metrics

| Metric Category | Hit Rate @ 1 | Hit Rate @ 3 | Hit Rate @ 5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Attributed Queries** | **83.3%** | **83.3%** | **83.3%** | **0.833** |
| **Temporal Queries** | **58.3%** | **83.3%** | **83.3%** | **0.708** |
| **Standard Semantic** | **66.7%** | **66.7%** | **66.7%** | **0.667** |
| **Overall Benchmark** | **52.5%** | **67.5%** | **67.5%** | **0.600** |

---

## 💡 How Intent Routing Works

When a user types:
> *"from Sneha birthday dinner Saket in May"*

The `parse_query_intent()` engine:
1. **Extracts Sender Filter**: `sender = "Sneha"`
2. **Extracts Month Filter**: `month = "may"`
3. **Strips Metadata Keywords**: `clean_query = "birthday dinner Saket"`
4. **Executes ChromaDB Query**:
   ```python
   collection.query(
       query_embeddings=[embed("birthday dinner Saket")],
       where={"$and": [{"sender": "Sneha"}, {"month_name": "may"}]}
   )
   ```

---

## 📜 License

MIT License. Designed for AI Search Engineering portfolios and open-source contributions.

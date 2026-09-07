# 💬 ChatGraph-Context | Semantic Group Chat Engine

**ChatGraph-Context** is an intelligent Python semantic search engine built to solve out-of-context text search failures in group chat exports (e.g. code-mixed Hinglish, typos, slang, missing conversational context, and member/temporal attribution).

---

## 📌 Key Features

1. **Crisp Light / White Theme & Color-Coded Members**:
   - Modern clean white dashboard interface where every participant (`Aarav`, `Priya`, `Rohan`, `Sneha`, `Vikram`, `Ananya`, `Kabir`, `Neha`) is assigned a distinct color avatar and bubble theme.
2. **Context-Aware Embeddings & Conversational Windows**:
   - Single chat messages are meaningless in isolation. **ChatGraph-Context** embeds preceding and succeeding context alongside every line, displaying full conversational thread context in the search UI.
3. **Multilingual Embedding Engine & Reranker**:
   - Uses `sentence-transformers` for dense vector representations of Latin-script Hindi mixed with English ("yaar plan fix hai", "budget tight hai").
4. **Metadata Intent Router**:
   - Parses free-form queries (e.g., `"from Sneha birthday dinner Saket in May"`) to automatically apply metadata filtering in **ChromaDB**.
5. **Synthetic Corpus Generator (`generate_corpus.py`)**:
   - 4,200+ realistic chat messages spanning 6 months with 5 decision threads (Manali Trip, Birthday Gift & Dinner, Flat Rent Agreement, Goa Cancellation, IPL Screening).

---

## 🏗️ Codebase Structure

```
.
├── chat_corpus.json          # Synthetic WhatsApp chat dataset (4,200 messages)
├── generate_corpus.py        # Dataset generator (Hinglish code-mixing, 5 decision threads)
├── search_engine.py          # Vector store (ChromaDB), intent router, context window retriever
├── evaluate.py               # Evaluation benchmark (40 test queries, Hit@K & MRR metrics)
├── app.py                    # ChatGraph-Context Streamlit White Theme Web Application
├── requirements.txt          # Python dependencies
└── README.md                 # Documentation
```

---

## 🚀 Quickstart & Setup

```bash
# 1. Clone Repository
git clone https://github.com/sulabh1-y/ChatGraph-Context.git
cd ChatGraph-Context

# 2. Install Dependencies
pip install -r requirements.txt

# 3. Generate Corpus & Vector Index
python generate_corpus.py
python search_engine.py

# 4. Launch Streamlit Web UI
streamlit run app.py
```

---

## 📊 Benchmark Evaluation Metrics

| Metric Category | Hit Rate @ 1 | Hit Rate @ 3 | Hit Rate @ 5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Temporal Queries** | **66.7%** | **91.7%** | **91.7%** | **0.791** |
| **Attributed Queries** | **83.3%** | **83.3%** | **83.3%** | **0.833** |
| **Overall Benchmark** | **52.5%** | **72.5%** | **72.5%** | **0.617** |

---

## 📜 License

MIT License. Designed for AI Search Engineering & NLP portfolios.

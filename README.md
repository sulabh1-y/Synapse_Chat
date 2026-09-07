# 🧠 Synapse_Chat | Intelligent WhatsApp Group Chat Semantic Search Engine

**Synapse_Chat** is a state-of-the-art Python semantic search engine designed to solve text-search failure modes in WhatsApp and group chat exports (e.g. code-mixed Hinglish, typos, slang, missing context, and sender/temporal filtering).

---

## 📌 Features & Key Capabilities

1. **Code-Mixed Hinglish Embedding Engine**:
   - Built on `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` and dense vector representations for Latin-script Hindi mixed with English ("yaar plan fix hai", "budget tight hai").
2. **Context-Aware Embeddings & Context Windows**:
   - Single messages out of context are meaningless. **Synapse_Chat** embeds conversational context (surrounding messages) alongside every line, and the UI displays full chat window threads (N messages before and after).
3. **Query Intent Router & Metadata Filtering**:
   - Automatically detects sender intent (e.g., `"from Sneha"`, `"sent by Rohan"`) and temporal intent (e.g., `"in July"`, `"during May 2026"`) to construct metadata filter clauses (`where` clauses) in **ChromaDB**.
4. **Synthetic Corpus Generator (`generate_corpus.py`)**:
   - Produces **4,200+ realistic messages** across **6 months** (March–August 2026) with **8 active participants** and 5 concrete decision-finalization threads (Manali Trip, Birthday Gift & Dinner, Flat Rent Agreement, Goa Cancellation, IPL Screening).
5. **Interactive Streamlit Web Dashboard (`app.py`)**:
   - **Tab 1: Semantic Search & Context Explorer** — Interactive search bar, preset query chips, hit scores, and contextual message bubbles.
   - **Tab 2: Corpus Analytics & Insights** — Participant message breakdown charts, monthly activity trends, and decision threads index.
   - **Tab 3: Live Message Simulator** — Type and dynamically index custom WhatsApp messages in real-time.
   - **Tab 4: Evaluation Benchmark Suite** — Run 40-query benchmark directly inside the browser.

---

## 🏗️ Codebase Structure

```
.
├── chat_corpus.json          # Generated synthetic chat dataset (4,200 messages)
├── generate_corpus.py        # Dataset generator (Hinglish code-mixing, 5 decision threads)
├── search_engine.py          # SynapseChatEngine (ChromaDB vector store, intent router, context retriever)
├── evaluate.py               # Evaluation benchmark (40 test queries, Hit@K & MRR metrics)
├── app.py                    # Synapse_Chat Streamlit Web Application
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## 📊 Benchmark Evaluation Metrics

| Metric Category | Hit Rate @ 1 | Hit Rate @ 3 | Hit Rate @ 5 | MRR |
| :--- | :---: | :---: | :---: | :---: |
| **Temporal Queries** | **66.7%** | **91.7%** | **91.7%** | **0.791** |
| **Attributed Queries** | **83.3%** | **83.3%** | **83.3%** | **0.833** |
| **Standard Semantic** | **50.0%** | **66.7%** | **66.7%** | **0.583** |
| **Overall Benchmark** | **52.5%** | **72.5%** | **72.5%** | **0.617** |

---

## 🚀 Quickstart Guide

### 1. Installation

```bash
git clone https://github.com/sulabh1-y/Synapse_Chat.git
cd Synapse_Chat
pip install -r requirements.txt
```

### 2. Generate Chat Corpus

```bash
python generate_corpus.py
```

### 3. Build Synapse_Chat Vector Store Index

```bash
python search_engine.py
```

### 4. Run Evaluation Benchmark

```bash
python evaluate.py
```

### 5. Launch Interactive Streamlit UI

```bash
streamlit run app.py
```

---

## 📜 License

MIT License. Designed for AI Search Engineering & Multi-Modal NLP portfolios.

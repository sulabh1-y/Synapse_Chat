import os
import json
import time
import pandas as pd
import streamlit as st
from datetime import datetime
from search_engine import SynapseChatEngine
from evaluate import evaluate_search_engine, get_test_suite

# Set page config
st.set_page_config(
    page_title="Synapse_Chat | Semantic Search Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Synapse_Chat UI theme
st.markdown("""
<style>
    /* Global Theme Overrides */
    .stApp {
        background-color: #090d12;
        color: #e2e8f0;
    }
    
    /* Header Styling */
    .main-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0px;
    }
    .sub-title {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-bottom: 20px;
    }
    
    /* Metrics Card Styling */
    .metric-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 16px;
        text-align: center;
        backdrop-filter: blur(10px);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 700;
        color: #38bdf8;
    }
    .metric-lbl {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* WhatsApp Chat Window Container */
    .chat-container {
        background-color: #0b141a;
        border-radius: 14px;
        padding: 22px;
        margin-bottom: 25px;
        border: 1px solid #1e293b;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 12px;
        margin-bottom: 16px;
        border-bottom: 1px solid #1e293b;
    }
    .intent-pill {
        background-color: rgba(56, 189, 248, 0.15);
        color: #38bdf8;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.82rem;
        font-weight: 600;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }
    .score-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }

    /* Message Bubbles */
    .chat-bubble {
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        position: relative;
        max-width: 88%;
        line-height: 1.45;
        font-size: 0.95rem;
    }
    .bubble-context {
        background-color: #111b21;
        color: #cbd5e1;
        border-left: 4px solid #38bdf8;
    }
    .bubble-target {
        background-color: #054640;
        color: #f8fafc;
        border: 2px solid #25d366;
        box-shadow: 0 0 15px rgba(37, 211, 102, 0.25);
    }
    
    .sender-name {
        font-weight: 700;
        font-size: 0.88rem;
        margin-bottom: 4px;
    }
    .sender-aarav { color: #f472b6; }
    .sender-priya { color: #38bdf8; }
    .sender-rohan { color: #4ade80; }
    .sender-sneha { color: #fbbf24; }
    .sender-vikram { color: #c084fc; }
    .sender-ananya { color: #fb923c; }
    .sender-kabir { color: #22d3ee; }
    .sender-neha { color: #f43f5e; }

    .msg-timestamp {
        font-size: 0.72rem;
        color: #94a3b8;
        float: right;
        margin-top: 4px;
        margin-left: 14px;
    }
    .target-tag {
        display: inline-block;
        background-color: #25d366;
        color: #064e3b;
        font-size: 0.72rem;
        font-weight: 800;
        padding: 2px 7px;
        border-radius: 4px;
        margin-left: 8px;
        letter-spacing: 0.5px;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_search_engine():
    engine = SynapseChatEngine()
    engine.index_corpus()
    return engine

search_engine = get_search_engine()

# --- MAIN HEADER ---
st.markdown('<div class="main-title">🧠 Synapse_Chat</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">High-Precision Multilingual Group Chat Semantic Search & Context Retrieval System</div>', unsafe_allow_html=True)

# Metrics Grid
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{len(search_engine.corpus):,}</div><div class="metric-lbl">Chat Corpus Messages</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><div class="metric-val">8</div><div class="metric-lbl">Active Participants</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="metric-val">6 Months</div><div class="metric-lbl">Time Span (Mar-Aug 2026)</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><div class="metric-val">ChromaDB</div><div class="metric-lbl">Multilingual Vector DB</div></div>', unsafe_allow_html=True)

st.write("")

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ Synapse_Chat Engine")
st.sidebar.caption("Model Architecture: Multilingual Embeddings + Context Windowing")

top_k = st.sidebar.slider("Top Results (K)", min_value=1, max_value=10, value=5)
window_size = st.sidebar.slider("Context Window Radius", min_value=1, max_value=7, value=3)

st.sidebar.divider()
st.sidebar.subheader("🎯 Metadata Filters Override")
senders_list = ["All Senders", "Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya", "Kabir", "Neha"]
selected_sender = st.sidebar.selectbox("Sender Attribution Filter", senders_list)

months_list = ["All Months", "March", "April", "May", "June", "July", "August"]
selected_month = st.sidebar.selectbox("Temporal Month Filter", months_list)

manual_sender_arg = None if selected_sender == "All Senders" else selected_sender
manual_month_arg = None if selected_month == "All Months" else selected_month.lower()

# --- INTERACTIVE UI TABS ---
tab_search, tab_analytics, tab_simulator, tab_eval = st.tabs([
    "🔍 Semantic Search & Thread Explorer",
    "📊 Corpus Analytics & Insights",
    "🧪 Live Message Simulator",
    "📈 Evaluation Benchmark Suite"
])

# ==========================================
# TAB 1: SEMANTIC SEARCH & THREAD EXPLORER
# ==========================================
with tab_search:
    st.subheader("🔍 Query Group Chat History")
    
    if "query_text" not in st.session_state:
        st.session_state["query_text"] = "from Sneha birthday dinner Saket"

    # Quick Preset Chips
    st.markdown("**Quick Preset Search Chips:**")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("🎯 Zero-Overlap Query", use_container_width=True):
            st.session_state["query_text"] = "Which restaurant location is reserved for evening food?"
    with p2:
        if st.button("👤 Sender Attributed Query", use_container_width=True):
            st.session_state["query_text"] = "from Sneha birthday dinner Saket"
    with p3:
        if st.button("📅 Temporal Month Query", use_container_width=True):
            st.session_state["query_text"] = "Manali holiday dates in July"
    with p4:
        if st.button("💬 Semantic Topic Query", use_container_width=True):
            st.session_state["query_text"] = "Sector 43 Gurgaon 3BHK flat rent"

    user_query = st.text_input("Enter free-form query:", value=st.session_state["query_text"], key="main_search_input")

    if user_query:
        start_t = time.time()
        hits = search_engine.search(
            user_query,
            top_k=top_k,
            manual_sender=manual_sender_arg,
            manual_month=manual_month_arg
        )
        search_latency = round((time.time() - start_t) * 1000, 1)
        
        if not hits:
            st.warning("No matching messages found for your query.")
        else:
            intent = hits[0]["intent"]
            intent_info = [f"⏱️ Search Latency: **{search_latency} ms**"]
            
            if intent["sender_filter"] or manual_sender_arg:
                intent_info.append(f"👤 Sender Filter: **{manual_sender_arg or intent['sender_filter']}**")
            if intent["month_filter"] or manual_month_arg:
                intent_info.append(f"📅 Temporal Filter: **{(manual_month_arg or intent['month_filter']).capitalize()}**")
            if intent["clean_query"] != intent["raw_query"]:
                intent_info.append(f"✨ Semantic Term: *'{intent['clean_query']}'*")

            st.info(" | ".join(intent_info))
            st.subheader(f"Top {len(hits)} Search Hits:")

            for idx, hit in enumerate(hits):
                msg_id = hit["message_id"]
                score_pct = round(hit["similarity_score"] * 100, 1)
                vec_pct = round(hit.get("vector_similarity", 0) * 100, 1)
                
                context_msgs = search_engine.get_context_window(msg_id, window_size=window_size)
                
                st.markdown(f"""
                <div class="chat-container">
                    <div class="chat-header">
                        <div>
                            <span class="intent-pill">Rank #{idx+1} | Message #{msg_id}</span>
                        </div>
                        <div>
                            <span class="score-badge">Relevance Score: {score_pct}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Render Thread Messages
                for ctx in context_msgs:
                    is_target = ctx["is_target"]
                    bubble_class = "bubble-target" if is_target else "bubble-context"
                    sender_class = f"sender-{ctx['sender'].lower()}"
                    target_badge = '<span class="target-tag">MATCHED HIT</span>' if is_target else ""
                    
                    dt = datetime.fromisoformat(ctx['timestamp'])
                    time_str = dt.strftime("%b %d, %Y - %I:%M %p")
                    
                    st.markdown(f"""
                    <div class="chat-bubble {bubble_class}">
                        <div class="sender-name {sender_class}">
                            {ctx['sender']} {target_badge}
                            <span class="msg-timestamp">{time_str}</span>
                        </div>
                        <div>{ctx['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Inspect JSON metadata expander
                with st.expander(f"⚙️ Inspect Raw Vector Metadata (Message #{msg_id})"):
                    st.json({
                        "message_id": msg_id,
                        "sender": hit["sender"],
                        "timestamp": hit["timestamp"],
                        "hybrid_score": hit["similarity_score"],
                        "vector_similarity": hit.get("vector_similarity"),
                        "keyword_score": hit.get("keyword_score"),
                        "parsed_intent": hit["intent"]
                    })
                    
                st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: CORPUS ANALYTICS & INSIGHTS
# ==========================================
with tab_analytics:
    st.subheader("📊 Chat Corpus Insights & Activity Analytics")
    
    if search_engine.corpus:
        df = pd.DataFrame(search_engine.corpus)
        df["dt"] = pd.to_datetime(df["timestamp"])
        df["month_name"] = df["dt"].dt.strftime("%B")
        
        ac1, ac2 = st.columns(2)
        
        with ac1:
            st.markdown("### 💬 Messages by Participant")
            sender_counts = df["sender"].value_counts().reset_index()
            sender_counts.columns = ["Sender", "Message Count"]
            st.bar_chart(sender_counts, x="Sender", y="Message Count", color="#38bdf8")
            
        with ac2:
            st.markdown("### 📅 Monthly Activity Distribution")
            month_counts = df["month_name"].value_counts().reset_index()
            month_counts.columns = ["Month", "Message Count"]
            st.bar_chart(month_counts, x="Month", y="Message Count", color="#4ade80")

        st.divider()
        st.markdown("### 📌 Decision Threads Index")
        st.dataframe(
            pd.DataFrame([
                {"Thread": "Manali Trip", "Target Dates": "June 12-16, 2026", "Hotel": "Snow Peak Resort", "Budget": "₹8,500/head"},
                {"Thread": "Priya's Birthday", "Date": "April 22, 2026", "Gift": "AirPods Pro 2nd Gen", "Dinner": "The Spice Route Saket"},
                {"Thread": "Flat Hunting", "Location": "Sector 43 Gurgaon", "Rent": "₹15,000/head", "Move-in": "May 1, 2026"},
                {"Thread": "IPL Screening", "Venue": "Aarav's Home", "Snacks": "₹350/head", "Setup": "Projector"},
                {"Thread": "Goa Trip", "Status": "Cancelled", "Reason": "Monsoon Flooding", "Refund": "Processed"}
            ]),
            use_container_width=True
        )

# ==========================================
# TAB 3: LIVE MESSAGE SIMULATOR & INDEXER
# ==========================================
with tab_simulator:
    st.subheader("🧪 Add Mock Message & Test Real-time Search")
    st.markdown("Insert a custom message into the **Synapse_Chat** corpus and immediately query it.")
    
    sim_col1, sim_col2 = st.columns(2)
    with sim_col1:
        sim_sender = st.selectbox("Sender", senders_list[1:], key="sim_sender")
        sim_text = st.text_input("Message Text", "yaar Ladakh trip ka budget 12000 decide hua hai", key="sim_text")
    with sim_col2:
        sim_date = st.date_input("Date", datetime.now())
        sim_time = st.time_input("Time", datetime.now().time())
        
    if st.button("🚀 Index Custom Message into Synapse_Chat"):
        new_id = len(search_engine.corpus) + 1
        new_ts = f"{sim_date}T{sim_time.strftime('%H:%M:%S')}"
        new_msg = {
            "message_id": new_id,
            "sender": sim_sender,
            "timestamp": new_ts,
            "text": sim_text
        }
        
        search_engine.corpus.append(new_msg)
        search_engine.corpus_by_id[new_id] = new_msg
        
        # Index single document into ChromaDB
        dt = datetime.fromisoformat(new_ts)
        doc_text = f"{sim_sender}: {sim_text}"
        emb = search_engine.model.encode([doc_text]).tolist()
        
        search_engine.collection.add(
            documents=[doc_text],
            embeddings=emb,
            metadatas=[{
                "message_id": new_id,
                "sender": sim_sender,
                "timestamp": new_ts,
                "year": dt.year,
                "month": dt.month,
                "month_name": dt.strftime("%B").lower(),
                "iso_date": dt.strftime("%Y-%m-%d")
            }],
            ids=[str(new_id)]
        )
        st.success(f"✅ Successfully indexed Message #{new_id}! Go to Tab 1 to search for it.")

# ==========================================
# TAB 4: EVALUATION BENCHMARK SUITE
# ==========================================
with tab_eval:
    st.subheader("📈 Synapse_Chat Evaluation Benchmark Suite")
    st.markdown("Run the 40-query benchmark to measure Hit Rate @ K and Mean Reciprocal Rank (MRR).")
    
    if st.button("⚡ Run Full Benchmark Test"):
        with st.spinner("Executing 40 evaluation queries..."):
            evaluate_search_engine(top_k=top_k)
            if os.path.exists("eval_results.json"):
                with open("eval_results.json", "r") as f:
                    res = json.load(f)
                    
                eb1, eb2, eb3, eb4 = st.columns(4)
                with eb1:
                    st.metric("Hit Rate @ 1", f"{res['hit_at_1']}%")
                with eb2:
                    st.metric("Hit Rate @ 3", f"{res['hit_at_3']}%")
                with eb3:
                    st.metric("Hit Rate @ 5", f"{res['hit_at_5']}%")
                with eb4:
                    st.metric("Mean Reciprocal Rank (MRR)", f"{res['mrr']}")
                    
                st.subheader("📌 Per-Category Performance Breakdown")
                cat_df = pd.DataFrame(res["category_stats"]).T.reset_index()
                cat_df.columns = ["Category", "Total", "Hit@1", "Hit@3", "Hit@5", "MRR Score"]
                st.dataframe(cat_df, use_container_width=True)

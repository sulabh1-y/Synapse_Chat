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
    page_title="ChatGraph-Context | Semantic Group Chat Engine",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Light/White Theme & Color-Coded Participants
st.markdown("""
<style>
    /* Global App White Theme */
    .stApp {
        background-color: #f8fafc;
        color: #0f172a;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }
    
    /* Title & Banner */
    .main-title {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #0284c7 0%, #2563eb 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .sub-title {
        color: #64748b;
        font-size: 1.05rem;
        margin-bottom: 24px;
        font-weight: 500;
    }
    
    /* Metrics Card Styling */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.03);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #0284c7;
    }
    .metric-lbl {
        font-size: 0.82rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.6px;
        font-weight: 600;
        margin-top: 4px;
    }

    /* Outer Chat Result Card */
    .chat-container {
        background-color: #ffffff;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 25px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.04);
    }
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 14px;
        margin-bottom: 18px;
        border-bottom: 1px solid #f1f5f9;
    }
    .intent-pill {
        background-color: #e0f2fe;
        color: #0369a1;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.84rem;
        font-weight: 700;
        border: 1px solid #bae6fd;
    }
    .score-badge {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: #ffffff;
        padding: 5px 14px;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 700;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.25);
    }

    /* Message Bubbles Base */
    .chat-bubble {
        padding: 12px 18px;
        border-radius: 14px;
        margin-bottom: 12px;
        position: relative;
        max-width: 88%;
        line-height: 1.5;
        font-size: 0.96rem;
        border: 1px solid transparent;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.03);
    }

    /* Member-Specific Color-Coded Bubble Styles */
    .bubble-aarav {
        background-color: #fdf2f8;
        border-color: #fbcfe8;
        color: #831843;
    }
    .bubble-priya {
        background-color: #f0f9ff;
        border-color: #bae6fd;
        color: #0c4a6e;
    }
    .bubble-rohan {
        background-color: #ecfdf5;
        border-color: #a7f3d0;
        color: #064e3b;
    }
    .bubble-sneha {
        background-color: #fffbeb;
        border-color: #fde68a;
        color: #78350f;
    }
    .bubble-vikram {
        background-color: #faf5ff;
        border-color: #e9d5ff;
        color: #581c87;
    }
    .bubble-ananya {
        background-color: #fff7ed;
        border-color: #ffedd5;
        color: #7c2d12;
    }
    .bubble-kabir {
        background-color: #ecfeff;
        border-color: #cffaff;
        color: #164e63;
    }
    .bubble-neha {
        background-color: #fff1f2;
        border-color: #fecdd3;
        color: #881337;
    }

    /* Target Search Match Highlight Overlay */
    .bubble-target {
        border: 2px solid #10b981 !important;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.3) !important;
    }

    /* Member Sender Names & Badges */
    .sender-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        font-weight: 800;
        font-size: 0.88rem;
        margin-bottom: 4px;
    }
    
    .sender-aarav { color: #db2777; }
    .sender-priya { color: #0284c7; }
    .sender-rohan { color: #059669; }
    .sender-sneha { color: #d97706; }
    .sender-vikram { color: #7e22ce; }
    .sender-ananya { color: #ea580c; }
    .sender-kabir { color: #0891b2; }
    .sender-neha { color: #e11d48; }

    .msg-timestamp {
        font-size: 0.74rem;
        color: #64748b;
        font-weight: 500;
    }
    .target-tag {
        background-color: #10b981;
        color: #ffffff;
        font-size: 0.7rem;
        font-weight: 800;
        padding: 2px 8px;
        border-radius: 6px;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        margin-left: 8px;
    }
    .avatar-badge {
        display: inline-block;
        width: 22px;
        height: 22px;
        border-radius: 50%;
        text-align: center;
        line-height: 22px;
        font-size: 0.72rem;
        color: #ffffff;
        font-weight: 800;
        margin-right: 6px;
    }
    .avatar-aarav { background-color: #db2777; }
    .avatar-priya { background-color: #0284c7; }
    .avatar-rohan { background-color: #059669; }
    .avatar-sneha { background-color: #d97706; }
    .avatar-vikram { background-color: #7e22ce; }
    .avatar-ananya { background-color: #ea580c; }
    .avatar-kabir { background-color: #0891b2; }
    .avatar-neha { background-color: #e11d48; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_search_engine():
    engine = SynapseChatEngine()
    engine.index_corpus()
    return engine

search_engine = get_search_engine()

# --- MAIN HEADER ---
st.markdown('<div class="main-title">💬 ChatGraph-Context</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Semantic Group Chat Search Engine with Participant Color-Coding & Context Windows</div>', unsafe_allow_html=True)

# Metrics Grid
m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-val">{len(search_engine.corpus):,}</div><div class="metric-lbl">Total Chat Messages</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown('<div class="metric-card"><div class="metric-val">8 Members</div><div class="metric-lbl">Color-Coded Participants</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown('<div class="metric-card"><div class="metric-val">6 Months</div><div class="metric-lbl">Time Horizon (2026)</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown('<div class="metric-card"><div class="metric-val">ChromaDB</div><div class="metric-lbl">Context Vector Store</div></div>', unsafe_allow_html=True)

st.write("")

# --- SIDEBAR CONTROLS ---
st.sidebar.title("⚙️ ChatGraph Engine")
st.sidebar.caption("Clean White Theme & Color-Coded Members")

top_k = st.sidebar.slider("Top Search Hits (K)", min_value=1, max_value=10, value=5)
window_size = st.sidebar.slider("Context Radius (Messages)", min_value=1, max_value=7, value=3)

st.sidebar.divider()
st.sidebar.subheader("🎯 Participant & Time Filters")
senders_list = ["All Senders", "Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya", "Kabir", "Neha"]
selected_sender = st.sidebar.selectbox("Filter by Member", senders_list)

months_list = ["All Months", "March", "April", "May", "June", "July", "August"]
selected_month = st.sidebar.selectbox("Filter by Month", months_list)

manual_sender_arg = None if selected_sender == "All Senders" else selected_sender
manual_month_arg = None if selected_month == "All Months" else selected_month.lower()

# --- INTERACTIVE UI TABS ---
tab_search, tab_members, tab_analytics, tab_eval = st.tabs([
    "🔍 Semantic Search & Context Window",
    "👥 Group Members Palette",
    "📊 Chat History Insights",
    "📈 Benchmark Suite"
])

# ==========================================
# TAB 1: SEMANTIC SEARCH & CONTEXT WINDOW
# ==========================================
with tab_search:
    st.subheader("🔍 Query Group Chat History")
    
    if "query_text" not in st.session_state:
        st.session_state["query_text"] = "from Sneha birthday dinner Saket"

    st.markdown("**Sample Search Presets:**")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("🎯 Zero-Overlap Query", use_container_width=True):
            st.session_state["query_text"] = "Which restaurant location is reserved for evening food?"
    with p2:
        if st.button("👤 Member Attributed", use_container_width=True):
            st.session_state["query_text"] = "from Sneha birthday dinner Saket"
    with p3:
        if st.button("📅 Temporal Month Query", use_container_width=True):
            st.session_state["query_text"] = "Manali holiday dates in July"
    with p4:
        if st.button("💬 Semantic Topic Query", use_container_width=True):
            st.session_state["query_text"] = "Sector 43 Gurgaon 3BHK flat rent"

    user_query = st.text_input("Type your free-form query:", value=st.session_state["query_text"], key="main_search_input")

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
            st.warning("No matching messages found.")
        else:
            intent = hits[0]["intent"]
            intent_info = [f"⚡ Search Latency: **{search_latency} ms**"]
            
            if intent["sender_filter"] or manual_sender_arg:
                intent_info.append(f"👤 Member: **{manual_sender_arg or intent['sender_filter']}**")
            if intent["month_filter"] or manual_month_arg:
                intent_info.append(f"📅 Month: **{(manual_month_arg or intent['month_filter']).capitalize()}**")
            if intent["clean_query"] != intent["raw_query"]:
                intent_info.append(f"✨ Search Terms: *'{intent['clean_query']}'*")

            st.info(" | ".join(intent_info))
            st.subheader(f"Search Results ({len(hits)} Matching Threads Found):")

            for idx, hit in enumerate(hits):
                msg_id = hit["message_id"]
                score_pct = round(hit["similarity_score"] * 100, 1)
                
                context_msgs = search_engine.get_context_window(msg_id, window_size=window_size)
                
                st.markdown(f"""
                <div class="chat-container">
                    <div class="chat-header">
                        <div>
                            <span class="intent-pill">Rank #{idx+1} | Message #{msg_id}</span>
                        </div>
                        <div>
                            <span class="score-badge">Match Relevance: {score_pct}%</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Render Thread Messages Color-Coded by Participant
                for ctx in context_msgs:
                    is_target = ctx["is_target"]
                    sender_clean = ctx['sender'].lower()
                    
                    bubble_class = f"bubble-{sender_clean}"
                    if is_target:
                        bubble_class += " bubble-target"
                        
                    sender_class = f"sender-{sender_clean}"
                    avatar_class = f"avatar-{sender_clean}"
                    avatar_initial = ctx['sender'][0].upper()
                    
                    target_badge = '<span class="target-tag">TARGET MATCH</span>' if is_target else ""
                    
                    dt = datetime.fromisoformat(ctx['timestamp'])
                    time_str = dt.strftime("%b %d, %Y • %I:%M %p")
                    
                    st.markdown(f"""
                    <div class="chat-bubble {bubble_class}">
                        <div class="sender-header {sender_class}">
                            <div>
                                <span class="avatar-badge {avatar_class}">{avatar_initial}</span>
                                {ctx['sender']} {target_badge}
                            </div>
                            <span class="msg-timestamp">{time_str}</span>
                        </div>
                        <div style="margin-top: 4px;">{ctx['text']}</div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with st.expander(f"⚙️ View Thread Vector Details (Message #{msg_id})"):
                    st.json({
                        "message_id": msg_id,
                        "sender": hit["sender"],
                        "timestamp": hit["timestamp"],
                        "hybrid_relevance_score": hit["similarity_score"],
                        "vector_similarity": hit.get("vector_similarity"),
                        "keyword_score": hit.get("keyword_score"),
                        "parsed_query_intent": hit["intent"]
                    })
                    
                st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# TAB 2: GROUP MEMBERS PALETTE
# ==========================================
with tab_members:
    st.subheader("👥 Group Participants & Color Palette")
    st.markdown("Each participant in **ChatGraph-Context** has a unique color identity for effortless visual tracking in conversation context windows.")
    
    members_data = [
        {"Name": "Aarav", "Role": "Trip Planner & Host", "Color Theme": "Pink (#DB2777)", "Badge": "A"},
        {"Name": "Priya", "Role": "Birthday Celebrant", "Color Theme": "Sky Blue (#0284C7)", "Badge": "P"},
        {"Name": "Rohan", "Role": "Treasurer & GPay Lead", "Color Theme": "Emerald Green (#059669)", "Badge": "R"},
        {"Name": "Sneha", "Role": "Booking Manager", "Color Theme": "Amber Gold (#D97706)", "Badge": "S"},
        {"Name": "Vikram", "Role": "Budget Negotiator", "Color Theme": "Purple (#7E22CE)", "Badge": "V"},
        {"Name": "Ananya", "Role": "Gift Coordinator", "Color Theme": "Orange (#EA580C)", "Badge": "A"},
        {"Name": "Kabir", "Role": "Flat Hunter", "Color Theme": "Cyan (#0891B2)", "Badge": "K"},
        {"Name": "Neha", "Role": "Event Host & Logistics", "Color Theme": "Rose (#E11D48)", "Badge": "N"}
    ]
    
    cols = st.columns(4)
    for idx, m in enumerate(members_data):
        with cols[idx % 4]:
            s_clean = m["Name"].lower()
            st.markdown(f"""
            <div class="chat-bubble bubble-{s_clean}" style="max-width: 100%; margin-bottom: 20px; border-radius: 12px;">
                <div class="sender-header sender-{s_clean}">
                    <div>
                        <span class="avatar-badge avatar-{s_clean}">{m['Badge']}</span>
                        {m['Name']}
                    </div>
                </div>
                <div style="font-size: 0.85rem; font-weight: 600; margin-top: 4px;">{m['Role']}</div>
                <div style="font-size: 0.75rem; opacity: 0.8; margin-top: 2px;">{m['Color Theme']}</div>
            </div>
            """, unsafe_allow_html=True)

# ==========================================
# TAB 3: CHAT HISTORY INSIGHTS
# ==========================================
with tab_analytics:
    st.subheader("📊 Chat History & Member Activity Insights")
    
    if search_engine.corpus:
        df = pd.DataFrame(search_engine.corpus)
        df["dt"] = pd.to_datetime(df["timestamp"])
        df["month_name"] = df["dt"].dt.strftime("%B")
        
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("### 💬 Messages by Member")
            sender_counts = df["sender"].value_counts().reset_index()
            sender_counts.columns = ["Member", "Messages"]
            st.bar_chart(sender_counts, x="Member", y="Messages", color="#0284c7")
            
        with c2:
            st.markdown("### 📅 Monthly Chat Volume")
            month_counts = df["month_name"].value_counts().reset_index()
            month_counts.columns = ["Month", "Messages"]
            st.bar_chart(month_counts, x="Month", y="Messages", color="#10b981")

# ==========================================
# TAB 4: BENCHMARK SUITE
# ==========================================
with tab_eval:
    st.subheader("📈 ChatGraph-Context Benchmark Evaluation")
    st.markdown("Run the 40-query test suite to measure Hit Rate @ K and Mean Reciprocal Rank (MRR).")
    
    if st.button("⚡ Run Full Benchmark Test"):
        with st.spinner("Evaluating 40 test queries..."):
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
                cat_df.columns = ["Category", "Total Queries", "Hit@1", "Hit@3", "Hit@5", "MRR Score"]
                st.dataframe(cat_df, use_container_width=True)

import json
import streamlit as st
from datetime import datetime
from search_engine import HinglishSearchEngine
from evaluate import evaluate_search_engine, get_test_suite

# Set page config
st.set_page_config(
    page_title="WhatsApp Semantic Search Engine",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for WhatsApp Chat UI styling
st.markdown("""
<style>
    .chat-container {
        background-color: #0b141a;
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 25px;
        border: 1px solid #222d34;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .chat-header {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding-bottom: 12px;
        margin-bottom: 15px;
        border-bottom: 1px solid #222d34;
    }
    .intent-badge {
        background-color: #1f2c34;
        color: #00a884;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .score-badge {
        background-color: #00a884;
        color: #ffffff;
        padding: 4px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 700;
    }
    .chat-bubble {
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 8px;
        position: relative;
        max-width: 85%;
        line-height: 1.4;
    }
    .bubble-context {
        background-color: #111b21;
        color: #d1d7db;
        border-left: 3px solid #3b82f6;
    }
    .bubble-target {
        background-color: #005c4b;
        color: #e9edef;
        border: 2px solid #25d366;
        box-shadow: 0 0 10px rgba(37, 211, 102, 0.3);
    }
    .sender-name {
        font-weight: bold;
        font-size: 0.88rem;
        margin-bottom: 2px;
    }
    .sender-aarav { color: #e542a3; }
    .sender-priya { color: #53bdeb; }
    .sender-rohan { color: #25d366; }
    .sender-sneha { color: #ffbc38; }
    .sender-vikram { color: #a855f7; }
    .sender-ananya { color: #f97316; }
    .sender-kabir { color: #06b6d4; }
    .sender-neha { color: #ec4899; }

    .msg-timestamp {
        font-size: 0.72rem;
        color: #8696a0;
        float: right;
        margin-top: 4px;
        margin-left: 12px;
    }
    .target-tag {
        display: inline-block;
        background-color: #25d366;
        color: #111b21;
        font-size: 0.75rem;
        font-weight: 800;
        padding: 2px 6px;
        border-radius: 4px;
        margin-left: 8px;
        text-transform: uppercase;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def get_search_engine():
    engine = HinglishSearchEngine()
    engine.index_corpus()
    return engine

search_engine = get_search_engine()

# --- HEADER SECTION ---
st.title("💬 WhatsApp Group Chat Semantic Search Engine")
st.markdown("*Solve text-search failures in Hinglish group chats using Multilingual Vector Embeddings, Intent Routing, and Context Windows.*")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Messages", f"{len(search_engine.corpus):,}")
with col2:
    st.metric("Active Participants", "8 Senders")
with col3:
    st.metric("Timeframe", "6 Months (2026)")
with col4:
    st.metric("Vector Index", "ChromaDB Active")

st.divider()

# --- SIDEBAR CONTROLS ---
st.sidebar.header("⚙️ Search Controls")
top_k = st.sidebar.slider("Top Results (K)", min_value=1, max_value=10, value=5)
window_size = st.sidebar.slider("Context Window Size (Radius)", min_value=1, max_value=7, value=3)

st.sidebar.divider()
st.sidebar.header("🎯 Manual Filter Overrides")
senders_list = ["All Senders", "Aarav", "Priya", "Rohan", "Sneha", "Vikram", "Ananya", "Kabir", "Neha"]
selected_sender = st.sidebar.selectbox("Sender Filter", senders_list)

months_list = ["All Months", "March", "April", "May", "June", "July", "August"]
selected_month = st.sidebar.selectbox("Month Filter", months_list)

manual_sender_arg = None if selected_sender == "All Senders" else selected_sender
manual_month_arg = None if selected_month == "All Months" else selected_month.lower()

st.sidebar.divider()
st.sidebar.header("📊 Evaluation Benchmark")
if st.sidebar.button("🚀 Run 40-Query Benchmark"):
    with st.spinner("Running evaluation benchmark..."):
        evaluate_search_engine(top_k=top_k)
        if os.path.exists("eval_results.json"):
            with open("eval_results.json", "r") as f:
                res = json.load(f)
            st.sidebar.success(f"**Hit@1:** {res['hit_at_1']}% | **Hit@3:** {res['hit_at_3']}%")
            st.sidebar.info(f"**Hit@5:** {res['hit_at_5']}% | **MRR:** {res['mrr']}")

# --- SEARCH INPUT & PRESETS ---
st.subheader("🔍 Query Chat Corpus")

# Preset Query Buttons
preset_cols = st.columns(4)
query_input = ""

if "query_text" not in st.session_state:
    st.session_state["query_text"] = "from Sneha birthday dinner Saket"

with preset_cols[0]:
    if st.button("🎯 Zero-Overlap"):
        st.session_state["query_text"] = "Which restaurant location is reserved for evening food?"
with preset_cols[1]:
    if st.button("👤 Attributed"):
        st.session_state["query_text"] = "from Sneha birthday dinner Saket"
with preset_cols[2]:
    if st.button("📅 Temporal"):
        st.session_state["query_text"] = "Manali holiday dates in July"
with preset_cols[3]:
    if st.button("💬 Semantic"):
        st.session_state["query_text"] = "Sector 43 Gurgaon 3BHK flat rent"

user_query = st.text_input("Enter free-form query:", value=st.session_state["query_text"])

if user_query:
    hits = search_engine.search(
        user_query,
        top_k=top_k,
        manual_sender=manual_sender_arg,
        manual_month=manual_month_arg
    )
    
    if not hits:
        st.warning("No matching messages found for your query.")
    else:
        # Display Intent Parsing metadata
        intent = hits[0]["intent"]
        intent_info = []
        if intent["sender_filter"] or manual_sender_arg:
            intent_info.append(f"👤 Sender: **{manual_sender_arg or intent['sender_filter']}**")
        if intent["month_filter"] or manual_month_arg:
            intent_info.append(f"📅 Month: **{(manual_month_arg or intent['month_filter']).capitalize()}**")
        if intent["clean_query"] != intent["raw_query"]:
            intent_info.append(f"✨ Semantic Term: *'{intent['clean_query']}'*")

        if intent_info:
            st.info(" | ".join(intent_info))

        st.subheader(f"Results ({len(hits)} Matches Found):")

        for idx, hit in enumerate(hits):
            msg_id = hit["message_id"]
            score_pct = round(hit["similarity_score"] * 100, 1)
            
            # Fetch surrounding context window
            context_msgs = search_engine.get_context_window(msg_id, window_size=window_size)
            
            st.markdown(f"""
            <div class="chat-container">
                <div class="chat-header">
                    <div>
                        <span class="intent-badge">Result #{idx+1} (Message #{msg_id})</span>
                    </div>
                    <span class="score-badge">Relevance Score: {score_pct}%</span>
                </div>
            """, unsafe_allow_html=True)
            
            # Render context window thread messages
            for ctx in context_msgs:
                is_target = ctx["is_target"]
                bubble_class = "bubble-target" if is_target else "bubble-context"
                sender_class = f"sender-{ctx['sender'].lower()}"
                
                target_badge = '<span class="target-tag">TARGET MATCH</span>' if is_target else ""
                
                # Format timestamp nicely
                dt = datetime.fromisoformat(ctx['timestamp'])
                time_str = dt.strftime("%b %d, %I:%M %p")
                
                st.markdown(f"""
                <div class="chat-bubble {bubble_class}">
                    <div class="sender-name {sender_class}">
                        {ctx['sender']} {target_badge}
                        <span class="msg-timestamp">{time_str}</span>
                    </div>
                    <div>{ctx['text']}</div>
                </div>
                """, unsafe_allow_html=True)
                
            st.markdown("</div>", unsafe_allow_html=True)

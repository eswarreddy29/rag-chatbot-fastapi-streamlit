import os
import time
from datetime import datetime
import requests
import streamlit as st

# -----------------------------------------------------------------------------
# Configuration & Backend Endpoint
# -----------------------------------------------------------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="DocuMind AI | Intelligent Document RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# Custom CSS for Modern, Premium UI & Glassmorphism Aesthetics
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700&family=Inter:wght@300;400;500;600&family=Fira+Code:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        letter-spacing: -0.02em;
    }

    /* Main Container Padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1100px;
    }

    /* Hero Header Styling */
    .hero-container {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.12) 0%, rgba(168, 85, 247, 0.12) 50%, rgba(236, 72, 153, 0.08) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 24px 30px;
        margin-bottom: 25px;
        backdrop-filter: blur(12px);
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.1);
    }

    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(120deg, #6366F1, #A855F7, #EC4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
    }

    .hero-subtitle {
        font-size: 1rem;
        color: #94A3B8;
        font-weight: 400;
        margin-bottom: 14px;
    }

    /* Badges */
    .badge-container {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        margin-top: 8px;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        font-size: 0.78rem;
        font-weight: 500;
        padding: 4px 12px;
        border-radius: 20px;
        background: rgba(99, 102, 241, 0.15);
        color: #818CF8;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }

    .badge-status-online {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }

    .badge-status-offline {
        background: rgba(239, 68, 68, 0.15);
        color: #F87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }

    /* Sidebar Customization */
    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .sidebar-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 16px;
        margin-bottom: 16px;
    }

    .sidebar-title {
        font-size: 1rem;
        font-weight: 600;
        color: #E2E8F0;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Empty state styling */
    .empty-state-card {
        border: 1px dashed rgba(148, 163, 184, 0.25);
        border-radius: 16px;
        padding: 36px 20px;
        text-align: center;
        background: rgba(255, 255, 255, 0.015);
        margin: 20px 0;
    }

    .empty-state-icon {
        font-size: 2.8rem;
        margin-bottom: 10px;
        display: inline-block;
    }

    /* Button Enhancements */
    div.stButton > button {
        border-radius: 10px;
        font-weight: 500;
        transition: all 0.25s ease;
    }

    div.stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(99, 102, 241, 0.25);
    }

    /* Code & Metrics */
    code {
        font-family: 'Fira Code', monospace;
    }

    /* Timestamp tag in messages */
    .msg-timestamp {
        font-size: 0.72rem;
        color: #64748B;
        margin-top: 4px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Session State Initialization
# -----------------------------------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_docs" not in st.session_state:
    st.session_state.indexed_docs = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

if "quick_prompt" not in st.session_state:
    st.session_state.quick_prompt = None

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------
def check_backend_health():
    """Check connectivity to the FastAPI backend."""
    try:
        response = requests.get(f"{BACKEND_URL}/docs", timeout=2)
        return response.status_code == 200
    except Exception:
        return False

# -----------------------------------------------------------------------------
# Sidebar: Document Management & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="display: flex; align-items: center; gap: 10px; margin-bottom: 12px;">
            <span style="font-size: 1.8rem;">⚡</span>
            <div>
                <h3 style="margin: 0; font-size: 1.25rem; font-weight: 700; color: #F8FAFC;">DocuMind</h3>
                <span style="font-size: 0.8rem; color: #94A3B8;">RAG Knowledge Assistant</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Server Status Card
    is_online = check_backend_health()
    status_class = "badge-status-online" if is_online else "badge-status-offline"
    status_icon = "🟢" if is_online else "🔴"
    status_text = "Backend Online" if is_online else "Backend Offline"

    st.markdown(
        f"""
        <div style="margin-bottom: 18px;">
            <span class="badge {status_class}">
                {status_icon} {status_text}
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )

    # Document Upload Section
    st.markdown("#### 📄 Document Ingestion")
    uploaded_file = st.file_uploader(
        "Upload reference PDF",
        type=["pdf"],
        help="Upload a PDF document to vectorize and add into your knowledge base."
    )

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if file_id not in st.session_state.processed_files:
            files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
            try:
                res = requests.post(f"{BACKEND_URL}/upload", files=files, timeout=60)
                if res.status_code == 200:
                    st.session_state.processed_files.add(file_id)
                    if uploaded_file.name not in st.session_state.indexed_docs:
                        st.session_state.indexed_docs.append(uploaded_file.name)
                        st.rerun()
                else:
                    detail = res.json().get("detail", "Error occurred during upload.")
                    st.error(f"❌ Upload Failed: {detail}")
            except Exception as e:
                st.error(f"❌ Could not connect to backend: {e}")

    # Indexed Documents Pill List
    if st.session_state.indexed_docs:
        st.markdown("###### 📚 Active Knowledge Sources")
        for doc in st.session_state.indexed_docs:
            st.markdown(f"- 📄 `{doc}`")

    st.divider()

    # Session & Database Controls
    st.markdown("#### ⚙️ Workspace Actions")

    col_btn1, col_btn2 = st.columns(2)

    with col_btn1:
        if st.button("🧹 Clear Chat", use_container_width=True, help="Clear conversation from UI"):
            st.session_state.messages = []
            st.rerun()

    with col_btn2:
        if st.button("🗑️ Reset All", use_container_width=True, help="Wipe Qdrant collection, documents & chat"):
            try:
                with st.spinner("Resetting vector database..."):
                    res = requests.post(f"{BACKEND_URL}/clear", timeout=30)
                    if res.status_code == 200:
                        st.session_state.messages = []
                        st.session_state.indexed_docs = []
                        st.session_state.processed_files = set()
                        st.toast("Database & conversation wiped successfully!", icon="✨")
                        st.rerun()
                    else:
                        st.error("Failed to clear backend database.")
            except Exception as e:
                st.error(f"Connection error: {e}")

    # Export Chat Feature
    if st.session_state.messages:
        chat_text = "# Conversation History\n\n"
        for m in st.session_state.messages:
            role_title = "User" if m["role"] == "user" else "Assistant"
            chat_text += f"### {role_title} ({m.get('time', '')})\n{m['content']}\n\n"

        st.download_button(
            label="💾 Export Conversation (.md)",
            data=chat_text,
            file_name=f"chat_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
            mime="text/markdown",
            use_container_width=True
        )

    st.divider()
    st.caption("🚀 Powered by **FastAPI**, **Qdrant Vector DB**, **FastEmbed** & **Groq LLM**")

# -----------------------------------------------------------------------------
# Main Chat Header / Hero Section
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">Intelligent Document Assistant</div>
        <div class="hero-subtitle">
            Upload PDFs in the sidebar and chat intelligently with deep semantic retrieval powered by RAG.
        </div>
        <div class="badge-container">
            <span class="badge">🤖 Groq LLM</span>
            <span class="badge">🔍 Qdrant Cloud Vector Store</span>
            <span class="badge">⚡ FastEmbed BGE-Small</span>
            <span class="badge">⚡ LangChain Classic Chains</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------------------------------------------------------
# Empty State & Suggested Starters
# -----------------------------------------------------------------------------
if not st.session_state.messages:
    st.markdown(
        """
        <div class="empty-state-card">
            <div class="empty-state-icon">💬</div>
            <h3 style="margin-bottom: 6px; color: #F1F5F9;">No conversation yet</h3>
            <p style="color: #94A3B8; font-size: 0.92rem; max-width: 500px; margin: 0 auto 16px auto;">
                Upload a document on the left sidebar to index your knowledge base, or pick a starter question below.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("###### 💡 Quick Starter Questions")
    quick_cols = st.columns(3)
    
    starters = [
        "📄 Summarize the main points of the uploaded document.",
        "🔍 What are the key findings or conclusions?",
        "💡 Explain the core concepts in simple terms."
    ]

    for idx, starter in enumerate(starters):
        with quick_cols[idx]:
            if st.button(starter, key=f"quick_starter_{idx}", use_container_width=True):
                st.session_state.quick_prompt = starter
                st.rerun()

# -----------------------------------------------------------------------------
# Display Existing Messages
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar_icon = "👤" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar_icon):
        st.markdown(msg["content"])
        if "time" in msg:
            st.markdown(f"<div class='msg-timestamp'>{msg['time']}</div>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Process Chat Input (From input box or quick prompt buttons)
# -----------------------------------------------------------------------------
input_query = st.chat_input("Ask anything about your documents...")

# Check if a quick prompt was clicked
if st.session_state.quick_prompt and not input_query:
    input_query = st.session_state.quick_prompt
    st.session_state.quick_prompt = None

if input_query:
    current_time = datetime.now().strftime("%I:%M %p")

    # Render User Message
    with st.chat_message("user", avatar="👤"):
        st.markdown(input_query)
        st.markdown(f"<div class='msg-timestamp'>{current_time}</div>", unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "user",
        "content": input_query,
        "time": current_time
    })

    # Query Backend RAG API
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("🧠 Searching knowledge embeddings & generating response..."):
            try:
                start_time = time.time()
                response = requests.post(
                    f"{BACKEND_URL}/query",
                    json={"question": input_query},
                    timeout=60
                )
                elapsed_time = round(time.time() - start_time, 2)

                if response.status_code == 200:
                    answer = response.json().get("answer", "No answer found in context.")
                    st.markdown(answer)
                    resp_time = datetime.now().strftime("%I:%M %p")
                    st.markdown(
                        f"<div class='msg-timestamp'>{resp_time} • Generated in {elapsed_time}s</div>",
                        unsafe_allow_html=True
                    )
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "time": f"{resp_time} • {elapsed_time}s"
                    })
                else:
                    detail = response.json().get("detail", "Unknown server error.")
                    err_msg = f"⚠️ Server Error ({response.status_code}): {detail}"
                    st.error(err_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": err_msg,
                        "time": datetime.now().strftime("%I:%M %p")
                    })
            except requests.exceptions.ConnectionError:
                err_msg = f"🔌 **Connection Error**: Unable to connect to the FastAPI backend. Please ensure the backend server is running on `{BACKEND_URL}`."
                st.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                    "time": datetime.now().strftime("%I:%M %p")
                })
            except Exception as e:
                err_msg = f"❌ **Unexpected Error**: {str(e)}"
                st.error(err_msg)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": err_msg,
                    "time": datetime.now().strftime("%I:%M %p")
                })
import os
import requests
import streamlit as st

# Dynamically pull backend host URL (Set BACKEND_URL in Render Environment settings)
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

st.set_page_config(
    page_title="Enterprise Document RAG",
    page_icon="🤖",
    layout="wide"
)

# Hide top header, toolbar, GitHub icon, and footer
hide_streamlit_elements = """
    <style>
    header {visibility: hidden;}
    [data-testid="stHeader"] {display: none;}
    [data-testid="stToolbar"] {display: none;}
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
"""
st.markdown(hide_streamlit_elements, unsafe_allow_html=True)

# Sidebar UI for File Uploading & Controls
with st.sidebar:
    st.title("📁 Document Control Center")
    
    # Updated Clear Chat Button to wipe the backend database too
    if st.button("🗑️ Clear Chat & Knowledge Base", use_container_width=True):
        try:
            # Tell the backend to wipe Qdrant Cloud collection and temporary local documents
            response = requests.post(f"{BACKEND_URL}/clear")
            if response.status_code == 200:
                st.session_state.messages = []
                st.success("Database and chat cleared!")
                st.rerun()
            else:
                st.error("Failed to clear backend database.")
        except Exception as e:
            st.error(f"Connection error: {e}")
        
    # Clear Chat Button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
        
    st.markdown("---")
    
    st.subheader("Upload context files")

    uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
    
    if uploaded_file is not None:
        if st.button("Index Document", use_container_width=True):
            with st.spinner("Processing document..."):
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                try:
                    response = requests.post(f"{BACKEND_URL}/upload", files=files)
                    if response.status_code == 200:
                        st.success(response.json()["message"])
                    else:
                        st.error(f"Error: {response.json()['detail']}")
                except Exception as e:
                    st.error(f"Could not connect to backend: {e}")
                    
    st.markdown("---")
    st.caption("Powered by FastAPI, Qdrant Cloud & LangChain")

# Main Interface Header
st.title("🤖 Intelligent Document Chatbot")
st.markdown("Ask natural language questions regarding your uploaded document context.")

# Initialize session state for conversation memory tracking
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display persistent conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user inputs
if user_query := st.chat_input("What would you like to ask?"):
    # Display human message
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Fetch response from backend api
    with st.chat_message("assistant"):
        with st.spinner("Analyzing knowledge base..."):
            try:
                response = requests.post(
                    f"{BACKEND_URL}/query", 
                    json={"question": user_query}
                )
                if response.status_code == 200:
                    answer = response.json()["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                else:
                    st.error("Failed to retrieve a valid response from the core backend service.")
            except Exception as e:
                st.error(f"Connection error: {e}")
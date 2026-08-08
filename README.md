# 🚀 Intelligent Document RAG Chatbot

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-1C3C3C?logo=langchain&logoColor=white)
![ChromaDB](https://img.shields.io/badge/ChromaDB-FF6F00?logo=chroma&logoColor=white)

An end-to-end Retrieval-Augmented Generation (RAG) application built to transform static PDF documents into dynamic, interactive conversational experiences. 

This project seamlessly bridges a blazingly fast **FastAPI** backend with a sleek **Streamlit** frontend. It leverages state-of-the-art open-source LLMs via the **Groq API** and efficient local text embeddings to provide high-quality, context-aware answers without breaking the bank.

## ✨ Key Features

*   **📄 Seamless PDF Ingestion:** Upload any PDF, and the engine automatically extracts, chunks, and indexes the content.
*   **🧠 Local CPU-Optimized Embeddings:** Uses `BAAI/bge-small-en-v1.5` from HuggingFace—a zero-budget, highly efficient embedding model that runs perfectly on CPUs.
*   **⚡ Lightning Fast Inference:** Powered by **Llama-3.1-8b-instant** via Groq's API for ultra-low latency generation.
*   **🗄️ Persistent Vector Storage:** Utilizes ChromaDB to store document embeddings, ensuring your knowledge base persists across sessions.
*   **🧹 Smart State Management:** Built-in endpoints and UI controls to wipe the knowledge base, clear chat history, and delete physical files in one click.

---

## 🏗️ Architecture & How It Works

This application follows a classic RAG architecture, separated into a robust backend and an intuitive frontend:

1.  **Document Processing (`PyPDFLoader`):** Extracts raw text from uploaded PDFs.
2.  **Smart Chunking (`RecursiveCharacterTextSplitter`):** Splits text into manageable 1000-character chunks with a 200-character overlap to preserve semantic context.
3.  **Vectorization (`HuggingFaceEmbeddings`):** Converts text chunks into dense vector representations.
4.  **Storage & Retrieval (`Chroma`):** Stores vectors locally. Upon a user query, it retrieves the top 3 most semantically relevant chunks ($k=3$).
5.  **Generation (`ChatGroq`):** Combines the user query with the retrieved context and a customized system prompt, passing it to the Llama-3.1 model to generate a natural, conversational response.

---

## 🛠️ Tech Stack

### Backend
*   **Framework:** FastAPI, Uvicorn (ASGI server)
*   **AI/Orchestration:** LangChain
*   **Vector Database:** ChromaDB
*   **Embeddings:** HuggingFace (`BAAI/bge-small-en-v1.5`)
*   **LLM:** Groq API (`llama-3.1-8b-instant`)

### Frontend
*   **Framework:** Streamlit
*   **Integration:** `requests` for seamless REST API communication

---

## 🚀 Getting Started

Follow these steps to get the project up and running on your local machine.

### 1. Prerequisites
*   Python 3.8+
*   A free API key from [Groq](https://console.groq.com/)

### 2. Clone the Repository
```bash
git clone [https://github.com/eswarreddy29/rag-chatbot-fastapi-streamlit.git](https://github.com/eswarreddy29/rag-chatbot-fastapi-streamlit.git)
cd rag-chatbot-fastapi-streamlit

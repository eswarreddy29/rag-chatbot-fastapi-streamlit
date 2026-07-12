import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

app = FastAPI(title="LLM RAG Engine Backend")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "backend/documents"
DB_DIR = "backend/chroma_db"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize local embeddings (Zero Budget - Runs free on CPU)
embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en-v1.5")

# Initialize Vector Store
vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)

# Initialize Open-Source LLM via Groq Free Tier
llm = ChatGroq(
    temperature=0.4, 
    model_name="llama-3.1-8b-instant", 
    groq_api_key=os.getenv("GROQ_API_KEY")
)

class QueryRequest(BaseModel):
    question: str

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        f.write(await file.read())
        
    try:
        # 1. Load document
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        
        # 2. Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        final_documents = text_splitter.split_documents(docs)
        
        # 3. Embed and store chunks into ChromaDB
        vector_store.add_documents(final_documents)
        
        return {"message": f"Successfully processed and indexed {file.filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query")
async def query_rag(request: QueryRequest):
    try:
        # Configure retriever to fetch top 3 matching contexts
        retriever = vector_store.as_retriever(search_kwargs={"k": 3})
        
        # Define strict contextual system prompt
        system_prompt = (
            "You are a highly engaging, brilliant AI assistant helping a user with their documents.\n"
            "Below is information extracted from their uploaded files.\n\n"
            "Context:\n{context}\n\n"
            "Your instructions:\n"
            "1. Answer the user's question primarily using the context provided above.\n"
            "2. You do not need to be strictly confined to the document. Feel free to use your own "
            "general knowledge to elaborate, provide examples, or make the answer more comprehensive, "
            "but clearly mention when you are adding outside information not found in the text.\n"
            "3. Keep your tone energetic, conversational, and incredibly helpful, just like a friendly AI partner!"
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])
        
        # Create RAG Chain
        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)
        
        response = rag_chain.invoke({"input": request.question})
        return {"answer": response["answer"]}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/clear")
async def clear_database():
    try:
        global vector_store
        # Delete the existing collection data from ChromaDB
        vector_store.delete_collection()
        # Re-initialize a clean, empty vector store
        vector_store = Chroma(persist_directory=DB_DIR, embedding_function=embeddings)
        
        # Clear out physical files in the upload directory
        for filename in os.listdir(UPLOAD_DIR):
            file_path = os.path.join(UPLOAD_DIR, filename)
            if os.path.isfile(file_path):
                os.remove(file_path)
                
        return {"message": "Knowledge base and chat history cleared successfully!"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
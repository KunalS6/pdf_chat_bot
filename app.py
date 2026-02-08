# =========================
# Streamlit UI
# =========================
import streamlit as st

st.set_page_config(page_title="PDF Chatbot (RAG)", layout="wide")
st.title("📄 Conversational PDF Chatbot")
st.caption("Upload a PDF and ask questions with memory (answers ONLY from the PDF)")

# =========================
# Standard imports
# =========================
import os
import uuid
import re
import csv
import io
import atexit

# =========================
# Environment setup
# =========================
os.environ["ANONYMIZED_TELEMETRY"] = "False"

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in environment")
    st.stop()

# Embedding provider key (OpenAI-style API)
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    st.error("❌ OPENAI_API_KEY not found in environment (needed for embeddings)")
    st.stop()

# =========================
# LLM (Groq)
# =========================
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    groq_api_key=groq_api_key,
    temperature=0.2,
)

# =========================
# Embeddings (hosted, no local libs)
# =========================
from langchain_community.embeddings import OpenAIEmbeddings

@st.cache_resource(show_spinner=False)
def load_embedding():
    # Change model name if you use a different OpenAI-compatible embedding model
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=openai_api_key,
    )

embedding = load_embedding()

# =========================
# Chroma (persistent)
# =========================
from chromadb import PersistentClient
from langchain_chroma import Chroma

@st.cache_resource(show_spinner=False)
def chroma_client():
    os.makedirs("chroma_store", exist_ok=True)
    return PersistentClient(path="chroma_store")

chroma = chroma_client()

def sanitize(name: str) -> str:
    name = os.path.splitext(name)[0].lower()
    name = re.sub(r"[^a-z0-9_-]+", "_", name)
    return name[:63]

# =========================
# Loaders & splitters
# =========================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# Prompts
# =========================
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Reformulate the user question into a standalone question "
     "using chat history. Do NOT answer."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer ONLY using the CONTEXT below.\n\n{context}\n\n"
     "Rules:\n"
     "- Use only the provided context\n"
     "- If the answer is not in context, say you don't know based on the uploaded PDF"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# =========================
# LangChain chains
# =========================
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain

# =========================
# Chat memory
# =========================
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# =========================
# Streamlit session state
# =========================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "collection_name" not in st.session_state:
    st.session_state.collection_name = None

if "pdf_path" not in st.ses

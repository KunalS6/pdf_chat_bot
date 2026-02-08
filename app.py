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
#from dotenv import load_dotenv

# =========================
# Environment setup
# =========================
os.environ["ANONYMIZED_TELEMETRY"] = "False"
#load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found in environment")
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
# Embeddings (Streamlit Cloud SAFE)
# =========================
from langchain_huggingface import HuggingFaceEmbeddings

@st.cache_resource(show_spinner=False)
def load_embedding():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
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
     "- If the answer is not in context, say you don't know"),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# =========================
# Chains
# =========================

from langchain.chains.history_aware_retriever import (
    create_history_aware_retriever
)

from langchain.chains.retrieval import (
    create_retrieval_chain
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

if "pdf_path" not in st.session_state:
    st.session_state.pdf_path = None

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# =========================
# Auto cleanup on app close
# =========================
def cleanup():
    if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
        try:
            os.remove(st.session_state.pdf_path)
        except:
            pass

    if st.session_state.collection_name:
        try:
            chroma.delete_collection(st.session_state.collection_name)
        except:
            pass

atexit.register(cleanup)

# =========================
# Sidebar
# =========================
with st.sidebar:
    st.subheader("Session controls")

    # 🔽 Download chat CSV
    if st.session_state.chat_log:
        buffer = io.StringIO()
        writer = csv.DictWriter(buffer, fieldnames=["question", "answer"])
        writer.writeheader()
        writer.writerows(st.session_state.chat_log)

        st.download_button(
            "⬇️ Download chat (CSV)",
            buffer.getvalue(),
            "chat_history.csv",
            "text/csv",
        )

    # 🗑️ Delete current PDF
    if st.button("🗑️ Delete current PDF"):
        store.pop(st.session_state.session_id, None)
        st.session_state.chat_log.clear()

        if st.session_state.collection_name:
            chroma.delete_collection(st.session_state.collection_name)

        if st.session_state.pdf_path and os.path.exists(st.session_state.pdf_path):
            os.remove(st.session_state.pdf_path)

        st.session_state.rag_chain = None
        st.session_state.collection_name = None
        st.session_state.pdf_path = None

        st.success("PDF & chat cleared")

# =========================
# Upload PDF
# =========================
uploaded = st.file_uploader("📤 Upload a PDF", type=["pdf"])

if uploaded and not st.session_state.rag_chain:
    with st.spinner("Processing PDF..."):
        name = sanitize(uploaded.name)
        path = f"temp_{name}.pdf"

        with open(path, "wb") as f:
            f.write(uploaded.read())

        st.session_state.pdf_path = path
        st.session_state.collection_name = name

        loader = PyPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1500,
            chunk_overlap=150
        )
        splits = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            collection_name=name,
            client=chroma
        )

        retriever = vectorstore.as_retriever(k=4)

        history_retriever = create_history_aware_retriever(
            llm, retriever, contextualize_q_prompt
        )

        qa_chain = create_stuff_documents_chain(llm, qa_prompt)

        rag_chain = create_retrieval_chain(
            history_retriever, qa_chain
        )

        st.session_state.rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
            output_messages_key="answer",
        )

    st.success("✅ PDF processed. Start chatting!")

# =========================
# Chat UI
# =========================
if st.session_state.rag_chain:
    user_input = st.chat_input("Ask a question about the PDF")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = st.session_state.rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {
                        "session_id": st.session_state.session_id
                    }}
                )

                answer = result["answer"]
                st.write(answer)

                # Save chat
                st.session_state.chat_log.append({
                    "question": user_input,
                    "answer": answer
                })
else:
    st.info("⬆️ Upload a PDF to begin")

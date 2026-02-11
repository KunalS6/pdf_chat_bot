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
# Embeddings (local via transformers)
# =========================
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from langchain_core.embeddings import Embeddings

class HFEmbeddings(Embeddings):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def _encode(self, texts: List[str]) -> List[List[float]]:
        with torch.no_grad():
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            out = self.model(**enc)
            embeddings = out.last_hidden_state.mean(dim=1)
        return embeddings.cpu().tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self._encode(texts)

    def embed_query(self, text: str) -> List[float]:
        return self._encode([text])[0]

@st.cache_resource(show_spinner=False)
def load_embedding():
    return HFEmbeddings()

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
# Runnables / manual RAG
# =========================
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.documents import Document

def docs_to_context(docs: List[Document]) -> str:
    """Join retrieved docs into a single context string."""
    return "\n\n".join(d.page_content for d in docs)

def qa_with_context(inputs: dict) -> str:
    """Take docs + question + history and call the LLM with qa_prompt."""
    context = docs_to_context(inputs["context"])
    messages = qa_prompt.format_messages(
        context=context,
        chat_history=inputs.get("chat_history", []),
        input=inputs["input"],
    )
    res = llm.invoke(messages)
    return res.content if hasattr(res, "content") else str(res)

qa_chain = RunnableLambda(qa_with_context)

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

        # -------- Manual RAG chain (no langchain.chains at all) --------

        def contextualize(inputs: dict) -> dict:
            messages = contextualize_q_prompt.format_messages(
                chat_history=inputs.get("chat_history", []),
                input=inputs["input"],
            )
            result = llm.invoke(messages)
            standalone = result.content if hasattr(result, "content") else str(result)
            return {"input": standalone, "chat_history": inputs.get("chat_history", [])}

        contextualize_chain = RunnableLambda(contextualize)

        rag_chain = (
            contextualize_chain
            | RunnableParallel(
                context=lambda x: retriever.get_relevant_documents(x["input"]),
                input=lambda x: x["input"],
                chat_history=lambda x: x.get("chat_history", []),
            )
            | qa_chain
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

                st.session_state.chat_log.append({
                    "question": user_input,
                    "answer": answer
                })
else:
    st.info("⬆️ Upload a PDF to begin")




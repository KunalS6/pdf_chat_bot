# =========================
# Streamlit UI
# =========================
import streamlit as st

st.set_page_config(page_title="PDF Chatbot (RAG)", layout="wide")
st.title("📄 Conversational PDF Chatbot")
st.caption("Upload a PDF and ask questions (answers ONLY from the PDF)")

# =========================
# Imports
# =========================
import os
import uuid
import re
import csv
import io
import atexit
from dotenv import load_dotenv
load_dotenv()

# =========================
# ENV
# =========================
os.environ["ANONYMIZED_TELEMETRY"] = "False"

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    st.error("❌ GROQ_API_KEY not found")
    st.stop()

# =========================
# LLM (Groq)
# =========================
from langchain_groq import ChatGroq

llm = ChatGroq(
    model="llama-3.1-8b-instant",  # stable free model
    groq_api_key=groq_api_key,
    temperature=0.2,
)

# =========================
# Embeddings (LOCAL - FREE)
# =========================
from transformers import AutoTokenizer, AutoModel
import torch
from typing import List
from langchain_core.embeddings import Embeddings

class HFEmbeddings(Embeddings):
    def __init__(self):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def _encode(self, texts: List[str]):
        with torch.no_grad():
            enc = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )
            out = self.model(**enc)
            embeddings = out.last_hidden_state.mean(dim=1)
        return embeddings.cpu().tolist()

    def embed_documents(self, texts):
        return self._encode(texts)

    def embed_query(self, text):
        return self._encode([text])[0]

@st.cache_resource
def load_embedding():
    return HFEmbeddings()

embedding = load_embedding()

# =========================
# Chroma DB
# =========================
from chromadb import PersistentClient
from langchain_chroma import Chroma

@st.cache_resource
def chroma_client():
    os.makedirs("chroma_store", exist_ok=True)
    return PersistentClient(path="chroma_store")

chroma = chroma_client()

# =========================
# Utils
# =========================
def sanitize(name: str):
    name = os.path.splitext(name)[0].lower()
    name = re.sub(r"[^a-z0-9_-]+", "_", name)
    return name[:63]

# =========================
# PDF Loader
# =========================
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# =========================
# Prompt
# =========================
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

qa_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "Answer ONLY from the context.\n\n{context}\n\n"
     "If not found, say: 'I don't know based on the PDF'."),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

# =========================
# Memory
# =========================
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda

store = {}

def get_session_history(session_id):
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

# =========================
# Session State
# =========================
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "rag_chain" not in st.session_state:
    st.session_state.rag_chain = None

if "chat_log" not in st.session_state:
    st.session_state.chat_log = []

# =========================
# Upload PDF
# =========================
uploaded = st.file_uploader("📤 Upload PDF", type=["pdf"])

if uploaded and not st.session_state.rag_chain:
    with st.spinner("Processing PDF..."):
        path = f"temp_{sanitize(uploaded.name)}.pdf"

        with open(path, "wb") as f:
            f.write(uploaded.read())

        loader = PyPDFLoader(path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=100
        )
        splits = splitter.split_documents(docs)

        vectorstore = Chroma.from_documents(
            splits,
            embedding=embedding,
            collection_name=sanitize(uploaded.name),
            client=chroma
        )

        # =========================
        # RAG FUNCTION
        # =========================
        def rag_fn(inputs):
            try:
                query = inputs["input"]

                # Retrieve docs
                retriever = vectorstore.as_retriever(k=4)
                docs = retriever.invoke(query)

                if not docs:
                    return "No relevant info found in PDF."

                context = "\n\n".join(d.page_content for d in docs)

                messages = qa_prompt.format_messages(
                    context=context,
                    chat_history=inputs.get("chat_history", []),
                    input=query
                )

                response = llm.invoke(messages)

                return response.content if hasattr(response, "content") else str(response)

            except Exception as e:
                return f"⚠️ Error: {str(e)}"

        rag_chain = RunnableLambda(rag_fn)

        st.session_state.rag_chain = RunnableWithMessageHistory(
            rag_chain,
            get_session_history,
            input_messages_key="input",
            history_messages_key="chat_history",
        )

    st.success("✅ PDF ready!")

# =========================
# Chat UI
# =========================
if st.session_state.rag_chain:
    user_input = st.chat_input("Ask something...")

    if user_input:
        with st.chat_message("user"):
            st.write(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = st.session_state.rag_chain.invoke(
                    {"input": user_input},
                    config={"configurable": {
                        "session_id": st.session_state.session_id
                    }}
                )

            st.write(answer)

            st.session_state.chat_log.append({
                "question": user_input,
                "answer": answer
            })

else:
    st.info("⬆️ Upload a PDF to start")
# 📄 PDF Chatbot (RAG) with Groq, LangChain & Streamlit

An interactive **“chat with your PDF”** app built with **Streamlit**, **LangChain 1.x**, **ChromaDB**, and **Groq LLaMA 3.1**.  
Upload any PDF and ask questions; the bot answers **only from the document content** using a Retrieval-Augmented Generation (RAG) pipeline. [web:241][web:243][web:166]

---

## ✨ Features

- 📤 Upload a PDF and chat with it in your browser.
- 🧠 Conversational memory across turns (context-aware questions).
- 📚 RAG pipeline: chunking, embeddings, and retrieval via **ChromaDB**.
- ⚡ Fast LLM responses using **Groq** (`llama-3.1-8b-instant`).
- 💾 Persistent vector store folder (`chroma_store`).
- ⬇️ Export full chat history as CSV.
- 🗑 Clear PDF, vectors, and chat in one click.

---

## 🧱 Tech Stack

- **Frontend:** Streamlit
- **LLM:** Groq `llama-3.1-8b-instant` via `langchain-groq` [web:166][web:165]
- **RAG Orchestration:** LangChain 1.x (LCEL `Runnable` + memory) [web:173]
- **Vector Store:** ChromaDB
- **Embeddings:** Local transformer model (`sentence-transformers/all-MiniLM-L6-v2`) via `transformers` + `torch`
- **Document Loader:** `PyPDFLoader`
- **Chunking:** `RecursiveCharacterTextSplitter`

---

## 📂 Project Structure

```text
.
├── app.py             # Main Streamlit app (UI + RAG logic)
├── requirements.txt   # Python dependencies
└── chroma_store/      # Persistent ChromaDB directory (auto-created)

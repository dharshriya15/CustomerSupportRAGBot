# 🤖 Customer Support RAG Bot

An interactive **Retrieval-Augmented Generation (RAG)** chatbot designed for customer support automation.  
Built with **Streamlit**, **LangChain**, **FAISS**, **HuggingFace embeddings**, and **Groq LLMs**, it allows you to upload documents (FAQs, PDFs, CSVs) and chat with them in real-time.

---

## 🚀 Features
- 📂 Upload **TXT, PDF, CSV** documents as your knowledge base  
- ✂️ Smart text chunking with **LangChain RecursiveCharacterTextSplitter**  
- 🔎 Semantic search powered by **FAISS vectorstore**  
- 💬 Conversational Q&A with **Groq LLMs** (Llama, Mixtral, Gemma, etc.)  
- 🧠 **Conversation memory** for contextual multi-turn dialogue  
- 📑 **Source citations** displayed for every answer  
- 🎨 Modern **Streamlit UI** with sidebar controls and chat history  

---

## 📚 Tech Stack
- **Python 3.10+**
- [Streamlit](https://streamlit.io/) – UI framework  
- [LangChain](https://www.langchain.com/) – RAG pipeline  
- [FAISS](https://faiss.ai/) – Vector database  
- [HuggingFace Transformers](https://huggingface.co/) – Embeddings  
- [Groq LLMs](https://groq.com/) – Fast inference  

---

## ⚙️ Installation

1. **Clone the repository**
   git clone https://github.com/dharshriya15/CustomerSupportRAGBot.git
   cd CustomerSupportRAGBot

2. **Create and activate virtual environment**
  python -m venv venv
  source venv/bin/activate   # Linux/Mac
  venv\Scripts\activate      # Windows

3. **Create and activate virtual environment**
  pip install -r requirements.txt

▶️ Usage

Run the Streamlit app:

  streamlit run app.py

- Enter your Groq API Key in the sidebar

- Select an LLM model (e.g., mixtral-8x7b-32768)

- Upload TXT, PDF, or CSV documents

- Adjust chunk size and overlap (optional)

- Click Process Documents

- Start chatting in the main interface 🚀

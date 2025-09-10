import streamlit as st
import os
from typing import List, Dict
import tempfile
import pickle
from pathlib import Path

# LangChain imports
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferWindowMemory

# Set page config
st.set_page_config(
    page_title="Customer Support RAG Bot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: #ffffff; /* changed to white for better visibility */
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: #212121; /* dark gray text for readability on light backgrounds */
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .bot-message {
        background-color: #f1f8e9;
        border-left: 4px solid #4caf50;
    }
    .sidebar-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: #212121; /* ensures dark text on white background */
    }
</style>

""", unsafe_allow_html=True)

class CustomerSupportRAG:
    def __init__(self):
        self.embeddings = None
        self.vectorstore = None
        self.llm = None
        self.qa_chain = None
        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True)
        
    def initialize_embeddings(self):
        """Initialize HuggingFace embeddings"""
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
        
    def initialize_llm(self, api_key: str, model: str = "mixtral-8x7b-32768"):
        """Initialize Groq LLM"""
        self.llm = ChatGroq(
            api_key=api_key,
            model_name=model,
            temperature=0.1,
            max_tokens=1000
        )
        
    def load_documents(self, uploaded_files) -> List[Document]:
        """Load documents from uploaded files"""
        documents = []
        
        for uploaded_file in uploaded_files:
            # Create temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=f".{uploaded_file.name.split('.')[-1]}") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
                
            try:
                # Load based on file type
                if uploaded_file.name.endswith('.pdf'):
                    loader = PyPDFLoader(tmp_file_path)
                elif uploaded_file.name.endswith('.csv'):
                    loader = CSVLoader(tmp_file_path)
                else:  # Assume text file
                    loader = TextLoader(tmp_file_path, encoding='utf-8')
                    
                docs = loader.load()
                
                # Add metadata
                for doc in docs:
                    doc.metadata['source'] = uploaded_file.name
                    
                documents.extend(docs)
                
            except Exception as e:
                st.error(f"Error loading {uploaded_file.name}: {str(e)}")
            finally:
                # Clean up temp file
                os.unlink(tmp_file_path)
                
        return documents
    
    def create_vectorstore(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200):
        """Create FAISS vectorstore from documents"""
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        
        # Create vectorstore
        self.vectorstore = FAISS.from_documents(chunks, self.embeddings)
        
        return len(chunks)
    
    def setup_qa_chain(self):
        """Setup the QA chain with custom prompt"""
        prompt_template = """
You are a helpful customer support assistant. Use the following context to answer the customer's question.
If you cannot find the answer in the context, politely say that you don't have that information and suggest contacting human support.

Always be:
- Polite and professional
- Clear and concise
- Helpful and solution-oriented

Context: {context}

Question: {question}

Answer:"""

        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )
    
    def get_answer(self, question: str) -> Dict:
        """Get answer from the RAG system"""
        try:
            result = self.qa_chain.invoke({"query": question})
            return {
                "answer": result["result"],
                "source_documents": result["source_documents"],
                "success": True
            }
        except Exception as e:
            return {
                "answer": f"Sorry, I encountered an error: {str(e)}",
                "source_documents": [],
                "success": False
            }
    
    def save_vectorstore(self, path: str):
        """Save vectorstore to disk"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
    
    def load_vectorstore(self, path: str):
        """Load vectorstore from disk"""
        if os.path.exists(path):
            self.vectorstore = FAISS.load_local(path, self.embeddings)
            return True
        return False

# Initialize session state
if 'rag_bot' not in st.session_state:
    st.session_state.rag_bot = CustomerSupportRAG()
    st.session_state.chat_history = []
    st.session_state.vectorstore_ready = False

# Main UI
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Customer Support RAG Bot</h1>
        <p>Powered by Groq, LangChain & FAISS</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("⚙️ Configuration")
        
        # Groq API Key
        groq_api_key = st.text_input(
            "Groq API Key",
            type="password",
            help="Get your API key from https://console.groq.com/keys"
        )
        
        # Model selection
        model_options = [
            "llama-3.3-70b-versatile",
            "mixtral-8x7b-32768",
            "llama2-70b-4096",
            "gemma-7b-it"
        ]
        selected_model = st.selectbox("Select Model", model_options)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Document Upload Section
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.header("📚 Knowledge Base")
        
        uploaded_files = st.file_uploader(
            "Upload Documents",
            accept_multiple_files=True,
            type=['txt', 'pdf', 'csv'],
            help="pload FAQs, manuals, or other support documents"
        )
        
        # Chunking parameters
        chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
        chunk_overlap = st.slider("Chunk Overlap", 50, 500, 200)
        
        # Process documents button
        if st.button("🔄 Process Documents", disabled=not uploaded_files or not groq_api_key):
            if uploaded_files and groq_api_key:
                with st.spinner("Processing documents..."):
                    try:
                        # Initialize components
                        st.session_state.rag_bot.initialize_embeddings()
                        st.session_state.rag_bot.initialize_llm(groq_api_key, selected_model)
                        
                        # Load and process documents
                        documents = st.session_state.rag_bot.load_documents(uploaded_files)
                        
                        if documents:
                            num_chunks = st.session_state.rag_bot.create_vectorstore(
                                documents, chunk_size, chunk_overlap
                            )
                            st.session_state.rag_bot.setup_qa_chain()
                            
                            st.session_state.vectorstore_ready = True
                            st.success(f"✅ Processed {len(documents)} documents into {num_chunks} chunks!")
                        else:
                            st.error("No documents were successfully loaded.")
                            
                    except Exception as e:
                        st.error(f"Error processing documents: {str(e)}")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Statistics
        if st.session_state.vectorstore_ready:
            st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
            st.header("📊 Statistics")
            st.metric("Documents Processed", len(uploaded_files) if uploaded_files else 0)
            st.metric("Chat Messages", len(st.session_state.chat_history))
            st.markdown('</div>', unsafe_allow_html=True)
    
    # Main chat interface
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.header("💬 Chat Interface")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["type"] == "user":
                    st.markdown(f"""
                    <div class="chat-message user-message">
                        <strong>You:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="chat-message bot-message">
                        <strong>Assistant:</strong> {message["content"]}
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if "sources" in message and message["sources"]:
                        with st.expander("📄 Sources"):
                            for i, source in enumerate(message["sources"], 1):
                                st.write(f"**Source {i}:** {source.metadata.get('source', 'Unknown')}")
                                st.write(f"*Content:* {source.page_content[:200]}...")
        
        # Chat input
        if st.session_state.vectorstore_ready:
            user_question = st.text_input(
                "Ask a question:",
                placeholder="How can I help you today?",
                key="user_input"
            )
            
            col_send, col_clear = st.columns([1, 1])
            
            with col_send:
                if st.button("Send", type="primary") and user_question:
                    # Add user message to history
                    st.session_state.chat_history.append({
                        "type": "user",
                        "content": user_question
                    })
                    
                    # Get bot response
                    with st.spinner("Thinking..."):
                        response = st.session_state.rag_bot.get_answer(user_question)
                        
                        # Add bot response to history
                        st.session_state.chat_history.append({
                            "type": "bot",
                            "content": response["answer"],
                            "sources": response["source_documents"] if response["success"] else []
                        })
                    
                    # Clear input and rerun
                    st.rerun()
            
            with col_clear:
                if st.button("Clear Chat"):
                    st.session_state.chat_history = []
                    st.rerun()
        else:
            st.info("👆 Please configure your API key and upload documents to start chatting!")
    
    with col2:
        st.header("🚀 Quick Start")
        st.markdown("""
        **Steps to get started:**
        
        1. **Get Groq API Key**
           - Visit [console.groq.com](https://console.groq.com/keys)
           - Create a free account
           - Generate API key
        
        2. **Upload Documents**
           - FAQs (TXT, PDF)
           - Product manuals
           - Policy documents
           - CSV files with Q&A
        
        3. **Configure Settings**
           - Choose model
           - Adjust chunk parameters
           - Process documents
        
        4. **Start Chatting!**
           - Ask questions
           - Get instant answers
           - View source documents
        """)
        
        # Sample questions
        if st.session_state.vectorstore_ready:
            st.header("💡 Sample Questions")
            sample_questions = [
                "What is your return policy?",
                "How do I reset my password?",
                "What are your business hours?",
                "How can I contact support?",
                "What payment methods do you accept?"
            ]
            
            for question in sample_questions:
                if st.button(question, key=f"sample_{question}"):
                    st.session_state.chat_history.append({
                        "type": "user",
                        "content": question
                    })
                    
                    with st.spinner("Thinking..."):
                        response = st.session_state.rag_bot.get_answer(question)
                        
                        st.session_state.chat_history.append({
                            "type": "bot",
                            "content": response["answer"],
                            "sources": response["source_documents"] if response["success"] else []
                        })
                    
                    st.rerun()

if __name__ == "__main__":
    main()
import streamlit as st
import os
import shutil
from langchain_community.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama
from get_embedding_function import get_embedding_function


CHROMA_PATH = "chroma"
DATA_PATH = "data"

st.set_page_config(
    page_title="Document Chat Assistant",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .assistant-message {
        background-color: #f3e5f5;
        border-left: 4px solid #9c27b0;
    }
    .source-info {
        font-size: 0.8rem;
        color: #666;
        margin-top: 0.5rem;
        font-style: italic;
    }
    .score-info {
        font-size: 0.7rem;
        color: #888;
        margin-top: 0.25rem;
        font-style: italic;
    }
    .stButton > button {
        width: 100%;
        margin-top: 0.5rem;
    }
    .database-status {
        padding: 0.5rem;
        border-radius: 0.25rem;
        margin: 0.5rem 0;
    }
    .status-success {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .status-warning {
        background-color: #fff3cd;
        color: #856404;
        border: 1px solid #ffeaa7;
    }
    .database-title {
        font-size: 1.5rem;
        font-weight: bold;
        color: white;
        margin-bottom: 1rem;
    }
    .reset-button {
        background-color: #dc2626 !important;
        color: white !important;
    }
    .reset-button:hover {
        background-color: #b91c1c !important;
    }
    .update-button {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    .update-button:hover {
        background-color: #2563eb !important;
    }
    /* Button styling */
    .update-db-btn {
        background-color: #3b82f6 !important;
        color: white !important;
    }
    .update-db-btn:hover {
        background-color: #2563eb !important;
    }
    .reset-db-btn {
        background-color: #dc2626 !important;
        color: white !important;
    }
    .reset-db-btn:hover {
        background-color: #b91c1c !important;
    }
    /* Hide chat icons */
    .stChatMessage {
        background: none !important;
    }
    .stChatMessage > div:first-child {
        display: none !important;
    }
    
    /* Chat layout styling */
    .chat-container {
        max-width: 800px;
        margin: 0 auto;
        padding: 20px;
    }
    
    .user-message-container {
        display: flex;
        justify-content: flex-end;
        margin-bottom: 20px;
    }
    
    .assistant-message-container {
        display: flex;
        justify-content: flex-start;
        margin-bottom: 20px;
    }
    
    .user-message-bubble {
        background-color: #3b82f6;
        color: white;
        padding: 12px 16px;
        border-radius: 18px 18px 4px 18px;
        max-width: 70%;
        word-wrap: break-word;
    }
    
    .assistant-message-bubble {
        background-color: #f3f4f6;
        color: #1f2937;
        padding: 12px 16px;
        border-radius: 18px 18px 18px 4px;
        max-width: 70%;
        word-wrap: break-word;
    }
    
    .chat-input-container {
        display: flex;
        justify-content: center;
        margin-top: 30px;
        padding: 20px;
    }
    
    .chat-input-wrapper {
        max-width: 600px;
        width: 100%;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'database_updated' not in st.session_state:
        st.session_state.database_updated = False
    if 'confirm_reset' not in st.session_state:
        st.session_state.confirm_reset = False

def get_available_documents():
    """Get list of available PDF documents in the data directory"""
    if not os.path.exists(DATA_PATH):
        return []
    
    pdf_files = []
    for file in os.listdir(DATA_PATH):
        if file.lower().endswith('.pdf'):
            pdf_files.append(file)
    return pdf_files

def save_uploaded_file(uploaded_file):
    """Save uploaded file to data directory"""
    try:
        if not os.path.exists(DATA_PATH):
            os.makedirs(DATA_PATH)
        
        file_path = os.path.join(DATA_PATH, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        
        if os.path.exists(file_path):
            file_size = os.path.getsize(file_path)
            return file_path, file_size
        else:
            return None, 0
    except Exception as e:
        st.error(f"Error saving file: {str(e)}")
        return None, 0

def check_database_status():
    """Check if the database exists and has documents"""
    if not os.path.exists(CHROMA_PATH):
        return False, 0
    
    try:
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
        existing_items = db.get(include=[])
        doc_count = len(existing_items["ids"])
        
        del db
        import gc
        gc.collect() 
        return True, doc_count
    except Exception as e:
        st.warning(f"Database check failed: {str(e)}")
        return False, 0

def clear_database():
    """Clear the vector database and all PDF files"""
    try:
        import time
        import gc
        
       
        gc.collect()
        
      
        if os.path.exists(DATA_PATH):
            pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith('.pdf')]
            deleted_count = 0
            for pdf_file in pdf_files:
                try:
                    pdf_path = os.path.join(DATA_PATH, pdf_file)
                    os.remove(pdf_path)
                    deleted_count += 1
                except Exception as e:
                    st.warning(f"Could not delete {pdf_file}: {str(e)}")
            
            if deleted_count > 0:
                st.success(f"Deleted {deleted_count} PDF files from data directory!")
            elif pdf_files:
                st.warning("Some PDF files could not be deleted")
        
        
        if os.path.exists(CHROMA_PATH):
            
            try:
               
                for root, dirs, files in os.walk(CHROMA_PATH, topdown=False):
                    for file in files:
                        try:
                            file_path = os.path.join(root, file)
                            os.remove(file_path)
                        except Exception as e:
                            st.warning(f"Could not delete {file}: {str(e)}")
                    
                    for dir in dirs:
                        try:
                            dir_path = os.path.join(root, dir)
                            os.rmdir(dir_path)
                        except Exception as e:
                            st.warning(f"Could not delete directory {dir}: {str(e)}")
                
                
                try:
                    os.rmdir(CHROMA_PATH)
                    st.success("Vector database cleared successfully!")
                except Exception as e:
                    st.warning(f"Could not remove main chroma directory: {str(e)}")
                    
                    try:
                        import tempfile
                        temp_dir = tempfile.mkdtemp()
                        new_name = os.path.join(temp_dir, "chroma_old")
                        os.rename(CHROMA_PATH, new_name)
                        st.success("Vector database marked for deletion!")
                    except Exception as rename_error:
                        st.warning(f"Database files may still be in use: {str(rename_error)}")
                
            except Exception as e:
                st.error(f"Error clearing database: {str(e)}")
                return False
        
       
        gc.collect()
        
        return True
    except Exception as e:
        st.error(f"Error clearing database: {str(e)}")
        return False

def load_documents():
    """Load documents from the data directory"""
    if not os.path.exists(DATA_PATH):
        st.error(f"Data directory '{DATA_PATH}' not found!")
        return []
    
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()

def split_documents(documents):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def calculate_chunk_ids(chunks):
    """Calculate unique IDs for document chunks"""
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id
        chunk.metadata["id"] = chunk_id

    return chunks

def update_database():
    """Update the vector database with documents using populate_database.py logic"""
    try:
        
        if not os.path.exists(DATA_PATH):
            st.error("Data directory does not exist!")
            return False
        
        pdf_files = [f for f in os.listdir(DATA_PATH) if f.lower().endswith('.pdf')]
        if not pdf_files:
            st.error("No PDF files found in the data directory!")
            return False
        
        st.info(f"Found {len(pdf_files)} PDF files: {', '.join(pdf_files)}")
        
        with st.spinner("Loading documents..."):
            documents = load_documents()
            if not documents:
                st.error("No documents could be loaded from PDF files!")
                return False
        
        with st.spinner("Splitting documents into chunks..."):
            chunks = split_documents(documents)
            st.info(f"Created {len(chunks)} text chunks")
        
        with st.spinner("Processing chunks..."):
            chunks_with_ids = calculate_chunk_ids(chunks)
        
        with st.spinner("Adding to vector database..."):
            db = Chroma(persist_directory=CHROMA_PATH, embedding_function=get_embedding_function())
            
            existing_items = db.get(include=[])
            existing_ids = set(existing_items["ids"])
            
            new_chunks = []
            for chunk in chunks_with_ids:
                if chunk.metadata["id"] not in existing_ids:
                    new_chunks.append(chunk)
            
            if new_chunks:
                new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
                db.add_documents(new_chunks, ids=new_chunk_ids)
                db.persist()
                st.success(f"Successfully added {len(new_chunks)} new chunks to database!")
                st.success(f"Total documents in database: {len(existing_ids) + len(new_chunks)}")
            else:
                st.info("No new documents to add - database is up to date!")
                st.info(f"Total documents in database: {len(existing_ids)}")
            
            
            del db
            import gc
            gc.collect()
        
        return True
    except Exception as e:
        st.error(f"Error updating database: {str(e)}")
        st.error(f"Error details: {type(e).__name__}")
        return False

def query_rag(query_text):
    """Query the RAG system"""
    try:
        embedding_function = get_embedding_function()
        db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
        
        results = db.similarity_search_with_score(query_text, k=5)
        
        context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
        
        prompt_template = ChatPromptTemplate.from_template("""
        Answer the question based only on the following context.

        {context}

        ---

        Question: {question}

        Instructions: Provide ONLY a direct, concise answer. Do not say "Based on the context" or "According to the documents" or any other introductory phrases. Do not elaborate or explain your reasoning. Give the answer immediately in 1-2 sentences maximum.
        
        Answer:""")
        
        prompt = prompt_template.format(context=context_text, question=query_text)
        
        model = Ollama(model="deepseek-r1:1.5b")
        response_text = model.invoke(prompt)
        
        sources = [doc.metadata.get("id", None) for doc, _score in results]
        scores = [score for doc, score in results]
        
        
        del db
        import gc
        gc.collect()
        
        return response_text, sources, scores
    except Exception as e:
        return f"Error: {str(e)}", [], []

def main():
    initialize_session_state()
    
   
    with st.sidebar:
        st.markdown('<div class="database-title">Database Management</div>', unsafe_allow_html=True)
        
        # Database actions
        st.subheader("Database Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            update_clicked = st.button("Update Database", key="update_db", use_container_width=True)
            if update_clicked:
                if update_database():
                    st.session_state.database_updated = True
                    st.rerun()
        
        with col2:
            reset_clicked = st.button("Reset Database", key="reset_db", use_container_width=True)
            if reset_clicked:
                
                if st.session_state.get('confirm_reset', False):
                    if clear_database():
                        st.session_state.database_updated = True
                        st.session_state.confirm_reset = False
                        st.rerun()
                else:
                    st.session_state.confirm_reset = True
                    st.warning("⚠️ This will delete ALL documents and the database. Click 'Reset Database' again to confirm.")
                    st.rerun()
        
        st.divider()
        
        # File upload section
        st.subheader("Upload Files")
        uploaded_files = st.file_uploader(
            "Choose PDF files to upload",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF files to add them to your document collection"
        )
        
        if uploaded_files:
            for uploaded_file in uploaded_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.write(f"📄 {uploaded_file.name} ({uploaded_file.size} bytes)")
                with col2:
                    if st.button(f"Save", key=f"save_{uploaded_file.name}"):
                        file_path, file_size = save_uploaded_file(uploaded_file)
                        if file_path:
                            st.success(f"✅ Saved: {uploaded_file.name} ({file_size} bytes)")
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to save: {uploaded_file.name}")
        
        st.divider()
        
        # Available documents (real-time)
        st.subheader("Available Documents")
        available_docs = get_available_documents()
        if available_docs:
            st.write(f"📚 **{len(available_docs)} PDF files found:**")
            for doc in available_docs:
                doc_path = os.path.join(DATA_PATH, doc)
                if os.path.exists(doc_path):
                    file_size = os.path.getsize(doc_path)
                    st.write(f"• **{doc}** ({file_size:,} bytes)")
                else:
                    st.write(f"• {doc} (file not found)")
        else:
            st.info("No PDF documents found in data directory")
        
        # Auto-refresh available documents
        if st.button("🔄 Refresh Documents List"):
            st.rerun()
    
    # Check if database is ready
    db_exists, _ = check_database_status()
    if not db_exists:
        st.warning("Database not initialized. Please update the database from the sidebar first.")
        return
    
    # Main chat interface - Title and description in main content area
    st.markdown("""
    <div style="margin-bottom: 30px;">
        <h1 style="font-size: 4.5rem; font-weight: 900; margin: 0; color: white; line-height: 1;">VectorVault</h1>
        <p style="font-size: 1.4rem; font-weight: 600; margin: 10px 0 0 0; color: #e5e7eb; font-style: italic;">knowledge in, insights out</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Chat interface
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Display chat messages
    for message in st.session_state.messages:
        if message["role"] == "user":
            st.markdown(f'''
                <div class="user-message-container">
                    <div class="user-message-bubble">
                        {message["content"]}
                    </div>
                </div>
            ''', unsafe_allow_html=True)
        else:
            st.markdown(f'''
                <div class="assistant-message-container">
                    <div class="assistant-message-bubble">
                        {message["content"]}
                        {f'<div class="source-info">Sources: {", ".join(message["sources"])}</div>' if "sources" in message and message["sources"] else ""}
                        {f'<div class="score-info">Average Relevance Score: {sum(message["scores"]) / len(message["scores"]):.3f}</div>' if "scores" in message and message["scores"] else ""}
                    </div>
                </div>
            ''', unsafe_allow_html=True)
    
    # Chat input
    st.markdown('<div class="chat-input-container">', unsafe_allow_html=True)
    st.markdown('<div class="chat-input-wrapper">', unsafe_allow_html=True)
    
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get AI response
        with st.spinner("Thinking..."):
            response, sources, scores = query_rag(prompt)
            
            # Add assistant message to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response,
                "sources": sources,
                "scores": scores
            })
        
        st.rerun()
    
    st.markdown('</div></div></div>', unsafe_allow_html=True)
    
    # Clear chat button
    if st.session_state.messages:
        if st.button("Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

if __name__ == "__main__":
    main()

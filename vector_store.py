import chromadb
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Initialize embedding model 
embedding_model = HuggingFaceEmbeddings(model_name="BAAI/bge-base-en")

# ChromaDB Storage Path (Works in Streamlit Cloud)
CHROMA_PERSIST_DIR = "/tmp/chroma_db"  # Streamlit cloud will use `/tmp/`

def store_vectors(texts):
    """Chunk text and store embeddings in ChromaDB."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.create_documents(texts)
    
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    vector_store = Chroma.from_documents(chunks, embedding_model, persist_directory=CHROMA_PERSIST_DIR)
    
    return vector_store

def load_vector_store():
    """Load existing ChromaDB index."""
    return Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embedding_model)

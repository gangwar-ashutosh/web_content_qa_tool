__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import requests
from bs4 import BeautifulSoup
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.documents import Document
import os

# Define the directory for persistent storage of the Chroma vector database
PERSIST_DIRECTORY = os.path.join(os.getcwd(), 'data', 'chroma_db')

# Initialize Hugging Face embeddings using a high-performance sentence transformer
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

# Initialize or load Chroma vector store with persistence enabled
vector_store = Chroma(
    collection_name="web_content",  # Name of the Chroma collection
    embedding_function=embeddings,  # Function to generate text embeddings
    persist_directory=PERSIST_DIRECTORY  # Directory to store the vector database
)

def fetch_and_process(url):
    """
    Fetches and processes web page content, splits it into chunks, 
    and stores the embeddings in ChromaDB.

    Args:
        url (str): The URL of the web page to scrape and process.

    Returns:
        None
    """
    # Send an HTTP GET request to fetch the web page content
    response = requests.get(url)
    
    # Parse the HTML content using BeautifulSoup
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract only the readable text, removing HTML tags
    content = soup.get_text()

    # Split text into manageable chunks for embedding and retrieval
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(content)

    # Convert text chunks into LangChain Document objects
    documents = [Document(page_content=chunk) for chunk in chunks]

    # Store the document embeddings in the Chroma vector store
    vector_store.add_documents(documents)

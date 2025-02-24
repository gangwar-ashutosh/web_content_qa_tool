import os
from langchain.llms import Groq
from langchain.chains import RetrievalQA
from vector_store import load_vector_store
from dotenv import load_dotenv, dotenv_values
from config import config

env_vars = dotenv_values(config['api_key_path'])
os.environ['GROQ_API_KEY']=env_vars.get("GROQ_API_KEY")

#GROQ_API_KEY = os.getenv("GROQ_API_KEY")

def get_llm():
    """Initialize GROQ LLM (Mistral or DeepSeek)."""
    return Groq(model_name="mistral-7b", temperature=0.2)

def query_rag(question):
    """Retrieve relevant context & generate an answer using RAG."""
    vector_store = load_vector_store()
    retriever = vector_store.as_retriever()
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=get_llm(),
        retriever=retriever
    )
    
    return qa_chain.run(question)

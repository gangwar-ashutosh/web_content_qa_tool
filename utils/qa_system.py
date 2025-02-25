from langchain.chat_models import init_chat_model
from langchain.prompts import PromptTemplate
from utils.content_processor import vector_store  # Import vector store for similarity search
from dotenv import dotenv_values  # Load environment variables from .env file
from langchain_groq import ChatGroq  # Groq API for LLM inference
import os
import streamlit as st


# Load environment variables from .env file, when running locally
# env_vars = dotenv_values('.env')
# os.environ['GROQ_API_KEY'] = env_vars.get("GROQ_API_KEY")  # Set API key for authentication

## Import key from streamlit secret file, when deployed to streamlit
os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]

# ------------------------ Custom Prompt Template ------------------------
custom_prompt = """
You are a helpful assistant tasked with providing answers based only on the given context.

- If the answer is not in the context, respond with "I don't know." Do not create or assume information.  
- Keep your response concise, with a maximum of four sentences.  
- Stick strictly to the provided context—do not add extra details.  
- Do not generate any content that is offensive or irrelevant.

Context:  
{context}  

Question:  
{question}  
"""

# Define a prompt template using LangChain's PromptTemplate
prompt_template = PromptTemplate.from_template(custom_prompt)

# ------------------------ Answer Generation Function ------------------------
def answer_question(question, model, temp=0.0):
    """
    Retrieves relevant documents, formats a prompt, and generates an answer using a selected LLM.

    Args:
        question (str): The user's question.
        model (str): The name of the language model to use ("Mistral" or "LLaMA").
        temp (float, optional): The temperature setting for response generation (default: 0.0).

    Returns:
        str: The model-generated response based on retrieved context.
    """

    # ------------------ Initialize the Groq Chat Model ------------------
    if model == "Mistral":
        llm = ChatGroq(
            model="mixtral-8x7b-32768",  # Mistral-based model
            temperature=temp  
        )
    elif model == "LLaMA":
        llm = ChatGroq(
            model="llama-3.2-1b-preview",  # LLaMA-based model
            temperature=temp
        )
    else:
        raise ValueError("Invalid model name. Choose either 'Mistral' or 'LLaMA'.")

    # ------------------ Retrieve Relevant Documents ------------------
    retrieved_docs = vector_store.similarity_search(question, k=5)  # Fetch top 5 relevant documents
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])  # Combine retrieved document content

    # ------------------ Format the Prompt ------------------
    prompt = prompt_template.format(context=context, question=question)

    # ------------------ Generate a Response ------------------
    response = llm.invoke(prompt)  # Query the selected LLM with the formatted prompt

    return response.content  # Return the generated answer

__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from dotenv import dotenv_values
from utils.content_processor import fetch_and_process  # Function to fetch & process web content
from utils.qa_system import answer_question  # Function to generate answers using LLM
from utils.config import config  # Configuration settings
from dotenv import load_dotenv
import os
import torch
from urllib.parse import urlparse  # Used for URL validation


# Workaround for potential Torch class path issues
torch.classes.__path__ = []

# ------------------------ Load Environment Variables ------------------------
# env_vars = dotenv_values('.env')  # Load values from .env file, when running locally
# os.environ['GROQ_API_KEY'] = env_vars.get("GROQ_API_KEY")  # Set API key for authentication

## Import key from streamlit secret file, when deployed to streamlit
os.environ['GROQ_API_KEY'] = st.secrets["GROQ_API_KEY"]

# ------------------------ Initialize Session State Variables ------------------------
if 'urls_processed' not in st.session_state:
    st.session_state['urls_processed'] = False  # Track if URLs are processed
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = []  # Store chat history

# ------------------------ Sidebar Configuration ------------------------
st.sidebar.header("Configuration")

# Model selection (Default: Mistral)
model = st.sidebar.selectbox(
    "Select LLM:",
    ("DeepSeek","Mistral"),
    index=0  # Default to "Mistral" (0-based index)
)

# Temperature slider for controlling response randomness
temp = st.sidebar.slider(
    "Select Temperature of LLM:",
    min_value=0.0,
    max_value=1.0,
    value=0.0,  # Default to deterministic responses
    step=0.1
)

# ------------------------ Main Application ------------------------
st.title("Web Content Q&A Tool")

# Input fields for URLs
url1 = st.text_input("Enter URL 1:")
url2 = st.text_input("Enter URL 2:")
url3 = st.text_input("Enter URL 3:")

# Collect entered URLs (remove empty inputs)
urls = [url.strip() for url in [url1, url2, url3] if url.strip()]

# ------------------------ Button to Process URLs ------------------------
if st.button("Run"):
    # Reset processing state and chat history
    st.session_state['urls_processed'] = False

    if not urls:
        st.error("Please enter at least one URL.")
    else:
        ## find valid URLS list
        valid_urls_list=[]

        for url in urls:
         # Validate URL format
          result = urlparse(url)
          if all([result.scheme, result.netloc]):
             valid_urls_list.append(url)

        if len(valid_urls_list>=1):
            st.session_state['chat_history'] = []  # Reset chat history

            with st.spinner("Fetching and Processing Content..."):
                # Fetch and process each valid URL
                for url in valid_urls_list:
                    fetch_and_process(url)

            st.session_state['urls_processed'] = True
            st.success("Processing complete. Processed all valid URLs.")
        else:
            st.error("All URLs entered are invalid, please enter again.")


# ------------------------ Q&A Interface (Only Show After Processing) ------------------------
if st.session_state['urls_processed']:
    st.write("### Q&A Interface:")

    # Display chat history
    for role, message in st.session_state['chat_history']:
        with st.chat_message(role):
            st.markdown(message)

    # Chat input field
    if user_input := st.chat_input("Your Question:"):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Append user question to chat history
        st.session_state['chat_history'].append(("user", user_input))

        # Generate response from selected LLM
        answer = answer_question(user_input, model, temp)

        # Display assistant's response
        with st.chat_message("assistant"):
            st.markdown(answer)
        
        # Append assistant response to chat history
        st.session_state['chat_history'].append(("assistant", answer))

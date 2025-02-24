import streamlit as st
from scrape import fetch_multiple_urls
from vector_store import store_vectors
from qa_chain import query_rag
from langchain.memory import ConversationBufferMemory


## CONVERSATIONAL BUFFER MEMORY
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True
)


st.set_page_config(page_title="Web Content Q&A", layout="wide")
st.title("📖 Web Content Q&A Tool (RAG + GROQ)")

st.sidebar.header("🔗 Enter up to 3 URLs")
urls_input = st.sidebar.text_area("Enter URLs (comma-separated)")
process_btn = st.sidebar.button("Ingest Content")

if process_btn:
    urls = [url.strip() for url in urls_input.split(",") if url.strip()]
    if len(urls) > 3:
        st.sidebar.error("Only up to 3 URLs allowed!")
    else:
        texts = fetch_multiple_urls(urls)
        if texts:
            store_vectors(texts)
            st.sidebar.success("Content stored successfully! You can now ask questions.")
        else:
            st.sidebar.error("Failed to process URLs.")

st.header("❓ Ask a Question")
question = st.text_input("Enter your question:")
ask_btn = st.button("Get Answer")

if ask_btn:
    if not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Searching for an answer..."):
            answer = query_rag(question)
            memory.save_context({"input": question}, {"output": answer})
        
        st.write(f"### 📢 Answer: {answer}")

        # Display memory
        st.subheader("💬 Conversation History")
        for msg in memory.chat_memory.messages:
            st.text(f"{msg.type.upper()}: {msg.content}")

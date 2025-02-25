# Web Content Q&A Tool  

A **Streamlit-based app** that extracts web content from URLs and enables **Q&A using LLMs (Mistral & LLaMA)**.  

## 🚀 Features  
- Extracts & processes web content  
- Uses **ChromaDB** for efficient search  
- Supports **Mistral & LLaMA** for Q&A  
- Configurable **temperature settings**  

## 📌 Installation  
```bash
git clone https://github.com/gangwar-ashutosh/web_content_qa_tool.git  
cd web_content_qa_tool  
python -m venv venv  
source venv/bin/activate  # (Windows: venv\Scripts\activate)  
pip install -r requirements.txt  
```

## 🔑 API Key Configuration  
Create a `.env` file and add your Groq API key:  
```ini
GROQ_API_KEY=your_groq_api_key_here
```

## ▶️ Usage  
Uncomment the code for loading env variables when running locally in qa_system.py and app.py files
```bash
streamlit run app.py
```

## 📂 Folder Structure  
```
📂 utils/  
   ├── content_processor.py  # Fetches content from URLs, chunks text, and stores in ChromaDB  
   ├── qa_system.py          # Generates responses based on vector search results  
📂 data/chroma_db/           # Persistent ChromaDB storage  
app.py                       # Main Streamlit app  
requirements.txt             # Dependencies  
.env                         # API Key (ignored in Git)  
```


import requests
from bs4 import BeautifulSoup
from newspaper import Article

def clean_url(url):
    """Ensure proper URL format."""
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    return url

def extract_content(url):
    """Scrape content from the given URL."""
    try:
        url = clean_url(url)
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except Exception as e:
        return f"Error extracting content from {url}: {e}"

def fetch_multiple_urls(urls):
    """Fetch content from up to 3 URLs."""
    urls = list(set(urls))[:3]  
    all_texts = []
    
    for url in urls:
        text = extract_content(url)
        if text:
            all_texts.append(text)
    
    return all_texts

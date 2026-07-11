# 🏢 GigaCorp Customer Support Assistant (RAG Chatbot)

A modern, interactive customer support assistant that answers queries using a mock knowledge base (RAG), citations, and conversation context. Built using Python, Streamlit, and LangChain.

## 🚀 Key Features

1. **Hybrid Retrieval Engines**:
   - **Semantic Search**: Powered by LangChain's FAISS indexing using OpenAI or Google Gemini embeddings.
   - **Keyword Search (TF-IDF)**: A zero-dependency pure-Python TF-IDF vectorizer that allows the application to search the FAQ database instantly offline with no API keys.
2. **Dynamic LLM Providers**:
   - Supports **OpenAI** (`gpt-4o-mini`), **Google Gemini** (`gemini-1.5-flash`), and **Anthropic** (`claude-3-5-haiku`).
   - Includes **Demo / Mock Mode** which functions 100% offline, retrieving matching FAQ sections without making API calls—allowing immediate evaluation.
3. **Conversational Memory & Context-Aware Rephrasing**:
   - Automatically maintains the context of the conversation.
   - Rephrases pronouns and relative references (e.g., "Do you ship to India?" followed by "How much does it cost *there*?") to perform precise semantic retrieval.
4. **Verified Citations UI**:
   - Lists exact line numbers (e.g., `[Lines 6-7]`) and highlights retrieved source texts directly in the conversation panel.
5. **Premium Dark/Light Aesthetic**:
   - Polished typography, custom banners, custom CSS blocks, and layout design.

---

## 🛠️ Local Installation & Setup

### Prerequisites
- Python 3.9 or higher

### Steps
1. **Clone or Navigate to the Directory**:
   ```bash
   cd /Users/akshaypartapsingh/Documents/assignment1_truelyIAS
   ```

2. **Set up a Virtual Environment** (Highly Recommended):
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Application**:
   ```bash
   streamlit run app.py
   ```

5. Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## ☁️ Free-Tier Hosting Guide

Streamlit applications can be hosted for free on **Streamlit Community Cloud**, **Hugging Face Spaces**, or **Render**. Here are the step-by-step guidelines for Streamlit Community Cloud:

### Option A: Streamlit Community Cloud (Recommended)
This is the easiest, fastest, and most performant hosting option for this application.
1. Push this directory to a public GitHub repository.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and log in with your GitHub account.
3. Click **"New App"**.
4. Select your Repository, Branch, and Main File Path (`app.py`).
5. Click **"Deploy"**. Your application will be live in under 2 minutes!

### Option B: Hugging Face Spaces
1. Create a free account on [Hugging Face](https://huggingface.co/).
2. Create a new Space:
   - Select **Streamlit** as the SDK.
   - Choose the **Free** CPU Basic hardware tier.
3. Commit and push the repository files (`app.py`, `requirements.txt`, `gigacorp_faq.txt`) to the Space repository.
4. Hugging Face will build and launch your application automatically!

---

## 📁 File Structure
- `app.py`: Main Streamlit app containing UI layout, search engine routing, custom styling, and RAG execution logic.
- `gigacorp_faq.txt`: Structured knowledge base FAQ file for "GigaCorp".
- `requirements.txt`: Python package dependencies.
- `README.md`: Setup and deployment manual.

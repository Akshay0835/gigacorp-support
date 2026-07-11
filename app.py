import streamlit as st
import os
import re
import math
from collections import Counter
from langchain.schema import Document

# Set page config
st.set_page_config(
    page_title="GigaCorp Support Center",
    layout="centered",
    initial_sidebar_state="expanded"
)

# =====================================================================
# 1. PURE PYTHON VECTOR STORE & RETRIEVER (TF-IDF Indexer)
# =====================================================================
def tokenize(text):
    return re.findall(r'\w+', text.lower())

class PurePythonVectorStore:
    def __init__(self, documents):
        self.documents = documents
        self.doc_tokens = [tokenize(doc.page_content) for doc in documents]
        self.doc_counts = [Counter(tokens) for tokens in self.doc_tokens]
        
        # Build vocabulary
        self.vocab = set(word for tokens in self.doc_tokens for word in tokens)
        
        # Document Frequency
        self.df = Counter()
        for unique_words in [set(tokens) for tokens in self.doc_tokens]:
            for word in unique_words:
                self.df[word] += 1
                
        self.num_docs = len(documents)
        
    def get_tfidf_vector(self, counter):
        vec = {}
        for word, count in counter.items():
            tf = count
            # Smoothed IDF
            idf = math.log((1 + self.num_docs) / (1 + self.df.get(word, 0))) + 1
            vec[word] = tf * idf
        return vec

    def cosine_similarity(self, vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum(vec1[x] * vec2[x] for x in intersection)
        
        sum1 = sum(val**2 for val in vec1.values())
        sum2 = sum(val**2 for val in vec2.values())
        
        if not sum1 or not sum2:
            return 0.0
        return numerator / (math.sqrt(sum1) * math.sqrt(sum2))

    def similarity_search_with_score(self, query, k=3):
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
            
        query_counter = Counter(query_tokens)
        query_vec = self.get_tfidf_vector(query_counter)
        
        results = []
        for idx, doc in enumerate(self.documents):
            doc_counter = self.doc_counts[idx]
            doc_vec = self.get_tfidf_vector(doc_counter)
            sim = self.cosine_similarity(query_vec, doc_vec)
            results.append((sim, doc))
            
        # Sort by similarity score descending
        results.sort(key=lambda x: x[0], reverse=True)
        # Return docs and scores
        return [(doc, sim) for sim, doc in results[:k] if sim > 0]

    def similarity_search(self, query, k=3):
        res = self.similarity_search_with_score(query, k=k)
        return [doc for doc, score in res]

# =====================================================================
# 2. DOCUMENT PARSER (Retaining Line Numbers & Sections)
# =====================================================================
@st.cache_data
def load_and_parse_faq(file_path="gigacorp_faq.txt"):
    if not os.path.exists(file_path):
        return []
        
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        
    chunks = []
    current_section = "General"
    current_block = []
    start_line = 1
    
    for i, line in enumerate(lines, 1):
        line_str = line.strip()
        
        # If we see a new section heading
        if line_str.startswith("=== SECTION"):
            if current_block:
                chunks.append(Document(
                    page_content="\n".join(current_block),
                    metadata={
                        "section": current_section,
                        "start_line": start_line,
                        "end_line": i - 1,
                        "source": f"{current_section} (Lines {start_line}-{i-1})"
                    }
                ))
                current_block = []
            current_section = line_str.replace("===", "").strip()
            start_line = i + 1
            continue
            
        # If we see a new question tag
        if line_str.startswith("[Line") and re.search(r"Q\d*:", line_str):
            if current_block:
                chunks.append(Document(
                    page_content="\n".join(current_block),
                    metadata={
                        "section": current_section,
                        "start_line": start_line,
                        "end_line": i - 1,
                        "source": f"{current_section} (Lines {start_line}-{i-1})"
                    }
                ))
                current_block = []
            
            # Extract line numbers from format e.g. [Line 6] or standard lines
            line_match = re.search(r'\[Line (\d+)\]', line_str)
            if line_match:
                start_line = int(line_match.group(1))
            else:
                start_line = i
                
            current_block.append(line.rstrip())
        else:
            if line_str or current_block: # Prevent leading blank lines
                current_block.append(line.rstrip())
                
    if current_block:
        chunks.append(Document(
            page_content="\n".join(current_block),
            metadata={
                "section": current_section,
                "start_line": start_line,
                "end_line": len(lines),
                "source": f"{current_section} (Lines {start_line}-{len(lines)})"
            }
        ))
        
    return chunks

# =====================================================================
# 3. KNOWLEDGE RETRIEVAL ORCHESTRATOR
# =====================================================================
@st.cache_resource
def get_faiss_vector_store(chunks, provider, api_key):
    try:
        from langchain_community.vectorstores import FAISS
        if provider == "OpenAI":
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
        elif provider == "Gemini":
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            embeddings = GoogleGenerativeAIEmbeddings(google_api_key=api_key, model="models/text-embedding-004")
        else:
            return None
            
        return FAISS.from_documents(chunks, embeddings)
    except Exception as e:
        st.sidebar.warning(f"Failed to initialize FAISS: {str(e)}. Falling back to local Keyword search.")
        return None

# =====================================================================
# 4. CHAT LOGIC AND RAG CHAIN IMPLEMENTATION
# =====================================================================
def rephrase_query(provider, api_key, query, chat_history):
    if not chat_history:
        return query
        
    history_str = ""
    for msg in chat_history[-4:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
        
    rephrase_prompt = (
        "Given the following chat history and a follow-up question, rephrase the follow-up question "
        "to be a standalone question that can be searched in an FAQ database. "
        "Do not answer the question, just return the rephrased question. "
        "If the question is already standalone, return the original question exactly.\n\n"
        f"Chat History:\n{history_str}\n"
        f"Follow-up Question: {query}\n"
        "Standalone Question:"
    )
    
    if provider == "Demo / Mock Mode" or not api_key:
        pronouns = ["it", "there", "they", "them", "cost", "price", "fee", "how much", "when"]
        if any(w in query.lower() for w in pronouns) and len(chat_history) >= 2:
            last_user_query = chat_history[-2]["content"]
            keywords = [w for w in tokenize(last_user_query) if w not in ["do", "you", "ship", "to", "how", "much", "is", "the", "a", "an", "what", "are", "do"]]
            if keywords:
                return f"{query} (regarding: {' '.join(keywords[:3])})"
        return query
        
    try:
        if provider == "OpenAI":
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini", temperature=0)
        elif provider == "Gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI
            llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash", temperature=0)
        elif provider == "Anthropic":
            from langchain_anthropic import ChatAnthropic
            llm = ChatAnthropic(anthropic_api_key=api_key, model="claude-3-5-haiku-20241022", temperature=0)
        else:
            return query
            
        response = llm.invoke([("system", "You are a helpful query rephraser."), ("human", rephrase_prompt)])
        return response.content.strip()
    except Exception as e:
        return query

def generate_llm_response(provider, api_key, query, chat_history, context_docs, persona, temperature, max_tokens):
    context_str = ""
    for idx, doc in enumerate(context_docs):
        context_str += f"--- Document {idx+1} ({doc.metadata['source']}) ---\n{doc.page_content}\n\n"
        
    history_str = ""
    for msg in chat_history[-6:]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"
        
    # Persona mapping
    personas = {
        "Professional & Direct": (
            "You are a helpful, professional, and direct customer support assistant for GigaCorp. "
            "Provide accurate answers with clear structuring, bullet points if helpful, and keep standard business etiquette."
        ),
        "Warm & Supportive": (
            "You are a friendly, empathetic, and warm customer support representative for GigaCorp. "
            "Express understanding, be welcoming, and show a genuine desire to assist the user while maintaining accuracy."
        ),
        "Ultra-Concise": (
            "You are a highly efficient, direct customer support bot for GigaCorp. "
            "Provide absolute minimal answers. Focus exclusively on the facts and numbers. Avoid conversational pleasantries."
        )
    }
    
    system_prompt = (
        f"{personas.get(persona, personas['Professional & Direct'])}\n\n"
        "Your task is to answer the user's question based strictly on the retrieved context documents provided below.\n"
        "If the answer cannot be found in the context documents, say: 'I apologize, but I don't have information about that in my knowledge base. Please contact our support team at support@gigacorp.com for assistance.' Do not make up any information.\n\n"
        "CRITICAL REQUIREMENT:\n"
        "You must cite the source line numbers in your response for any facts you state. Use the format '[Lines X-Y]' (for example, '[Lines 10-13]'). Place the citation right after the sentence or point where the information is used, not just at the end of the paragraph. Keep your citations precise.\n\n"
        f"Context Documents:\n{context_str}\n"
    )
    
    user_prompt = f"Chat History:\n{history_str}\nUser: {query}\nAssistant:"
    
    if provider == "OpenAI":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(openai_api_key=api_key, model="gpt-4o-mini", temperature=temperature, max_tokens=max_tokens)
    elif provider == "Gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(google_api_key=api_key, model="gemini-1.5-flash", temperature=temperature, max_output_tokens=max_tokens)
    elif provider == "Anthropic":
        from langchain_anthropic import ChatAnthropic
        llm = ChatAnthropic(anthropic_api_key=api_key, model="claude-3-5-haiku-20241022", temperature=temperature, max_tokens=max_tokens)
    else:
        return "Invalid provider specified."
        
    response = llm.invoke([
        ("system", system_prompt),
        ("human", user_prompt)
    ])
    return response.content

def generate_mock_response(query, context_docs):
    if not context_docs:
        return (
            "I apologize, but I don't have information about that in my knowledge base. "
            "Please contact our support team at support@gigacorp.com or call +1-800-555-GIGA (4442) during business hours (Mon-Fri, 8 AM - 8 PM EST) for assistance."
        )
        
    response = "**[Offline Demo Mode - Simulating Answer from FAQ Data]**\n\n"
    
    for doc in context_docs:
        lines_info = f"[Lines {doc.metadata['start_line']}-{doc.metadata['end_line']}]"
        lines = doc.page_content.split('\n')
        question_line = ""
        answer_lines = []
        for line in lines:
            if "Q:" in line:
                question_line = line.split("Q:", 1)[1].strip()
            elif "Answer:" in line:
                answer_lines.append(line.split("Answer:", 1)[1].strip())
            elif line.strip() and not line.startswith("===") and not line.startswith("Document Code") and not line.startswith("Last Updated"):
                answer_lines.append(line.strip())
                
        answer_text = " ".join(answer_lines)
        if question_line:
            response += f"Regarding **\"{question_line}\"**:\n"
        response += f"{answer_text} {lines_info}\n\n"
        
    response += "*(Note: In live mode with an API key, this is summarized dynamically by the LLM.)*"
    return response

# =====================================================================
# 5. UI STYLING & CUSTOM CSS
# =====================================================================
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap" rel="stylesheet">

<style>
    /* Premium font style settings */
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }

    /* Gradient page banner */
    .app-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #1e3a8a 100%);
        padding: 2.2rem;
        border-radius: 16px;
        color: white;
        text-align: center;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.08);
        position: relative;
        overflow: hidden;
    }
    .app-header::after {
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at top right, rgba(59, 130, 246, 0.15), transparent 60%);
        pointer-events: none;
    }
    .app-header h1 {
        margin: 0;
        font-size: 2.3rem;
        font-weight: 700;
        letter-spacing: -0.8px;
    }
    .app-header p {
        margin: 8px 0 0 0;
        font-size: 1rem;
        opacity: 0.85;
        font-weight: 400;
        letter-spacing: 0.2px;
    }
    
    /* Sleek citations */
    .citation-container {
        margin-top: 10px;
        background-color: #f8fafc;
        border-left: 3px solid #3b82f6;
        border-radius: 8px;
        padding: 14px 18px;
        font-size: 0.88rem;
        color: #1e293b;
        box-shadow: 0 1px 3px rgba(0,0,0,0.02);
        border: 1px solid #f1f5f9;
        border-top-right-radius: 8px;
        border-bottom-right-radius: 8px;
    }
    .citation-title {
        font-weight: 600;
        color: #1d4ed8;
        font-size: 0.9rem;
        margin-bottom: 6px;
        letter-spacing: -0.1px;
    }
    .citation-snippet {
        font-family: 'SFMono-Regular', Consolas, "Liberation Mono", Menlo, monospace;
        background-color: #f1f5f9;
        padding: 8px 12px;
        border-radius: 5px;
        font-size: 0.82rem;
        color: #334155;
        display: block;
        margin-top: 6px;
        border: 1px solid #e2e8f0;
        white-space: pre-wrap;
        line-height: 1.4;
    }
    
    /* Dynamic confidence score tags */
    .score-badge {
        display: inline-flex;
        align-items: center;
        padding: 3px 10px;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-top: 8px;
        letter-spacing: 0.3px;
    }
    .score-high {
        background-color: #dcfce7;
        color: #14532d;
        border: 1px solid #bbf7d0;
    }
    .score-medium {
        background-color: #fef9c3;
        color: #713f12;
        border: 1px solid #fef08a;
    }
    .score-low {
        background-color: #fee2e2;
        color: #7f1d1d;
        border: 1px solid #fecaca;
    }
    
    /* Dark mode responsive styling overrides */
    @media (prefers-color-scheme: dark) {
        .citation-container {
            background-color: #1e293b;
            border-left: 3px solid #60a5fa;
            color: #f1f5f9;
            border: 1px solid #334155;
        }
        .citation-title {
            color: #93c5fd;
        }
        .citation-snippet {
            background-color: #0f172a;
            color: #cbd5e1;
            border: 1px solid #334155;
        }
        .score-high {
            background-color: rgba(20, 83, 45, 0.3);
            color: #4ade80;
            border: 1px solid rgba(74, 222, 128, 0.4);
        }
        .score-medium {
            background-color: rgba(113, 63, 18, 0.3);
            color: #facc15;
            border: 1px solid rgba(250, 204, 21, 0.4);
        }
        .score-low {
            background-color: rgba(127, 29, 29, 0.3);
            color: #f87171;
            border: 1px solid rgba(248, 113, 113, 0.4);
        }
    }
    
    /* Suggestion cards styling */
    .stButton>button {
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        border-radius: 8px;
        font-weight: 500;
        letter-spacing: -0.1px;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Main Application Banner
st.markdown("""
<div class="app-header">
    <div style="display:flex; align-items:center; justify-content:center; gap:12px; margin-bottom:8px;">
        <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
            <rect x="2" y="3" width="20" height="14" rx="2" ry="2"></rect>
            <line x1="8" y1="21" x2="16" y2="21"></line>
            <line x1="12" y1="17" x2="12" y2="21"></line>
        </svg>
        <h1 style="margin:0; font-size:2.2rem; font-weight:700;">GigaCorp Support</h1>
    </div>
    <p style="margin:0; font-size:1rem; opacity:0.9; font-weight:400; letter-spacing:0.5px;">Precision Search & Conversational Assistant</p>
</div>
""", unsafe_allow_html=True)

# Initialize Session States
if "messages" not in st.session_state:
    st.session_state.messages = []
if "input_value" not in st.session_state:
    st.session_state.input_value = ""

# Load FAQ Documents
faq_chunks = load_and_parse_faq()

# Create local term search index
local_keyword_retriever = PurePythonVectorStore(faq_chunks)

# =====================================================================
# 6. SIDEBAR CONFIGURATION (Saas Dashboard controls)
# =====================================================================
st.sidebar.markdown("""
<div style="display:flex; align-items:center; gap:10px; margin-bottom:15px; margin-top:10px;">
    <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#3b82f6" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="3"></circle>
        <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 1 1 2.83-2.83l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 1 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
    </svg>
    <h3 style="margin:0; font-size:1.2rem; font-weight:600;">Control Panel</h3>
</div>
""", unsafe_allow_html=True)

# 6a. Live Metrics Widgets
col_m1, col_m2 = st.sidebar.columns(2)
col_m1.metric("FAQ Blocks", len(faq_chunks))
col_m2.metric("History Turns", len(st.session_state.messages) // 2)

st.sidebar.divider()

# 6b. LLM Setup
provider = st.sidebar.selectbox(
    "Select LLM Provider",
    ["Demo / Mock Mode", "OpenAI", "Gemini", "Anthropic"],
    help="Demo Mode is offline and free. Other models require direct API keys."
)

api_key = ""
if provider != "Demo / Mock Mode":
    key_labels = {
        "OpenAI": "OpenAI API Key (sk-...)",
        "Gemini": "Google Gemini API Key (AIzaSy...)",
        "Anthropic": "Anthropic API Key (sk-ant-...)"
    }
    api_key = st.sidebar.text_input(
        key_labels.get(provider, "API Key"),
        type="password",
        help=f"Enter your private {provider} key. This is never saved to disk."
    )

# 6c. Advanced Parameters
with st.sidebar.expander("Advanced Settings"):
    persona = st.selectbox(
        "Assistant Persona",
        ["Professional & Direct", "Warm & Supportive", "Ultra-Concise"],
        index=0,
        help="Customizes the system instructions and behavior style of the model."
    )
    
    retrieval_method = st.radio(
        "Retrieval Engine",
        ["Semantic Search (FAISS)", "Keyword Search (TF-IDF Local)"],
        index=0 if provider in ["OpenAI", "Gemini"] else 1
    )
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.1,
        step=0.05,
        help="Filter out retrieved FAQ matches below this relevance score."
    )
    
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Higher values make responses more creative but potentially less exact."
    )
    
    max_tokens = st.slider(
        "Max Output Length",
        min_value=100,
        max_value=1000,
        value=400,
        step=50,
        help="Maximum length of the assistant response."
    )

# Enforce local search override for Demo Mode
if provider == "Demo / Mock Mode":
    retrieval_method = "Keyword Search (TF-IDF Local)"
    st.sidebar.caption("Running in Offline Demo Mode. Local Keyword search enabled.")
elif provider == "Anthropic" and retrieval_method == "Semantic Search (FAISS)":
    st.sidebar.caption("Anthropic uses OpenAI embeddings for semantic search. If unavailable, local keyword is recommended.")

# Initialize FAISS if needed
vector_store = None
if retrieval_method == "Semantic Search (FAISS)" and api_key:
    with st.sidebar.spinner("Indexing Knowledge Base..."):
        vector_store = get_faiss_vector_store(faq_chunks, provider, api_key)
        if vector_store:
            st.sidebar.success("FAISS Vector Index ready!")
        else:
            retrieval_method = "Keyword Search (TF-IDF Local)"

# 6d. Expandable Database Viewer
with st.sidebar.expander("View FAQ Database"):
    if os.path.exists("gigacorp_faq.txt"):
        with open("gigacorp_faq.txt", "r", encoding="utf-8") as f:
            st.text(f.read())
    else:
        st.error("FAQ file missing.")

# 6e. Session Actions
st.sidebar.divider()

# Download chat history log
chat_log = ""
for msg in st.session_state.messages:
    role_label = "User" if msg["role"] == "user" else "Assistant"
    chat_log += f"**{role_label}**:\n{msg['content']}\n\n"
    if "sources" in msg and msg["sources"]:
        chat_log += "*Sources Cited:*\n"
        for src in msg["sources"]:
            meta = src.get("metadata", {})
            chat_log += f"- {meta.get('section', 'Source')} (Lines {meta.get('start_line', '?')}-{meta.get('end_line', '?')})\n"
        chat_log += "\n"

st.sidebar.download_button(
    label="Download Chat Log",
    data=chat_log,
    file_name="gigacorp_chat_transcript.md",
    mime="text/markdown",
    disabled=len(st.session_state.messages) == 0,
    use_container_width=True
)

if st.sidebar.button("Clear Chat History", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

st.sidebar.info(
    "**Tips to try:**\n"
    "1. *Do you ship to India?*\n"
    "2. *How much to ship there?* (Tests memory context)\n"
    "3. *What does Premium support cost?*\n"
    "4. *How do I do a domestic return?*"
)

# =====================================================================
# 7. MAIN CHAT INTERFACE & RAG PIPELINE
# =====================================================================

# Show existing message history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        
        # Display confidence and sources if assistant message
        if msg["role"] == "assistant":
            if "confidence" in msg and msg["confidence"] is not None:
                score = msg["confidence"]
                if score >= 0.7:
                    badge_class, badge_lbl = "score-high", "High Confidence"
                elif score >= 0.4:
                    badge_class, badge_lbl = "score-medium", "Medium Confidence"
                else:
                    badge_class, badge_lbl = "score-low", "Low Confidence"
                    
                st.markdown(f'<div class="score-badge {badge_class}">{badge_lbl}: {int(score * 100)}%</div>', unsafe_allow_html=True)
                
            if "sources" in msg and msg["sources"]:
                with st.expander("Citations & Verified Sources"):
                    for src in msg["sources"]:
                        st.markdown(f"""
                        <div class="citation-container">
                            <div class="citation-title">{src['metadata'].get('section', 'Source')} (Lines {src['metadata'].get('start_line', '?')}-{src['metadata'].get('end_line', '?')})</div>
                            <div class="citation-snippet">{src['page_content']}</div>
                        </div>
                        """, unsafe_allow_html=True)

# Suggestions Quick Links (Display only if history is empty)
if not st.session_state.messages:
    st.markdown("### How can I help you today?")
    st.write("Select a topic below to interact with the RAG database:")
    col1, col2 = st.columns(2)
    suggestions = [
        "Do you ship to India?",
        "What are GigaCorp's service tiers?",
        "How do I initiate a return?",
        "Are you open on weekends?"
    ]
    
    with col1:
        if st.button("Shipping to India?", use_container_width=True):
            st.session_state.input_value = suggestions[0]
            st.rerun()
        if st.button("How to make a return?", use_container_width=True):
            st.session_state.input_value = suggestions[2]
            st.rerun()
            
    with col2:
        if st.button("What are the Service Tiers?", use_container_width=True):
            st.session_state.input_value = suggestions[1]
            st.rerun()
        if st.button("Are you open on weekends?", use_container_width=True):
            st.session_state.input_value = suggestions[3]
            st.rerun()

# Setup input chat bar
user_query = st.chat_input("Ask a question about GigaCorp...")

# Override query if suggestion button clicked
if st.session_state.input_value:
    user_query = st.session_state.input_value
    st.session_state.input_value = "" # Clear suggestion buffer

# Handle live query submission
if user_query:
    # 1. Display user query
    with st.chat_message("user"):
        st.markdown(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # 2. Process query in RAG system
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        
        # UI Spinner while generating
        with st.spinner("Searching GigaCorp Knowledge Base..."):
            # Context-Aware Rephrasing for history queries
            rephrased = rephrase_query(provider, api_key, user_query, st.session_state.messages[:-1])
            if rephrased != user_query:
                st.caption(f"Rephrased query for search: \"{rephrased}\"")
                
            # Perform RAG retrieval
            retrieved_docs_with_scores = []
            retrieved_docs = []
            max_similarity = 0.0
            
            if retrieval_method == "Semantic Search (FAISS)" and vector_store:
                # FAISS similarity search returns docs and distances.
                # For FAISS, lower distance means higher similarity.
                # Let's map L2 distances to normalized confidence scores
                raw_matches = vector_store.similarity_search_with_relevance_scores(rephrased, k=3)
                for doc, score in raw_matches:
                    if score >= confidence_threshold:
                        retrieved_docs_with_scores.append((doc, score))
                        retrieved_docs.append(doc)
                if retrieved_docs_with_scores:
                    max_similarity = max(score for doc, score in retrieved_docs_with_scores)
            else:
                # Local Keyword Search return documents with custom scores
                raw_matches = local_keyword_retriever.similarity_search_with_score(rephrased, k=3)
                for doc, score in raw_matches:
                    if score >= confidence_threshold:
                        retrieved_docs_with_scores.append((doc, score))
                        retrieved_docs.append(doc)
                if retrieved_docs_with_scores:
                    max_similarity = max(score for doc, score in retrieved_docs_with_scores)
            
            # Generate Response (Live LLM or Mock Demo)
            if provider == "Demo / Mock Mode":
                assistant_response = generate_mock_response(rephrased, retrieved_docs)
            elif not api_key:
                assistant_response = (
                    "API Key Required. Please enter a valid API key for "
                    f"{provider} in the sidebar to enable live summaries. "
                    "Alternatively, select Demo / Mock Mode to test the system offline."
                )
                retrieved_docs = [] # Suppress citations
                max_similarity = 0.0
            else:
                try:
                    assistant_response = generate_llm_response(
                        provider, api_key, rephrased, st.session_state.messages[:-1], retrieved_docs,
                        persona, temperature, max_tokens
                    )
                except Exception as e:
                    assistant_response = f"Error generating response from {provider} API: {str(e)}"
                    retrieved_docs = []
                    max_similarity = 0.0
            
            # Show output response
            response_placeholder.markdown(assistant_response)
            
            # Render confidence badge
            if retrieved_docs:
                if max_similarity >= 0.7:
                    badge_class, badge_lbl = "score-high", "High Confidence"
                elif max_similarity >= 0.4:
                    badge_class, badge_lbl = "score-medium", "Medium Confidence"
                else:
                    badge_class, badge_lbl = "score-low", "Low Confidence"
                
                st.markdown(f'<div class="score-badge {badge_class}">{badge_lbl}: {int(max_similarity * 100)}%</div>', unsafe_allow_html=True)
            
            # Render citations
            if retrieved_docs:
                with st.expander("Citations & Verified Sources"):
                    for doc in retrieved_docs:
                        st.markdown(f"""
                        <div class="citation-container">
                            <div class="citation-title">{doc.metadata.get('section', 'Source')} (Lines {doc.metadata.get('start_line', '?')}-{doc.metadata.get('end_line', '?')})</div>
                            <div class="citation-snippet">{doc.page_content}</div>
                        </div>
                        """, unsafe_allow_html=True)
                        
        # Append answer to chat history
        history_entry = {
            "role": "assistant",
            "content": assistant_response,
            "confidence": max_similarity if retrieved_docs else None,
            "sources": [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in retrieved_docs]
        }
        st.session_state.messages.append(history_entry)
        
        # Trigger page rerun to maintain clean input bar
        st.rerun()

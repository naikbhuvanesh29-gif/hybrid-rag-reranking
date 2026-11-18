import streamlit as st
from pypdf import PdfReader
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from flashrank import Ranker, RerankRequest
from google import genai

# --- PAGE SETUP ---
st.set_page_config(page_title="Hybrid RAG + Reranking", page_icon="🔍", layout="wide")
st.title("🔍 Hybrid RAG + Reranking")
st.caption("Combine Keyword + Vector Search • Rerank Top Results • Answer with Citations")

# --- SIDEBAR: API KEY & PDF UPLOAD ---
with st.sidebar:
    st.header("⚙️ Setup")
    api_key = st.text_input("1. Enter Gemini API Key:", type="password")
    st.markdown("[Get a free Gemini API Key here](https://aistudio.google.com/app/apikey)")
    uploaded_file = st.file_uploader("2. Upload a PDF Document", type="pdf")

# --- LOAD AI MODELS (Cached so they only load once) ---
@st.cache_resource
def load_models():
    # Vector Search Model (turns text into numbers by meaning)
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    # Reranker Model (scores and re-orders the best results)
    reranker = Ranker(model_name="ms-marco-MiniLM-L-12-v2")
    return embedder, reranker

embedder, reranker = load_models()

# --- STEP 1: READ & CHOP PDF INTO CHUNKS ---
@st.cache_data
def process_pdf(file_bytes):
    reader = PdfReader(file_bytes)
    chunks = []
    
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if not text:
            continue
        
        # Split each page into ~120-word chunks so citations are super specific
        words = text.split()
        chunk_size = 120
        for i in range(0, len(words), chunk_size):
            chunk_text = " ".join(words[i:i + chunk_size])
            if len(chunk_text.strip()) > 30:
                chunks.append({
                    "id": len(chunks),
                    "text": chunk_text,
                    "page": page_num + 1
                })
    return chunks

# --- MAIN APP LOGIC ---
if uploaded_file:
    chunks = process_pdf(uploaded_file)
    st.success(f"✅ Loaded PDF! Chopped into **{len(chunks)}** searchable paragraphs across all pages.")

    # Build Keyword Search Index (BM25)
    tokenized_corpus = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_corpus)

    # Build Vector Search Index (FAISS)
    embeddings = embedder.encode([c["text"] for c in chunks])
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(embeddings).astype('float32'))

    # --- USER QUESTION INPUT ---
    query = st.text_input("💬 Ask a question about your PDF:", placeholder="e.g., What is the main conclusion on page 2?")

    if query:
        # 1. KEYWORD SEARCH (Exact Words)
        bm25_scores = bm25.get_scores(query.lower().split())
        top_keyword_idx = np.argsort(bm25_scores)[::-1][:5]

        # 2. VECTOR SEARCH (Meaning/Vibe)
        query_embedding = embedder.encode([query]).astype('float32')
        _, top_vector_idx = faiss_index.search(query_embedding, min(5, len(chunks)))
        top_vector_idx = top_vector_idx[0]

        # COMBINE BOTH (Hybrid Search)
        combined_ids = list(set(top_keyword_idx) | set(top_vector_idx))
        hybrid_Passages = [chunks[i] for i in combined_ids if i < len(chunks)]

        # 3. RERANK THE TOP RESULTS
        rerank_request = RerankRequest(query=query, passages=hybrid_Passages)
        reranked_results = reranker.rerank(rerank_request)[:3] # Keep Top 3 Best!

        # --- DISPLAY HOW IT WORKED ---
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🥇 Step 1 & 2: Reranked Top Results")
            for rank, res in enumerate(reranked_results, 1):
                with st.expander(f"Rank #{rank} — Page {res['page']} (Score: {res['score']:.2f})", expanded=(rank==1)):
                    st.write(res["text"])

        with col2:
            st.subheader("🤖 Step 3: Answer with Citations")
            if not api_key:
                st.warning("👈 Please paste your free Gemini API Key in the left sidebar to generate the AI answer!")
            else:
                client = genai.Client(api_key=api_key)
                
                # Format the top 3 passages with page numbers for the AI
                context_text = "\n\n".join(
                    [f"Source [Page {r['page']}]: {r['text']}" for r in reranked_results]
                )
                
                prompt = f"""You are a helpful assistant. Answer the user's question using ONLY the sources below.
Always cite the exact page number at the end of every sentence like this: [Page X].
If the answer is not in the sources, say you cannot find it in the document.

Sources:
{context_text}

Question: {query}"""

                with st.spinner("Writing answer with citations..."):
                    response = client.models.generate_content(
                        model="gemini-3.8-flash",
                        contents=prompt
                    )
                    st.markdown(response.text)
else:
    st.info("👈 Upload any PDF in the left sidebar to start testing your Hybrid RAG + Reranking system!")

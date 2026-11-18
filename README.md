# 🔍 Hybrid RAG + Reranking

A document-based Retrieval-Augmented Generation (RAG) application that combines keyword search and vector search with reranking to improve document retrieval and response relevance.

## 🚀 Features

- **PDF Upload:** Upload PDF documents for processing.
- **Hybrid Search:** Combines BM25 keyword search and FAISS vector search.
- **Text Embeddings:** Uses Sentence Transformers to represent document content.
- **Reranking:** Uses FlashRank to improve the ranking of retrieved passages.
- **AI-Powered Answers:** Uses Google Gemini to generate responses.
- **Citations:** Displays source references to support generated answers.
- **Streamlit Interface:** Provides a simple interface for interacting with documents.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | User interface |
| PyPDF | PDF text extraction |
| BM25 | Keyword-based retrieval |
| FAISS | Vector similarity search |
| Sentence Transformers | Text embeddings |
| FlashRank | Reranking |
| Google Gemini | Answer generation |
| NumPy | Numerical operations |

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/naikbhuvanesh29-gif/hybrid-rag-reranking.git
```

### 2. Navigate to the project folder

```bash
cd hybrid-rag-reranking
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

### 5. Configure Gemini API

Enter your Gemini API key in the application's setup section to enable AI-powered responses.

## 🔄 How It Works

1. Upload a PDF document.
2. Extract and split document text into chunks.
3. Generate embeddings for the document chunks.
4. Retrieve relevant passages using keyword and vector search.
5. Rerank retrieved passages using FlashRank.
6. Generate an answer using Google Gemini.
7. Present the answer with source references.

## 📁 Project Structure

```text
hybrid-rag-reranking/
│
├── app.py
├── requirements.txt
└── README.md
```

## 🎯 Project Objective

The objective of this project is to improve the relevance of document retrieval by combining multiple search techniques and reranking, helping an LLM generate more relevant, context-based answers.

---

**Developed by:** Bhuvanesh M Naik

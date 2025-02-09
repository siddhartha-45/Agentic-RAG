import os
import PyPDF2
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
import torch
import psutil
import re
from googleapiclient.discovery import build
import requests
from bs4 import BeautifulSoup

# Ensure the temp directory exists for storing PDFs
os.makedirs("temp", exist_ok=True)

# Embedding models to choose from
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": "sentence-transformers/all-MiniLM-L6-v2",
    "all-mpnet-base-v2": "sentence-transformers/all-mpnet-base-v2",
    "openai-ada-002": "openai/ada-002",
    "jina-embeddings-v2": "jinaai/jina-embeddings-v2-base-en"
}
DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# API Key and CSE ID (Replace these with your actual key and ID)
google_api_key = "AIzaSyCyRXtj-ejPqupaRVr6BDXEiHun9tUponI"
google_cse_id = "94af5f14204ee4b88"

# Function to log memory usage
def log_memory_usage():
    process = psutil.Process(os.getpid())
    print(f"Memory usage: {process.memory_info().rss / (1024 * 1024)} MB")

# Function to normalize scores
def normalize_scores(scores):
    scores = np.array(scores)
    return (scores - scores.min()) / (scores.max() - scores.min())

# Function to evaluate retrieval performance
def evaluate_retrieval(query, texts, retrieved_docs, model_name):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODELS[model_name], device=device)
    query_embedding = model.encode([query])
    doc_embeddings = model.encode(retrieved_docs)

    scores = [cosine_similarity([query_embedding[0]], [doc_embedding])[0][0] for doc_embedding in doc_embeddings]
    return np.mean(scores)

# Step 1: Extract text from PDFs
def extract_pdf_text(pdf_file, max_length=10000):
    text = ""
    reader = PyPDF2.PdfReader(pdf_file)
    for page in reader.pages:
        page_text = page.extract_text()
        clean_text = re.sub(r'[^a-zA-Z0-9.,;\s]', '', page_text)
        text += clean_text.strip()
        if len(text) > max_length:
            text = text[:max_length]
            break
    return text

# Extract text from multiple PDFs
def extract_text_from_pdfs(pdf_files):
    texts = []
    for pdf_file in pdf_files:
        text = extract_pdf_text(pdf_file)
        texts.append(text)
    return texts

# Step 2: Embed the text using SentenceTransformers
def embed_text(texts, model_name, chunk_size=512):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODELS[model_name], device=device)
    embeddings = []

    for text in texts:
        text_chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        chunk_embeddings = model.encode(text_chunks, show_progress_bar=True)
        embeddings.append(np.mean(chunk_embeddings, axis=0))

    embeddings = np.array(embeddings)
    # Ensure all embeddings have the same dimensions
    assert len(set(e.shape[0] for e in embeddings)) == 1, "Inconsistent embedding dimensions detected"
    return embeddings

# Step 3: Index the embeddings in FAISS
def index_embeddings(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    
    # Convert to float32 for FAISS compatibility
    embeddings = np.array(embeddings, dtype=np.float32)
    index.add(embeddings)
    assert index.is_trained, "FAISS index is not properly trained"
    return index

# Step 4: Search the index for relevant documents
def search_index(query, index, texts, model_name, top_k=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODELS[model_name], device=device)
    query_embedding = model.encode([query])
    
    # Convert to float32 and check dimensions
    query_embedding = np.array(query_embedding, dtype=np.float32)
    assert query_embedding.shape[1] == index.d, "Embedding dimensions do not match FAISS index dimensions"

    distances, indices = index.search(query_embedding, top_k)
    relevant_docs = [texts[i] for i in indices[0]]
    return relevant_docs, distances

# Cosine Similarity Search
def cosine_similarity_search(query, embeddings, texts, model_name, top_k=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer(EMBEDDING_MODELS[model_name], device=device)
    query_embedding = model.encode([query])
    similarities = cosine_similarity(query_embedding, embeddings)
    ranked_indices = np.argsort(similarities[0])[::-1][:top_k]
    relevant_docs = [texts[i] for i in ranked_indices]
    scores = similarities[0][ranked_indices]
    return relevant_docs, scores

def hybrid_retrieval(query, index, embeddings, texts, model_name, top_k=3):
    faiss_docs, faiss_distances = search_index(query, index, texts, model_name, top_k=top_k)
    cosine_docs, cosine_scores = cosine_similarity_search(query, embeddings, texts, model_name, top_k=top_k)

    # Combine results (remove duplicates and prioritize by FAISS first)
    combined_docs = faiss_docs + [doc for doc in cosine_docs if doc not in faiss_docs]
    return combined_docs[:top_k]

# Reranker using CrossEncoder
def rerank_documents(query, documents, model_name="cross-encoder/ms-marco-MiniLM-L-6-v2"):
    reranker = CrossEncoder(model_name)
    query_doc_pairs = [(query, doc) for doc in documents]
    scores = reranker.predict(query_doc_pairs)
    ranked_indices = np.argsort(scores)[::-1]
    return [documents[i] for i in ranked_indices], scores

# Perform web search using Google Custom Search JSON API
def perform_web_search(query, num_results=3):
    service = build("customsearch", "v1", developerKey=google_api_key)
    res = service.cse().list(q=query, cx=google_cse_id, num=num_results).execute()
    search_results = [item['link'] for item in res.get('items', [])]
    return search_results

# Summarize content from a web page
def summarize_web_content(url, model_name="facebook/bart-large-cnn"):
    try:
        response = requests.get(url, timeout=10)
        soup = BeautifulSoup(response.content, 'html.parser')
        paragraphs = soup.find_all('p')
        content = ' '.join([p.get_text() for p in paragraphs])

        max_length = 1024
        content = content[:max_length]

        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        inputs = tokenizer(content, return_tensors="pt", truncation=True, max_length=1024)
        summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=50, max_length=200, early_stopping=True)
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        return summary
    except Exception as e:
        return f"Could not summarize content from {url}: {e}"

# Step 5: Generate a response using Hugging Face’s BART or T5 model
def generate_response(query, relevant_docs, model_name="facebook/bart-large-cnn"):
    context = "\n".join(relevant_docs)

    # Limiting the context size for the model (to avoid token overflow)
    max_input_length = 1024  # BART or T5 can handle up to ~1024 tokens

    # Truncate context if necessary
    if len(context.split()) > max_input_length:
        context = " ".join(context.split()[:max_input_length])

    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Build the prompt
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"

    # Tokenize input and generate response
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True, max_length=1024)
    summary_ids = model.generate(inputs["input_ids"], num_beams=4, min_length=150, max_length=500, early_stopping=True)

    answer = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return answer

# Streamlit UI
st.title("Research Paper RAG System")
st.sidebar.header("Upload PDFs")

# File uploader for PDFs
pdf_files = st.sidebar.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True)

# Select embedding models
embedding_model_1 = st.sidebar.selectbox("Primary Embedding Model", list(EMBEDDING_MODELS.keys()), index=0)
embedding_model_2 = st.sidebar.selectbox("Secondary Embedding Model", list(EMBEDDING_MODELS.keys()), index=1)
embedding_model_3 = st.sidebar.selectbox("Third Embedding Model", list(EMBEDDING_MODELS.keys()), index=2)

# Select retrieval strategy
retrieval_strategy = st.sidebar.radio("Select Retrieval Strategy", ["FAISS", "Cosine Similarity", "Hybrid", "Reranker"])

# Check if PDFs are uploaded
if pdf_files:
    with st.spinner("Extracting text from PDFs..."):
        texts = extract_text_from_pdfs(pdf_files)

    st.sidebar.success("PDFs uploaded and text extracted!")

    with st.spinner("Indexing documents..."):
        log_memory_usage()
        embeddings_1 = embed_text(texts, embedding_model_1)
        embeddings_2 = embed_text(texts, embedding_model_2)
        embeddings_3 = embed_text(texts, embedding_model_3)
        index = index_embeddings(embeddings_1)

    st.sidebar.success("Documents indexed successfully!")

    st.header("Query the RAG System")
    user_query = st.text_input("Enter your query")

    if user_query:
        with st.spinner("Retrieving relevant documents..."):
            if retrieval_strategy == "FAISS":
                relevant_docs_1, _ = search_index(user_query, index, texts, embedding_model_1)
            elif retrieval_strategy == "Cosine Similarity":
                relevant_docs_1 = cosine_similarity_search(user_query, embeddings_1, texts, embedding_model_1, top_k=3)[0]
            elif retrieval_strategy == "Hybrid":
                relevant_docs_1 = hybrid_retrieval(user_query, index, embeddings_1, texts, embedding_model_1)
            elif retrieval_strategy == "Reranker":
                initial_relevant_docs_1 = hybrid_retrieval(user_query, index, embeddings_1, texts, embedding_model_1)
                relevant_docs_1, _ = rerank_documents(user_query, initial_relevant_docs_1)

        # Web search triggered if no relevant documents found or if explicitly needed
        if all("http" not in doc for doc in relevant_docs_1):  # Check if all retrieved docs are not web-related
            st.success("Relevant documents retrieved!")
            # Display Top 3 Documents
            st.subheader("Top 3 Relevant Documents")
            for idx, doc in enumerate(relevant_docs_1[:3]):
                st.write(f"Document {idx + 1}: {doc[:500]}{'...' if len(doc) > 500 else ''}")
            
            with st.spinner("Generating response..."):
                response = generate_response(user_query, relevant_docs_1)
            st.subheader("Generated Response")
            st.write(response)
        else:  # If any document is a web link (irrelevant), perform web search
            st.warning("The retrieved documents are not highly relevant. Querying web search...")
            with st.spinner("Performing web search..."):
                web_results = perform_web_search(user_query, num_results=3)
                st.subheader("Web Search Summaries")
                web_summaries = []

                for url in web_results:
                    summary = summarize_web_content(url)
                    web_summaries.append(summary)
                    st.write(f"Summary from {url}:")
                    st.write(summary)

                with st.spinner("Generating response from web search..."):
                    response = generate_response(user_query, web_summaries)

                st.subheader("Generated Response from Web Search")
                st.write(response)
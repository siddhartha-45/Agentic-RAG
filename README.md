# Agentic RAG System

## Objectives

The objective of this system is to develop an end-to-end Retrieval-Augmented Generation (RAG) solution that can:

1. **Extract and Process Data:**
   - Extract meaningful text from user-uploaded PDF research papers.
   - Convert the extracted text into machine-readable embeddings.

2. **Embed and Index:**
   - Embed text using cutting-edge models like Sentence Transformers.
   - Store embeddings efficiently using FAISS for fast similarity search.

3. **Retrieve and Rank:**
   - Search for documents relevant to a user’s query using advanced retrieval methods like hybrid (FAISS + Cosine) or CrossEncoder-based reranking.
   - Evaluate the relevance of retrieved documents.
   - If the query is not related to uploaded documents, perform a web search.

4. **Answer Generation:**
   - Use retrieved documents to synthesize answers to user queries using pre-trained models like BART.
   - Utilize web search results for user queries and generate responses based on those results.

5. **Model Comparison:**
   - Evaluate the performance of different embedding models (e.g., MiniLM, MPNet, Jina) and retrieval strategies to select the best-performing setup.

## Dataset

The following PDF files have been considered for the dataset:
- Mistral_paper.pdf
- Gemini_paper.pdf
- Gpt4.pdf
- Attention_paper.pdf
- Instructgpt.pdf

## Used Models, Packages, and Frameworks

### Models:
- **SentenceTransformers Models:** Used for generating text embeddings to represent the content semantically.
  - all-MiniLM-L6-v2
  - all-mpnet-base-v2
  - openai-ada-002
  - jina-embeddings-v2
- **CrossEncoder (ms-marco-MiniLM-L-6-v2):** Used for reranking retrieved documents based on query-document relevance.
- **Hugging Face Models (facebook/bart-large-cnn):** Summarization model for generating concise summaries of web content or combined context.

### Packages:
- **PyPDF2:** Extracts text from PDF files for further processing.
- **FAISS (Facebook AI Similarity Search):** Efficient indexing and nearest neighbor search for embedding-based retrieval.
- **SentenceTransformers:** Provides pre-trained models for embedding generation and text similarity.
- **Transformers:** Accesses models like BART for summarization and text generation.
- **Streamlit:** Framework for building an interactive web UI for the system.
- **Scikit-learn (cosine_similarity):** Computes similarity between embeddings for document relevance scoring.
- **Torch:** Framework for deep learning models used in embeddings and summarization.
- **psutil:** Monitors system memory usage during runtime.
- **Google API Client:** Performs web searches using Google Custom Search Engine (CSE).
- **Requests and BeautifulSoup:** Scrapes and parses content from web pages for summarization.

### Frameworks:
- **Streamlit Framework:** For creating a user-friendly interface for uploading PDFs, querying the system, and visualizing results.

## Workflow

1. **Streamlit User Interface:**
   - Users can upload multiple PDF documents for processing.
   - Select different embedding models for text embeddings to suit specific requirements.
   - Enter queries to retrieve relevant documents and generate responses.
   - View document extraction results, model performance metrics, and the final responses generated.

2. **PDF Text Extraction:**
   - The system extracts text from PDF documents, making them machine-readable and ready for further processing.

3. **Text Embedding:**
   - Transforms the extracted text into vector representations (embeddings) that encode semantic meaning.

4. **Indexing the Embeddings with FAISS:**
   - Embeddings are indexed using FAISS, facilitating fast, scalable similarity-based searches.

5. **Retrieving Relevant Documents:**
   - Identifies documents that are most contextually relevant to the user’s query.

6. **Hybrid Retrieval with Cosine Similarity:**
   - Combines FAISS-based indexing with direct cosine similarity evaluation to improve retrieval accuracy.

7. **Reranking with CrossEncoder:**
   - Re-evaluates and ranks the retrieved documents based on semantic relevance to the query.

8. **Web Search and Summarization (Fallback Option):**
   - Performs a web search to augment information when document retrieval does not provide sufficient context.

9. **Response Generation:**
   - Synthesizes the retrieved context into a coherent and accurate response using a pre-trained sequence-to-sequence model.

10. **RAG (Retrieval-Augmented Generation) System Overview:**
    - Integrates retrieval and generation phases to provide accurate, context-rich, and reliable answers.

## Outcome

In the sidebar of the interface, users can browse or upload files. After uploading the PDF files, they will select two embedding models (i.e., all-MiniLM-L6-v2, all-mpnet-base-v2) and retrieval strategies. The system will extract text from the PDFs, index the documents, and allow users to enter queries. If the query is relevant, the system will return the top 3 relevant documents and generate a response. If not, it will perform a web search and generate a response based on the web results. 


## Acknowledgments

- [Streamlit](https://streamlit.io/)
- [Hugging Face](https://huggingface.co/)
- [FAISS](https://faiss.ai/)
- [PyPDF2](https://pypi.org/project/PyPDF2/) 

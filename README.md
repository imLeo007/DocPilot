# DocPilot

A fast, lightweight, and memory-efficient Retrieval-Augmented Generation (RAG) engine designed for high-accuracy document search and evidence retrieval.

DocPilot processes uploaded documents, converts them into clean text, and splits them into well-balanced semantic chunks. For search, it combines dense vector search with sparse keyword search into a single pipeline. By replacing heavy machine learning frameworks with optimized ONNX Runtime CPU models, DocPilot maintains an ultra-low memory footprint (~300 MB active RAM) while achieving enterprise-level retrieval accuracy.

---

## Skill Stack

<p align="left">
  <img src="https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/FastAPI-009688?logo=fastapi&logoColor=white" alt="FastAPI" />
  <img src="https://img.shields.io/badge/PostgreSQL-4169E1?logo=postgresql&logoColor=white" alt="PostgreSQL" />
  <img src="https://img.shields.io/badge/pgvector-4169E1?logo=postgresql&logoColor=white" alt="pgvector" />
  <img src="https://img.shields.io/badge/SQLAlchemy-D71F00?logo=sqlalchemy&logoColor=white" alt="SQLAlchemy" />
  <img src="https://img.shields.io/badge/Alembic-6BA81E?logoColor=white" alt="Alembic" />
  <img src="https://img.shields.io/badge/Docker-2496ED?logo=docker&logoColor=white" alt="Docker" />
  <img src="https://img.shields.io/badge/ONNX_Runtime-005CED?logo=onnx&logoColor=white" alt="ONNX Runtime" />
  <img src="https://img.shields.io/badge/Hugging_Face-FFD21E?logo=huggingface&logoColor=black" alt="Hugging Face" />
  <img src="https://img.shields.io/badge/DeepEval-6842FF?logoColor=white" alt="DeepEval" />
</p>

---

## How It Works

DocPilot achieves high search accuracy and fast response times through a five-step pipeline:

### 1. Document Ingestion & Chunking
When a document is uploaded, its text is cleanly extracted and structured. Using the **Chonkie** chunking library, the text is divided into balanced semantic chunks (`chunk_size=400`, `chunk_overlap=40`). This ensures sentences are never abruptly cut in half and natural context is preserved across paragraph boundaries.


### 2. Lightweight ONNX Embeddings
Instead of loading massive PyTorch libraries into memory, DocPilot uses an ONNX-compiled Bi-Encoder running directly on the CPU. Built-in threading caps and disabled memory arenas prevent RAM spikes under high traffic. The raw outputs are mean-pooled and $L_2$-normalized into 384-dimensional vector embeddings.


### 3. Dual-Indexed Database Storage
Each chunk is saved into PostgreSQL with two dedicated indexes created side by side:
* **HNSW Vector Index (`pgvector`):** Enables ultra-fast approximate nearest neighbor vector search using cosine distance.
* **GIN Full-Text Index (`tsvector`):** Enables sub-millisecond keyword matching using language-aware word stemming.


### 4. Hybrid Search & Reciprocal Rank Fusion (RRF)
When a user asks a question, DocPilot launches two searches in parallel:
1. **Semantic Search:** Finds the top 25 chunks closest in meaning to the query vector.
2. **Lexical Search:** Finds the top 25 chunks matching keywords and numbers using disjunctive (`OR`) cover-density ranking.

Both candidate lists are merged using **Reciprocal Rank Fusion (RRF)**. RRF scores items by their ranking positions rather than raw values, ensuring fair blending without score calibration issues.


### 5. Cross-Encoder Reranking
The fused candidate pool is passed to an ONNX-optimized Cross-Encoder model. The Cross-Encoder reads the question and each chunk together, scoring direct contextual relevance. The candidates are sorted by their rerank scores and the top-ranked results are delivered to the user or downstream language model.


---

## Visual Overview

### Workflow Architecture
![DocPilot Pipeline](screenshots/workflow.png)
*The end-to-end flow from document upload to dual-index storage, hybrid retrieval, and final reranking.*

### Interactive API Preview
![Swagger UI](screenshots/preview.png)
*FastAPI interactive documentation interface for uploading documents and running searches.*

### Benchmark Evaluation
![Evaluation Results](screenshots/benchmark.png)
*Automated evaluation run using DeepEval measuring contextual retrieval performance.*

---

## Performance & Benchmark

DocPilot was evaluated using an automated benchmark suite powered by **DeepEval**. Testing was conducted on a curated 100-question evaluation dataset built against standard, publicly available educational textbook documents.

Rather than relying on manual guesses or rigid chunk IDs, an independent LLM judge evaluated the factual completeness and ordering of the retrieved contexts directly against known reference answers.

### Benchmark Results

| Metric | Measured Score | Evaluation Threshold | Status |
| :--- | :---: | :---: | :---: |
| **Contextual Recall** | **99.0%** | 70.0% | **PASSED** |
| **Contextual Precision** | **96.0%** | 70.0% | **PASSED** |
| **Test Cases Evaluated** | **100 Questions** | 100 Questions | **96 / 100 Clean Pass** |

* **Contextual Recall (99%):** Confirms that in 99 out of 100 queries, the engine successfully retrieved all necessary factual information within the top results.
* **Contextual Precision (95%):** Confirms that high-relevance chunks were consistently prioritized at the very top of the retrieved context.

### Runtime Efficiency

```mermaid
flowchart LR
    A[Traditional RAG: 2-4 GB RAM] -.->|85% Memory Reduction| B[DocPilot: ~300 MB Active RAM]
    C[Query Latency] --> D[Hybrid Search + ONNX Reranking: Under 250ms]
```

* **RAM Footprint:** Stays at approximately **~300 MB** during active querying due to CPU-optimized ONNX inference and thread locks.
* **Query Latency:** The entire hybrid search, RRF merge, and reranker pass executes in under **250 milliseconds**.

---

## Architectural Comparison: GroundForge vs. DocPilot

| Aspect | GroundForge (Prototype) | DocPilot (This project) |
| :--- | :--- | :--- |
| **Primary Goal** | Deep document parsing prototype | Ultra-lightweight agent retrieval sidecar |
| **PDF Extraction** | Docling (layout & structure aware) | PyMuPDF (compiled C-speed extraction) |
| **Model Runtime** | Native PyTorch (CPU) | Pure ONNX Runtime (Zero-Torch dependency) |
| **Memory Footprint** | ~1.8 GB – 2.4 GB RAM | **~280 MB – 320 MB RAM** |
| **Retrieval Speed** | Standard (~150ms – 250ms) | **Sub-50ms** |
| **Recommended For** | Layout-heavy documents, deep parsing research | Fast, low-memory tool for Agno & LangGraph |

## Project Structure

```
docpilot/
├── app/
│   ├── ai/
│   │   ├── embedding.py       # SentenceTransformer vector embeddings
│   │   ├── reranker.py        # CrossEncoder scoring model
│   │   └── retriever.py       # Hybrid search (pgvector + tsvector) & RRF
│   ├── api/
│   │   ├── document.py        # Document upload and parsing endpoints
│   │   └── question.py        # Question retrieval endpoints
│   ├── core/
│   │   ├── config.py          # Environment settings
│   │   └── database.py        # Database initialization
│   ├── models/
│   │   ├── chunks.py          # Dual-indexed chunk table (HNSW + GIN)
│   │   └── document.py        # Document metadata table
|   ├── services/
|   |   ├── chunker.py         # Chunk page text
|   |   ├── pdf.py             # Document fetching services
|   |   └── pdf.py             # Docling parser integration
│   └── main.py                # FastAPI entrypoint
├── evaluation/
│   ├── dataset.json           # 100-question ground truth dataset
│   ├── evaluator.py           # DeepEval precision and recall runner
│   └── judge.py               # Custom structured LLM judge
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

## How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/imLeo007/docpilot-rag.git
cd docpilot-rag
```

### 2. Configure Environment Variables
Copy the sample environment file and adjust your settings:
```bash
cp .env.example .env
```

Your `.env` file should include the following variables (referenced from `.env.example`):
```env
# Database Settings
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/docpilot_db
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=docpilot_db
POSTGRES_PORT=5432

# Model Configurations
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
EMBEDDING_FILENAME=model.onnx
RERANKER_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
RERANKER_FILENAME=model.onnx

# Evaluation / Judge API Settings (Optional for evaluation runs)
OPENAI_API_KEY=your_api_key_here
OPENAI_MODEL=gpt-4o-mini
```

### 3. Start Database & App with Docker
```bash
docker compose up --build -d
```

### 4. Run Database Migrations
Apply the database schema and indexes:
```bash
docker compose exec api alembic upgrade head
```

### 5. Access the API
Open your browser and navigate to the interactive Swagger UI:
```
http://localhost:8000/docs
```

* Use `POST /document/upload` to upload and automatically index your PDF files.
* Use `POST /question/search` to run hybrid retrieval and inspect the ranked chunk results.

### 6. Run the Evaluation Suite (Optional)
To verify the retrieval metrics on your own machine:
```bash
python -m evaluation.evaluator
```

---

## Conclusion

DocPilot demonstrates that a high-performance RAG pipeline does not require massive infrastructure or gigabytes of memory. By combining clean document chunking, dual vector-and-keyword indexing, rank fusion, and lightweight CPU cross-encoders, it delivers reliable and verifiable retrieval accuracy while remaining simple to deploy, maintain, and integrate into larger agent systems.
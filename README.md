# Multimodal Depression RAG

Multimodal Depression RAG built around the **unstructured** library for ingestion of PDFs, images, docs, HTML, and audio (via transcription), with a hybrid vector store (text + vision) and an **evaluation suite** (RAGAS + retrieval metrics).

> ⚕️ **Important**: This app is for information only. It is **not** a medical device and must **not** be used for diagnosis or treatment. Always consult a licensed professional for clinical decisions.

## Features
- **Any-file ingestion** using `unstructured.partition.auto.partition` (+ optional OCR) and Whisper transcription for audio.
- **Dual indexes**
  - `text_index` (Sentence-Transformers text embeddings)
  - `vision_index` (CLIP embeddings for images; text queries can retrieve related images via cross-modal CLIP)
- **SQLite facts store** for numerics (e.g., PHQ-9 scores, prevalence rates) extracted during ingest.
- **Retriever** that merges text + image hits, deduplicates, and reranks.
- **FastAPI** endpoints: `/ingest` (files), `/ask` (QA), `/reset`, `/health`.
- **Evaluation** (`eval.py`) with:
  - RAGAS metrics: faithfulness, answer relevancy, context precision/recall (requires an LLM-as-judge; OpenAI or Bedrock via LangChain)
  - Retrieval metrics: Recall@K, MRR, latency breakdowns

## Folder layout
```
.
├─ app.py                  # FastAPI server (ingest + RAG)
├─ pipeline.py             # Orchestration: ingest, retrieve, generate
├─ embedder.py             # Text + CLIP embedders
├─ vectorstore.py          # Chroma wrappers for text + images
├─ extractors.py           # Unstructured + audio transcription + numeric extraction
├─ config.py               # All configuration and model choices
├─ eval.py                 # Full evaluation suite (RAGAS + retrieval metrics)
├─ requirements.txt
├─ env.example
└─ README.md
```

## Quickstart
1) **Python 3.10+** recommended. On macOS/Linux:
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp env.example .env  # and fill keys if you have them
```
2) (Optional) Install OCR deps for images/PDFs:
- macOS: `brew install tesseract`
- Linux: `sudo apt-get install tesseract-ocr`

3) Run the API:
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```
4) Ingest files:
```bash
curl -F "files=@/path/to/paper.pdf" -F "files=@/path/to/figure.png" http://localhost:8000/ingest
```
5) Ask a question:
```bash
curl -X POST http://localhost:8000/ask -H 'Content-Type: application/json' \
  -d '{"query":"What is PHQ-9 and how is it scored? Include any images if relevant."}'
```
6) Run eval (see `data/eval_questions.jsonl` format docs inside `eval.py`):
```bash
python eval.py --questions data/eval_questions.jsonl --k 5
```

## API Endpoints

### POST /ingest
Upload files for processing and indexing.

**Request**: Multipart form data with `files` field
**Response**: List of successfully ingested filenames

### POST /ask
Ask questions about the ingested content.

**Request**: JSON with `query` field (optional `k` for number of results)
**Response**: JSON with `answer` and `contexts` fields

### POST /reset
Clear all ingested data and reset the system.

**Response**: Success confirmation

### GET /health
Health check endpoint.

**Response**: System status

## Configuration

Copy `env.example` to `.env` and configure:

- **PROVIDER**: LLM provider (`openai`, `bedrock`)
- **OPENAI_API_KEY**: Your OpenAI API key
- **TEXT_EMBEDDER**: Text embedding model
- **CLIP_EMBEDDER**: CLIP model for images
- **CHROMA_DIR**: Vector database directory
- **SQLITE_PATH**: Facts database path

## Evaluation

The evaluation suite supports both retrieval metrics and RAGAS LLM-based metrics:

```bash
python eval.py --questions data/eval_questions.jsonl --k 5
```

Results are saved to `reports/eval_results.csv` with comprehensive metrics.

## Architecture

1. **Ingestion**: Files processed by `unstructured` library with OCR and audio transcription
2. **Embedding**: Text and images embedded using Sentence-Transformers and CLIP
3. **Storage**: Dual ChromaDB collections for text and vision, SQLite for facts
4. **Retrieval**: Hybrid search across text and images with deduplication
5. **Generation**: LLM-powered answers with source citations
6. **Evaluation**: Comprehensive metrics for system performance

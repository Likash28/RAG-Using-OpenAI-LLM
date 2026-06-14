# RAG Using OpenAI LLM

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green?logo=fastapi&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4-412991?logo=openai&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)

A Retrieval-Augmented Generation (RAG) pipeline built with OpenAI and FastAPI. Supports multi-modal document ingestion (PDF, images, text), vector semantic search, and a clean glassmorphism UI for querying your knowledge base.

## Architecture

```
User Query
    │
    ▼
Embedder (OpenAI text-embedding)
    │
    ▼
VectorStore (semantic search)
    │
    ▼
Context Retrieval
    │
    ▼
OpenAI GPT (generation with context)
    │
    ▼
Response + Source Citations
```

## Features

- Multi-modal document support — PDF, images, plain text
- Semantic vector search via embeddings
- Streaming responses from GPT-4
- Sentiment analysis on queries and responses
- Crisis detection with safety guardrails
- Topic-based query filtering
- Clean glassmorphism frontend

## Tech Stack

| Layer | Tech |
|-------|------|
| Backend | FastAPI, Python |
| LLM | OpenAI GPT-4 |
| Embeddings | OpenAI text-embedding-ada-002 |
| Vector Store | FAISS / Chroma |
| Frontend | HTML, CSS (glassmorphism) |

## Getting Started

```bash
git clone https://github.com/Likash28/RAG-Using-OpenAI-LLM.git
cd RAG-Using-OpenAI-LLM
pip install -r requirements.txt
```

Set up your `.env`:
```env
OPENAI_API_KEY=your_key_here
```

Run the app:
```bash
uvicorn app:app --reload
```

Open `http://localhost:8000` in your browser.

## Project Structure

```
├── app.py           # FastAPI entrypoint
├── pipeline.py      # RAG pipeline logic
├── embedder.py      # Embedding generation
├── vectorstore.py   # Vector DB operations
├── config.py        # Config and env vars
└── requirements.txt
```

## Contributing

Pull requests are welcome. For major changes, open an issue first.

## License

[MIT](LICENSE)

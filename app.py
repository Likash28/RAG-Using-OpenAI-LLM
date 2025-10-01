from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
from pipeline import RAGPipeline
from config import settings
from logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger("RAGApplication")

app = FastAPI(title="Multimodal Depression RAG")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Note: LoggingMiddleware removed for now to avoid import complexity

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

logger.info("Initializing RAG Pipeline")
pipe = RAGPipeline()
logger.info("RAG Pipeline initialized successfully")

@app.get("/")
async def root():
    logger.info("Serving main page")
    return FileResponse("static/index.html")

@app.get("/health")
async def health():
    logger.info("Health check requested")
    return {"status": "ok"}

@app.post("/reset")
async def reset():
    logger.info("Resetting RAG pipeline")
    pipe.reset()
    logger.info("RAG pipeline reset completed")
    return {"ok": True}

@app.post("/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    logger.info(f"Starting file ingestion for {len(files)} files")
    saved = []
    os.makedirs("uploads", exist_ok=True)
    
    for f in files:
        logger.info(f"Processing file: {f.filename}")
        path = os.path.join("uploads", f.filename)
        with open(path, "wb") as out:
            shutil.copyfileobj(f.file, out)
        saved.append(path)
    
    try:
        logger.info("Starting RAG pipeline ingestion")
        pipe.ingest_paths(saved)
        logger.info(f"Successfully ingested {len(saved)} files")
    except Exception as e:
        logger.error(f"File ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    
    return {"ingested": [os.path.basename(x) for x in saved]}

@app.post("/ask")
async def ask(payload: dict):
    q = payload.get("query")
    k = int(payload.get("k", settings.top_k))
    
    if not q:
        logger.warning("Query request missing 'query' parameter")
        raise HTTPException(status_code=400, detail="Missing 'query'")
    
    logger.info(f"Processing query: {q[:100]}{'...' if len(q) > 100 else ''}")
    
    try:
        res = pipe.query(q, k)
        logger.info("Query processed successfully")
        return JSONResponse(res)
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

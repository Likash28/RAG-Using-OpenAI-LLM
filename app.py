from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# from pipeline import RAGPipeline  # Defer import to avoid startup issues
from config import settings
from logging_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger("RAGApplication")

app = FastAPI(
    title="Multimodal Depression RAG",
    description="A specialized RAG system for depression and mental health information",
    version="1.0.0"
)

# CORS middleware - Updated for production deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Add your frontend domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"]
)

# Note: LoggingMiddleware removed for now to avoid import complexity

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize RAG Pipeline lazily
pipe = None

def get_pipeline():
    global pipe
    if pipe is None:
        try:
            from pipeline import RAGPipeline
            logger.info("Initializing RAG Pipeline")
            pipe = RAGPipeline()
            logger.info("RAG Pipeline initialized successfully")
        except ValueError as e:
            if "API_KEY" in str(e) or "API key" in str(e):
                logger.error(f"Missing API key: {str(e)}")
                raise HTTPException(status_code=500, detail="API key not configured. Please set OPENAI_API_KEY environment variable.")
            else:
                logger.error(f"Configuration error: {str(e)}")
                raise HTTPException(status_code=500, detail=f"Configuration error: {str(e)}")
        except Exception as e:
            logger.error(f"Failed to initialize RAG Pipeline: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Pipeline initialization failed: {str(e)}")
    return pipe

@app.get("/")
async def root():
    logger.info("Serving main page")
    return FileResponse("static/index.html")

# Health check endpoint
@app.get("/api/health")
async def health_check():
    logger.info("Health check requested")
    return {
        "ok": True, 
        "pipeline_initialized": pipe is not None,
        "status": "running"
    }

# Simple test endpoint
@app.get("/api/test")
async def test():
    return {"message": "API is working!", "endpoints": ["/", "/api/health", "/api/test", "/api/ask", "/api/ingest", "/api/reset"]}

# Startup check endpoint
@app.get("/api/startup")
async def startup_check():
    try:
        # Test if we can import the pipeline (without initializing)
        from pipeline import RAGPipeline
        return {
            "status": "ready",
            "message": "All modules can be imported successfully",
            "pipeline_available": True
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Module import failed: {str(e)}",
            "pipeline_available": False
        }

@app.post("/api/reset")
async def reset():
    pipeline = get_pipeline()
    logger.info("Resetting RAG pipeline")
    pipeline.reset()
    logger.info("RAG pipeline reset completed")
    return {"ok": True}

@app.post("/api/ingest")
async def ingest(files: List[UploadFile] = File(...)):
    pipeline = get_pipeline()
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
        result = pipeline.ingest_paths(saved)
        logger.info(f"Successfully ingested {len(saved)} files")
        
        # Prepare response with file names, captions, and transcripts
        logger.info(f"Preparing response with audio_transcripts: {result.get('audio_transcripts', {})}")
        ingested_files = []
        for path in saved:
            filename = os.path.basename(path)
            file_info = {"filename": filename}
            
            # Check if this file has a BLIP caption (image)
            if result.get("image_captions") and filename in result["image_captions"]:
                caption = result["image_captions"][filename]
                if caption:
                    file_info["blip_caption"] = caption
                    file_info["is_image"] = True
                    logger.info(f"Added BLIP caption for {filename}")
            
            # Check if this file has an audio transcript
            audio_transcripts = result.get("audio_transcripts", {})
            logger.info(f"Checking audio_transcripts for {filename}: {filename in audio_transcripts}")
            if filename in audio_transcripts:
                transcript = audio_transcripts[filename]
                logger.info(f"Found transcript for {filename}, length: {len(transcript) if transcript else 0}")
                if transcript:
                    file_info["audio_transcript"] = transcript
                    file_info["is_audio"] = True
                    logger.info(f"✅ Added audio transcript for {filename} (length: {len(transcript)} chars)")
            
            ingested_files.append(file_info)
        
        logger.info(f"Final ingested_files: {[f.get('filename') + (' (audio)' if f.get('is_audio') else '') + (' (image)' if f.get('is_image') else '') for f in ingested_files]}")
        
        return {
            "ingested": [os.path.basename(x) for x in saved],
            "files": ingested_files,
            "stats": {
                "text_chunks": result.get("text_chunks", 0),
                "images": result.get("images", 0),
                "facts": result.get("facts", 0)
            }
        }
    except Exception as e:
        logger.error(f"File ingestion failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ask")
async def ask(payload: dict):
    pipeline = get_pipeline()
    q = payload.get("query")
    k = int(payload.get("k", settings.top_k))
    
    if not q:
        logger.warning("Query request missing 'query' parameter")
        raise HTTPException(status_code=400, detail="Missing 'query'")
    
    logger.info(f"Processing query: {q[:100]}{'...' if len(q) > 100 else ''}")
    
    try:
        res = pipeline.query(q, k)
        logger.info(f"Query processed successfully - Response length: {len(res.get('main_response', ''))} chars")
        return JSONResponse(res)
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
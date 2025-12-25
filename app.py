from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
import threading
import time
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

def schedule_file_deletion(file_paths: List[str], delay_seconds: int = 120):
    """Schedule automatic deletion of uploaded files after a delay.

    Args:
        file_paths: List of file paths to delete
        delay_seconds: Delay in seconds before deletion (default: 120 = 2 minutes)
    """
    def delete_files():
        time.sleep(delay_seconds)
        for file_path in file_paths:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
                    logger.info(f"🗑️ Auto-deleted file after {delay_seconds}s: {file_path}")
                else:
                    logger.info(f"File already deleted or not found: {file_path}")
            except Exception as e:
                logger.error(f"Failed to auto-delete file {file_path}: {str(e)}")

    # Start deletion in a background thread
    deletion_thread = threading.Thread(target=delete_files, daemon=True)
    deletion_thread.start()
    logger.info(f"⏰ Scheduled deletion of {len(file_paths)} files in {delay_seconds} seconds")

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
        
        # Prepare response with file names, captions, transcripts, PDF texts, and text contents
        logger.info(f"Preparing response with audio_transcripts: {result.get('audio_transcripts', {})}")
        logger.info(f"Preparing response with pdf_texts: {result.get('pdf_texts', {})}")
        logger.info(f"Preparing response with text_contents: {result.get('text_contents', {})}")
        ingested_files = []
        for path in saved:
            filename = os.path.basename(path)
            file_info = {"filename": filename}
            
            # Determine file type from extension
            ext = os.path.splitext(filename)[1].lower().lstrip(".")
            is_image_file = ext in {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}
            is_audio_file = ext in {"mp3", "wav", "m4a", "flac", "ogg"}
            is_pdf_file = ext == "pdf"
            is_text_file = ext == "txt"
            
            # Check if this file has a BLIP caption (image)
            if result.get("image_captions") and filename in result["image_captions"]:
                caption = result["image_captions"][filename]
                if caption:
                    file_info["blip_caption"] = caption
                    file_info["is_image"] = True
                    logger.info(f"Added BLIP caption for {filename}")
            
            # Check if this file has OCR text (image)
            ocr_texts = result.get("ocr_texts", {})
            logger.info(f"Checking ocr_texts for {filename}: {filename in ocr_texts}")
            if filename in ocr_texts:
                ocr_text = ocr_texts[filename]
                logger.info(f"Found OCR text for {filename}, length: {len(ocr_text) if ocr_text else 0}")
                if ocr_text:
                    file_info["ocr_text"] = ocr_text
                    file_info["is_image"] = True
                    logger.info(f"✅ Added OCR text for {filename} (length: {len(ocr_text)} chars)")
            
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
            
            # Check if this file is a PDF with extracted text
            pdf_texts = result.get("pdf_texts", {})
            logger.info(f"Checking pdf_texts for {filename}: {filename in pdf_texts}")
            if filename in pdf_texts:
                pdf_text = pdf_texts[filename]
                logger.info(f"Found PDF text for {filename}, length: {len(pdf_text) if pdf_text else 0}")
                if pdf_text:
                    file_info["pdf_text"] = pdf_text
                    file_info["is_pdf"] = True
                    logger.info(f"✅ Added PDF text for {filename} (length: {len(pdf_text)} chars)")

            # Check if this file is a text file with extracted content
            text_contents = result.get("text_contents", {})
            logger.info(f"Checking text_contents for {filename}: {filename in text_contents}")
            if filename in text_contents:
                text_content = text_contents[filename]
                logger.info(f"Found text content for {filename}, length: {len(text_content) if text_content else 0}")
                if text_content:
                    file_info["text_content"] = text_content
                    file_info["is_text"] = True
                    logger.info(f"✅ Added text content for {filename} (length: {len(text_content)} chars)")
            
            # MANDATORY: Ensure ALL file types have text for sentiment analysis
            # If no text was extracted, use fallback descriptions to ensure sentiment analysis
            if is_image_file and not file_info.get("ocr_text") and not file_info.get("blip_caption"):
                # Image with no OCR or BLIP - use filename as fallback for sentiment analysis
                file_info["sentiment_fallback"] = f"Image file: {filename}. Visual content uploaded for analysis."
                file_info["is_image"] = True
                logger.info(f"⚠️ No OCR/BLIP for {filename}, using fallback text for sentiment analysis")
            elif is_audio_file and not file_info.get("audio_transcript"):
                # Audio with no transcript - use filename as fallback
                file_info["sentiment_fallback"] = f"Audio file: {filename}. Audio content uploaded for analysis."
                file_info["is_audio"] = True
                logger.info(f"⚠️ No transcript for {filename}, using fallback text for sentiment analysis")
            elif is_pdf_file and not file_info.get("pdf_text"):
                # PDF with no text - use filename as fallback
                file_info["sentiment_fallback"] = f"PDF document: {filename}. Document uploaded for analysis."
                file_info["is_pdf"] = True
                logger.info(f"⚠️ No text extracted from {filename}, using fallback text for sentiment analysis")
            elif is_text_file and not file_info.get("text_content"):
                # Text file with no content - use filename as fallback
                file_info["sentiment_fallback"] = f"Text file: {filename}. Text document uploaded for analysis."
                file_info["is_text"] = True
                logger.info(f"⚠️ No text extracted from {filename}, using fallback text for sentiment analysis")

            ingested_files.append(file_info)
        
        logger.info(f"Final ingested_files: {[f.get('filename') + (' (audio)' if f.get('is_audio') else '') + (' (image)' if f.get('is_image') else '') + (' (pdf)' if f.get('is_pdf') else '') + (' (text)' if f.get('is_text') else '') for f in ingested_files]}")

        # Schedule automatic file deletion after 2 minutes (120 seconds)
        schedule_file_deletion(saved, delay_seconds=120)

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
    skip_retrieval = payload.get("skip_retrieval", False)
    
    if not q:
        logger.warning("Query request missing 'query' parameter")
        raise HTTPException(status_code=400, detail="Missing 'query'")
    
    logger.info(f"Processing query: {q[:100]}{'...' if len(q) > 100 else ''} (skip_retrieval={skip_retrieval})")
    
    try:
        res = pipeline.query(q, k, skip_retrieval=skip_retrieval)
        main_resp_len = len(res.get('main_response', ''))
        sentiment_len = len(res.get('sentiment_analysis', ''))
        logger.info(f"Query processed successfully - Main response: {main_resp_len} chars, Sentiment analysis: {sentiment_len} chars")
        logger.info(f"Sentiment analysis preview: {res.get('sentiment_analysis', '')[:200]}...")
        return JSONResponse(res)
    except Exception as e:
        logger.error(f"Query processing failed: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
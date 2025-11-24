from typing import List, Tuple, Dict, Any, Optional
import os
import io
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element
from pydantic import BaseModel
import re
import tempfile
import subprocess

# Audio transcription using faster-whisper (faster and more efficient than openai-whisper)
try:
    from faster_whisper import WhisperModel
    FASTER_WHISPER_AVAILABLE = True
except Exception:
    FASTER_WHISPER_AVAILABLE = False
    WhisperModel = None

class Chunk(BaseModel):
    id: str
    text: str
    source: str
    doc_type: str
    metadata: Dict[str, Any] = {}

NUMERIC_PATTERNS = [
    ("phq9_total", re.compile(r"PHQ[-\s]?9\s*(?:score|total)\s*[:=]?\s*(\d{1,2})", re.I)),
    ("phq9_threshold", re.compile(r"PHQ[-\s]?9\s*(?:cut[-\s]?off|threshold)\s*[:=]?\s*(\d{1,2})", re.I)),
    ("prevalence_pct", re.compile(r"(\d{1,2}(?:\.\d+)?)\s*%\s*(?:prevalence|of\s+depression)", re.I)),
]

# Cache whisper model to avoid reloading on every call
_whisper_model_cache = None

def transcribe_audio(filepath: str, backend: str = "local", model_size: str = "base") -> str:
    """Transcribe audio file to text using faster-whisper.
    
    Args:
        filepath: Path to audio file
        backend: "local" (uses faster-whisper) | "none" (returns empty)
        model_size: Whisper model size - "tiny", "base", "small", "medium", "large-v2", "large-v3"
                   Default: "base" (good balance of speed and accuracy)
    
    Returns:
        Transcribed text as string
    """
    global _whisper_model_cache
    
    if backend == "none":
        return ""
    
    # local - use faster-whisper
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError(
            "faster-whisper not installed. Install it with: pip install faster-whisper\n"
            "Or set OPENAI_WHISPER=none to disable audio transcription."
        )
    
    # Load model once and cache it for performance
    if _whisper_model_cache is None:
        # Use CPU by default, can be changed to "cuda" if GPU available
        device = "cpu"  # Can be set to "cuda" for GPU acceleration
        compute_type = "int8"  # int8 is faster, use "float16" for better accuracy
        
        _whisper_model_cache = WhisperModel(
            model_size, 
            device=device,
            compute_type=compute_type
        )
    
    # Transcribe audio
    segments, info = _whisper_model_cache.transcribe(
        filepath,
        beam_size=5,  # Balance between speed and accuracy
        language="en"  # Can be set to None for auto-detection
    )
    
    # Combine all segments into a single transcript
    transcript_parts = []
    for segment in segments:
        transcript_parts.append(segment.text)
    
    transcript = " ".join(transcript_parts).strip()
    return transcript

def partition_any(path: str) -> List[Element]:
    # unstructured auto-detects. Set OCR to True via env vars if needed.
    elements = partition(filename=path, extract_images_in_pdf=True)
    return elements

def elements_to_text_chunks(elements: List[Element], source: str, chunk_size: int = 1000, overlap: int = 150) -> List[Chunk]:
    from logging_config import get_logger
    logger = get_logger("extractors")
    
    logger.debug(f"elements_to_text_chunks called with {len(elements)} element(s), source={source}")
    texts = []
    for idx, el in enumerate(elements):
        try:
            logger.debug(f"Processing element {idx}, type: {type(el)}")
            t = el.text
            logger.debug(f"Extracted text from element {idx}, length: {len(t) if t else 0}")
        except Exception as e:
            logger.debug(f"Error accessing .text on element {idx}: {e}, using str() instead")
            t = str(el)
        if t:
            texts.append(t.strip())
            logger.debug(f"Added text to texts list (total: {len(texts)})")
    
    logger.debug(f"Extracted {len(texts)} text segment(s)")
    if not texts:
        logger.debug("No texts extracted, returning empty list")
        return []
    
    merged = "\n\n".join(texts)
    logger.debug(f"Merged text length: {len(merged)} characters")
    
    chunks: List[Chunk] = []
    i = 0
    start = 0
    logger.debug(f"Starting chunking loop with chunk_size={chunk_size}, overlap={overlap}")
    
    while start < len(merged):
        end = min(start + chunk_size, len(merged))
        chunk_txt = merged[start:end]
        chunk_id = f"{source}::chunk::{i}"
        logger.debug(f"Creating chunk {i}: id={chunk_id}, text_length={len(chunk_txt)}, start={start}, end={end}")
        chunks.append(Chunk(id=chunk_id, text=chunk_txt, source=source, doc_type="text"))
        i += 1
        start = end - overlap
        if start < 0:
            start = 0
        if start >= len(merged):
            logger.debug(f"Chunking complete: start={start} >= merged_length={len(merged)}")
            break
    
    logger.debug(f"Chunking completed: created {len(chunks)} chunk(s)")
    return chunks

def extract_numeric_facts(chunks: List[Chunk]) -> List[Tuple[str, str, float]]:
    facts = []
    for c in chunks:
        for key, pat in NUMERIC_PATTERNS:
            for m in pat.finditer(c.text):
                val = m.group(1)
                try:
                    facts.append((c.source, key, float(val)))
                except ValueError:
                    continue
    return facts
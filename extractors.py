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

# PDF image extraction using PyMuPDF
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except Exception:
    PYMUPDF_AVAILABLE = False
    fitz = None

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
# Store as dict: {model_size: WhisperModel} to handle different model sizes
_whisper_model_cache = {}
_current_model_size = None

def transcribe_audio(filepath: str, backend: str = "local", model_size: str = "base") -> str:
    """Transcribe audio file to text using faster-whisper.
    
    Args:
        filepath: Path to audio file
        backend: "local" (uses faster-whisper) | "none" (returns empty)
        model_size: Whisper model size - "tiny", "base", "small", "medium", "large-v2", "large-v3"
                   Default: "base" (good balance of speed and accuracy)
                   This is read from .env file via WHISPER_MODEL_SIZE
    
    Returns:
        Transcribed text as string
    """
    global _whisper_model_cache, _current_model_size
    
    if backend == "none":
        return ""
    
    # local - use faster-whisper
    if not FASTER_WHISPER_AVAILABLE:
        raise RuntimeError(
            "faster-whisper not installed. Install it with: pip install faster-whisper\n"
            "Or set OPENAI_WHISPER=none to disable audio transcription."
        )
    
    # Load model once per model_size and cache it for performance
    # If model_size changes, load new model
    if model_size not in _whisper_model_cache:
        from logging_config import get_logger
        logger = get_logger("extractors")
        logger.info(f"Loading Whisper model: {model_size} (from WHISPER_MODEL_SIZE in .env)")
        
        # Use CPU by default, can be changed to "cuda" if GPU available
        device = "cpu"  # Can be set to "cuda" for GPU acceleration
        compute_type = "int8"  # int8 is faster, use "float16" for better accuracy
        
        _whisper_model_cache[model_size] = WhisperModel(
            model_size, 
            device=device,
            compute_type=compute_type
        )
        _current_model_size = model_size
        logger.info(f"✅ Whisper model {model_size} loaded and cached")
    
    # Use the cached model for this model_size
    whisper_model = _whisper_model_cache[model_size]
    
    # Transcribe audio using the cached model
    # Optimized settings for faster transcription
    segments, info = whisper_model.transcribe(
        filepath,
        beam_size=3,  # Reduced from 5 to 3 for faster processing (slight accuracy trade-off)
        language="en",  # Can be set to None for auto-detection
        vad_filter=True,  # Voice Activity Detection - skip silence for faster processing
        vad_parameters=dict(min_silence_duration_ms=500),  # Skip silence > 500ms
        condition_on_previous_text=False  # Disable for faster processing (slight accuracy trade-off)
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

def extract_ocr_text(image_path: str) -> str:
    """Extract text from image using OCR (unstructured with OCR enabled).
    
    Args:
        image_path: Path to image file
        
    Returns:
        Extracted text as string, empty string if no text found or error
    """
    from logging_config import get_logger
    logger = get_logger("extractors")
    
    try:
        logger.info(f"Attempting OCR extraction from image: {image_path}")
        # Use unstructured with OCR enabled for images
        # Set strategy to "ocr_only" to force OCR processing
        elements = partition(
            filename=image_path,
            strategy="ocr_only",  # Force OCR processing
            languages=["eng"]  # English language
        )
        
        # Extract text from all elements
        ocr_text_parts = []
        for el in elements:
            try:
                text = el.text if hasattr(el, 'text') else str(el)
                if text and text.strip():
                    ocr_text_parts.append(text.strip())
            except Exception as e:
                logger.debug(f"Error extracting text from element: {e}")
                continue
        
        # Combine all text parts
        ocr_text = "\n".join(ocr_text_parts).strip()
        
        if ocr_text:
            logger.info(f"OCR extracted {len(ocr_text)} characters from image")
        else:
            logger.info("OCR found no text in image")
        
        return ocr_text
        
    except Exception as e:
        logger.warning(f"OCR extraction failed for {image_path}: {str(e)}")
        return ""

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
    max_chunks = 10000  # Safety limit to prevent infinite loops
    logger.debug(f"Starting chunking loop with chunk_size={chunk_size}, overlap={overlap}, text_length={len(merged)}")
    
    while start < len(merged) and i < max_chunks:
        end = min(start + chunk_size, len(merged))
        chunk_txt = merged[start:end]
        
        # Skip empty chunks
        if not chunk_txt.strip():
            break
            
        chunk_id = f"{source}::chunk::{i}"
        logger.debug(f"Creating chunk {i}: id={chunk_id}, text_length={len(chunk_txt)}, start={start}, end={end}")
        chunks.append(Chunk(id=chunk_id, text=chunk_txt, source=source, doc_type="text"))
        i += 1
        
        # Calculate next start position with overlap
        next_start = end - overlap
        
        # CRITICAL FIX: Ensure we always advance forward
        # If overlap >= chunk_size or text is shorter than overlap, we'd loop infinitely
        if next_start <= start:
            # If we can't advance, move forward by at least 1 character to prevent infinite loop
            next_start = start + 1
            logger.debug(f"⚠️ Overlap too large or text too short - advancing by 1 char to prevent infinite loop")
        
        start = next_start
        
        # Safety check: if we've reached the end, break
        if start >= len(merged):
            logger.debug(f"Chunking complete: start={start} >= merged_length={len(merged)}")
            break
    
    if i >= max_chunks:
        logger.warning(f"⚠️ Chunking stopped at safety limit ({max_chunks} chunks) - possible infinite loop prevented!")
    
    logger.info(f"Chunking completed: created {len(chunks)} chunk(s) from {len(merged)} character text")
    return chunks

def extract_pdf_images(pdf_path: str, output_dir: Optional[str] = None) -> List[str]:
    """Extract images from PDF file using PyMuPDF.
    
    Args:
        pdf_path: Path to PDF file
        output_dir: Directory to save extracted images. If None, uses temp directory.
        
    Returns:
        List of paths to extracted image files (PNG format)
    """
    from logging_config import get_logger
    logger = get_logger("extractors")
    
    if not PYMUPDF_AVAILABLE:
        logger.warning("PyMuPDF not available. Install it with: pip install PyMuPDF")
        return []
    
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="pdf_images_")
    
    os.makedirs(output_dir, exist_ok=True)
    
    extracted_images = []
    
    try:
        logger.info(f"Extracting images from PDF: {pdf_path}")
        doc = fitz.open(pdf_path)
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            image_list = page.get_images()
            
            for img_index, img in enumerate(image_list):
                try:
                    # Get image data
                    xref = img[0]
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    # Save image to file
                    image_filename = f"page_{page_num + 1}_img_{img_index + 1}.{image_ext}"
                    image_path = os.path.join(output_dir, image_filename)
                    
                    with open(image_path, "wb") as img_file:
                        img_file.write(image_bytes)
                    
                    extracted_images.append(image_path)
                    logger.info(f"Extracted image: {image_filename} from page {page_num + 1}")
                    
                except Exception as e:
                    logger.warning(f"Error extracting image {img_index} from page {page_num + 1}: {str(e)}")
                    continue
        
        doc.close()
        logger.info(f"Extracted {len(extracted_images)} image(s) from PDF")
        
    except Exception as e:
        logger.error(f"Error extracting images from PDF {pdf_path}: {str(e)}", exc_info=True)
    
    return extracted_images

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
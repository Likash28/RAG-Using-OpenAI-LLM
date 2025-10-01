from typing import List, Tuple, Dict, Any, Optional
import os
import io
from unstructured.partition.auto import partition
from unstructured.documents.elements import Element
from pydantic import BaseModel
import re
import tempfile
import subprocess

# Optional: local Whisper fallback
try:
    import whisper
except Exception:
    whisper = None

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

def transcribe_audio(filepath: str, backend: str = "local") -> str:
    """Transcribe audio file to text. backend: local|api|none
    - local: uses openai-whisper (offline). Requires ffmpeg.
    - api: (left as placeholder to call OpenAI Whisper API if desired).
    - none: returns empty string.
    """
    if backend == "none":
        return ""
    if backend == "api":
        # Placeholder: implement OpenAI Whisper API call if you have a key.
        # Return empty string to avoid runtime errors when not configured.
        return ""
    # local
    if whisper is None:
        raise RuntimeError("Local whisper not installed. Set OPENAI_WHISPER=api|none or install openai-whisper.")
    model = whisper.load_model("base")
    result = model.transcribe(filepath)
    return result.get("text", "").strip()

def partition_any(path: str) -> List[Element]:
    # unstructured auto-detects. Set OCR to True via env vars if needed.
    elements = partition(filename=path, extract_images_in_pdf=True)
    return elements

def elements_to_text_chunks(elements: List[Element], source: str, chunk_size: int = 1000, overlap: int = 150) -> List[Chunk]:
    texts = []
    for el in elements:
        try:
            t = el.text
        except Exception:
            t = str(el)
        if t:
            texts.append(t.strip())
    if not texts:
        return []
    merged = "\n\n".join(texts)
    chunks: List[Chunk] = []
    i = 0
    start = 0
    while start < len(merged):
        end = min(start + chunk_size, len(merged))
        chunk_txt = merged[start:end]
        chunks.append(Chunk(id=f"{source}::chunk::{i}", text=chunk_txt, source=source, doc_type="text"))
        i += 1
        start = end - overlap
        if start < 0:
            start = 0
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

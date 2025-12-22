from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import os
import sqlite3
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

from config import settings
from logging_config import get_logger
from embedder import TextEmbedder, BLIPMultimodal
from vectorstore import DualIndex
from extractors import partition_any, elements_to_text_chunks, transcribe_audio, extract_numeric_facts, extract_ocr_text, extract_pdf_images, Chunk
from openai_llm import OpenAILLM
from prompts.loader import load_system_prompt, is_crisis_query, get_crisis_response, is_off_topic, get_off_topic_response

logger = get_logger("RAGPipeline")

@dataclass
class Stores:
    index: DualIndex
    sqlite_path: str

class RAGPipeline:
    def __init__(self):
        from config import validate_settings
        validate_settings()  # Validate settings before initializing
        logger.info("Initializing RAG Pipeline components")
        self.text_embedder = TextEmbedder(settings.text_model_name)
        logger.info("Text embedder initialized")
        self.blip_embedder = BLIPMultimodal(settings.blip_model_name)
        logger.info("BLIP embedder initialized")
        self.index = DualIndex(settings.chroma_dir)
        logger.info("Vector index initialized")
        self.sqlite_path = settings.sqlite_path
        self._ensure_sqlite()
        self.llm = self._init_llm()
        # Thread locks for model operations (PyTorch models may not be fully thread-safe)
        self._text_embedder_lock = threading.Lock()
        self._blip_embedder_lock = threading.Lock()
        logger.info("RAG Pipeline initialization completed")

    def _ensure_sqlite(self):
        os.makedirs(os.path.dirname(self.sqlite_path) or ".", exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS facts (source TEXT, key TEXT, value REAL, PRIMARY KEY(source, key, value))"
            )
            con.commit()

    def _init_llm(self):
        if settings.provider == "openai" and settings.openai_api_key:
            logger.info("Initializing OpenAI LLM")
            return OpenAILLM(api_key=settings.openai_api_key, model_name=settings.openai_model)
        # Return None if no valid API key is provided
        logger.warning("No valid LLM provider configured. Please set OPENAI_API_KEY environment variable.")
        return None

    # --------------- Ingest ---------------
    def _process_single_file(self, p: str) -> Tuple[List[Chunk], List[Dict[str, Any]], List[Tuple[str, str, float]], Dict[str, str], str, Dict[str, str], str]:
        """Process a single file and return chunks, image data, facts, image captions, audio transcript, OCR texts, and PDF text.
        Returns: (chunks, image_data_list, facts, image_captions, audio_transcript, ocr_texts, pdf_text)
        - chunks: List of text chunks
        - image_data_list: List of Dict with 'id', 'meta', 'emb' for each image (empty list if no images)
        - facts: List of numeric facts
        - image_captions: Dict mapping image_id to BLIP caption string
        - audio_transcript: Transcript text if audio file, else None
        - ocr_texts: Dict mapping image_id to OCR extracted text
        - pdf_text: Full extracted text from PDF if PDF file, else None
        """
        ext = Path(p).suffix.lower().lstrip(".")
        source_id = os.path.basename(p)
        chunks = []
        image_data_list = []  # Changed to list to handle multiple images from PDFs
        facts = []
        image_captions = {}  # Changed to dict to handle multiple captions
        audio_transcript = None
        ocr_texts = {}  # Changed to dict to handle multiple OCR texts
        pdf_text = None  # Full text extracted from PDF for sentiment analysis

        logger.info(f"🚀 Processing file: {source_id} ({ext})")

        try:
            if ext in {"mp3", "wav", "m4a", "flac", "ogg"}:
                # Transcribe audio to text
                logger.info(f"=== Transcribing audio file: {source_id} ===")
                try:
                    # Use settings from config (which reads from .env)
                    transcript = transcribe_audio(
                        p, 
                        backend=settings.openai_whisper,  # "local" or "none" from .env
                        model_size=settings.whisper_model_size  # "tiny", "base", "small", etc. from .env
                    )
                except Exception as e:
                    logger.error(f"❌ Error transcribing audio file {p}: {str(e)}", exc_info=True)
                    chunks = []
                    audio_transcript = None
                else:
                    if transcript:
                        logger.info(f"=== Audio Transcription Complete ===")
                        logger.info(f"Audio file: {source_id}")
                        logger.info(f"Transcript length: {len(transcript)} characters")
                        logger.info(f"Transcript preview: {transcript[:200]}...")
                        logger.info(f"=== End Transcription ===")
                        
                        # ⚡ OPTIMIZATION: Skip chunking/embedding/storage for audio transcripts
                        # Audio transcripts are sent directly to sentiment analysis, no need for vector DB
                        # This significantly speeds up the process - no embedding delay!
                        chunks = []  # No chunks - audio won't be stored in vector DB
                        audio_transcript = transcript  # Store full transcript for return to frontend
                        logger.info(f"✅ Audio transcript ready (skipping vector DB storage - will go directly to sentiment analysis)")
                    else:
                        logger.warning(f"⚠️ Audio transcription returned empty result for {source_id}")
                        chunks = []
                        audio_transcript = None
                    
            elif ext == "pdf":
                # PDF processing - extract images and text
                logger.info(f"📄 Processing PDF file: {source_id}")
                
                # Extract text from PDF using unstructured (current package)
                # Fallback to PyMuPDF if unstructured fails (e.g., poppler not installed)
                try:
                    elements = partition_any(p)
                    
                    # Extract full text for sentiment analysis (similar to audio transcripts)
                    full_text_parts = []
                    for el in elements:
                        try:
                            text = el.text if hasattr(el, 'text') else str(el)
                            if text and text.strip():
                                full_text_parts.append(text.strip())
                        except Exception as e:
                            logger.debug(f"Error extracting text from element: {e}")
                            continue
                    
                    # Combine all text into one string for sentiment analysis
                    if full_text_parts:
                        pdf_text = "\n\n".join(full_text_parts)
                        logger.info(f"✅ Extracted {len(pdf_text)} characters of text from PDF for sentiment analysis")
                        logger.info(f"PDF text preview: {pdf_text[:200]}...")
                    else:
                        logger.warning(f"⚠️ No text extracted from PDF {source_id}")
                        pdf_text = None
                    
                    # Also create chunks for vector DB storage (for RAG queries)
                    text_chunks = elements_to_text_chunks(elements, source=source_id)
                    chunks.extend(text_chunks)
                    logger.info(f"✅ Created {len(text_chunks)} text chunk(s) from PDF for vector DB storage")
                    
                except Exception as e:
                    logger.warning(f"Error extracting text from PDF {p} using unstructured: {str(e)}")
                    logger.info(f"🔄 Attempting fallback PDF text extraction using PyMuPDF...")
                    
                    # Fallback: Use PyMuPDF to extract text directly
                    try:
                        import fitz  # PyMuPDF
                        doc = fitz.open(p)
                        full_text_parts = []
                        
                        for page_num in range(len(doc)):
                            page = doc[page_num]
                            page_text = page.get_text()
                            if page_text and page_text.strip():
                                full_text_parts.append(page_text.strip())
                        
                        doc.close()
                        
                        if full_text_parts:
                            pdf_text = "\n\n".join(full_text_parts)
                            logger.info(f"✅ Extracted {len(pdf_text)} characters of text from PDF using PyMuPDF fallback")
                            logger.info(f"PDF text preview: {pdf_text[:200]}...")
                            
                            # Create chunks from extracted text
                            class DummyElement:
                                def __init__(self, text):
                                    self.text = text
                            
                            dummy_elements = [DummyElement(text) for text in full_text_parts]
                            text_chunks = elements_to_text_chunks(dummy_elements, source=source_id)
                            chunks.extend(text_chunks)
                            logger.info(f"✅ Created {len(text_chunks)} text chunk(s) from PDF using PyMuPDF fallback")
                        else:
                            logger.warning(f"⚠️ No text extracted from PDF {source_id} using PyMuPDF fallback")
                            pdf_text = None
                    except Exception as fallback_error:
                        logger.error(f"❌ Both unstructured and PyMuPDF failed to extract text from PDF {p}: {str(fallback_error)}")
                        pdf_text = None
                
                # Extract images from PDF using PyMuPDF
                pdf_temp_dir = None
                try:
                    # Extract images to a temp directory
                    import tempfile
                    pdf_temp_dir = tempfile.mkdtemp(prefix=f"pdf_images_{source_id}_")
                    pdf_images = extract_pdf_images(p, output_dir=pdf_temp_dir)
                    logger.info(f"📸 Extracted {len(pdf_images)} image(s) from PDF")
                    
                    # Process each extracted image with BLIP
                    for img_idx, img_path in enumerate(pdf_images):
                        try:
                            # Generate image embedding (with thread lock for safety)
                            with self._blip_embedder_lock:
                                img_emb = self.blip_embedder.embed_images([img_path])[0]
                            
                            # Create unique ID for this PDF image
                            img_id = f"{source_id}::pdf_image::{img_idx + 1}"
                            image_data = {
                                "id": img_id,
                                "meta": {
                                    "source": source_id,
                                    "image_path": os.path.abspath(img_path),
                                    "pdf_page": img_idx + 1,
                                    "is_pdf_image": True
                                },
                                "emb": img_emb
                            }
                            image_data_list.append(image_data)
                            
                            # Generate BLIP caption (with thread lock for safety)
                            logger.info(f"=== Generating BLIP caption for PDF image {img_idx + 1}: {source_id} ===")
                            try:
                                with self._blip_embedder_lock:
                                    blip_caption = self.blip_embedder.generate_caption(img_path)
                                
                                logger.info(f"=== BLIP CAPTION GENERATED ===")
                                logger.info(f"PDF Image {img_idx + 1}: {source_id}")
                                logger.info(f"Full Caption: {blip_caption}")
                                logger.info(f"=== End of BLIP Caption ===")
                                
                                image_captions[img_id] = blip_caption
                                
                                # Add BLIP caption as chunk for Vector DB
                                blip_chunk = Chunk(
                                    id=f"{img_id}::blip_caption",
                                    text=f"[PDF Image {img_idx + 1} Description from BLIP]: {blip_caption}",
                                    source=source_id,
                                    doc_type="pdf_image_caption"
                                )
                                chunks.append(blip_chunk)
                                logger.info(f"✅ Added BLIP caption as chunk for PDF image {img_idx + 1}")
                                
                            except Exception as caption_error:
                                logger.error(f"Error generating BLIP caption for PDF image {img_path}: {str(caption_error)}", exc_info=True)
                            
                            # Attempt OCR on PDF image
                            logger.info(f"=== Attempting OCR extraction for PDF image {img_idx + 1}: {source_id} ===")
                            try:
                                extracted_ocr_text = extract_ocr_text(img_path)
                                if extracted_ocr_text and len(extracted_ocr_text.strip()) > 0:
                                    logger.info(f"✅ OCR extracted {len(extracted_ocr_text)} characters from PDF image {img_idx + 1}")
                                    ocr_texts[img_id] = extracted_ocr_text
                                    
                                    # Create chunks from OCR text for Vector DB storage
                                    class DummyElement:
                                        def __init__(self, text):
                                            self.text = text
                                    
                                    dummy_element = DummyElement(extracted_ocr_text)
                                    ocr_chunks = elements_to_text_chunks(
                                        [dummy_element], 
                                        source=source_id,
                                        chunk_size=1000,
                                        overlap=150
                                    )
                                    
                                    # Mark OCR chunks with doc_type
                                    for chunk in ocr_chunks:
                                        chunk.doc_type = "pdf_ocr_text"
                                        chunk.id = f"{img_id}::ocr_chunk::{chunk.id.split('::')[-1]}"
                                    
                                    chunks.extend(ocr_chunks)
                                    logger.info(f"✅ PDF image OCR text split into {len(ocr_chunks)} chunk(s)")
                                else:
                                    logger.info(f"⚠️ OCR found no text in PDF image {img_idx + 1}")
                            except Exception as ocr_error:
                                logger.warning(f"⚠️ OCR extraction failed for PDF image {img_path}: {str(ocr_error)}")
                                
                        except Exception as img_error:
                            logger.error(f"Error processing PDF image {img_path}: {str(img_error)}", exc_info=True)
                            continue
                    
                    logger.info(f"✅ Processed {len(image_data_list)} image(s) from PDF {source_id}")
                    
                    # Clean up temp directory after processing all images
                    if pdf_temp_dir and os.path.exists(pdf_temp_dir):
                        try:
                            import shutil
                            shutil.rmtree(pdf_temp_dir)
                            logger.info(f"✅ Cleaned up temp directory for PDF {source_id}")
                        except Exception as cleanup_error:
                            logger.warning(f"Could not clean up temp directory {pdf_temp_dir}: {str(cleanup_error)}")
                    
                except Exception as pdf_error:
                    logger.error(f"Error extracting images from PDF {p}: {str(pdf_error)}", exc_info=True)
                    # Clean up temp directory on error
                    if pdf_temp_dir and os.path.exists(pdf_temp_dir):
                        try:
                            import shutil
                            shutil.rmtree(pdf_temp_dir)
                        except Exception:
                            pass
                
            elif ext in {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}:
                # Image processing
                try:
                    from PIL import Image
                    with Image.open(p) as img:
                        img.verify()
                    
                    # Generate image embedding (with thread lock for safety)
                    with self._blip_embedder_lock:
                        img_emb = self.blip_embedder.embed_images([p])[0]
                    img_id = f"{source_id}::image"
                    image_data = {
                        "id": img_id,
                        "meta": {"source": source_id, "image_path": os.path.abspath(p)},
                        "emb": img_emb
                    }
                    image_data_list.append(image_data)
                    
                    # Generate BLIP caption (with thread lock for safety)
                    logger.info(f"=== Generating BLIP caption for image: {source_id} ===")
                    try:
                        with self._blip_embedder_lock:
                            blip_caption = self.blip_embedder.generate_caption(p)
                        
                        logger.info(f"=== BLIP CAPTION GENERATED ===")
                        logger.info(f"Image: {source_id}")
                        logger.info(f"Full Caption: {blip_caption}")
                        logger.info(f"=== End of BLIP Caption ===")
                        
                        image_captions[img_id] = blip_caption
                    except Exception as caption_error:
                        logger.error(f"Error generating BLIP caption for {p}: {str(caption_error)}", exc_info=True)
                    
                    # ⚡ OCR Processing: Always attempt OCR for images with text
                    # OCR text will be stored in Vector DB AND sent for sentiment analysis
                    logger.info(f"=== Attempting OCR extraction for image: {source_id} ===")
                    try:
                        extracted_ocr_text = extract_ocr_text(p)
                        if extracted_ocr_text and len(extracted_ocr_text.strip()) > 0:
                            logger.info(f"✅ OCR extracted {len(extracted_ocr_text)} characters from {source_id}")
                            ocr_texts[img_id] = extracted_ocr_text
                            
                            # Create chunks from OCR text for Vector DB storage
                            class DummyElement:
                                def __init__(self, text):
                                    self.text = text
                            
                            dummy_element = DummyElement(extracted_ocr_text)
                            ocr_chunks = elements_to_text_chunks(
                                [dummy_element], 
                                source=source_id,
                                chunk_size=1000,
                                overlap=150
                            )
                            
                            # Mark OCR chunks with doc_type
                            for chunk in ocr_chunks:
                                chunk.doc_type = "ocr_text"
                            
                            chunks = ocr_chunks  # Use OCR chunks for Vector DB
                            logger.info(f"✅ OCR text split into {len(chunks)} chunk(s) - will be stored in Vector DB")
                            
                            # Also add BLIP caption as additional chunk if available (for better context)
                            if img_id in image_captions:
                                blip_caption = image_captions[img_id]
                                blip_chunk = Chunk(
                                    id=f"{img_id}::blip_caption",
                                    text=f"[Image Description from BLIP]: {blip_caption}",
                                    source=source_id,
                                    doc_type="image_caption"
                                )
                                chunks.append(blip_chunk)
                                logger.info(f"✅ Added BLIP caption as additional chunk for context")
                        else:
                            logger.info(f"⚠️ OCR found no text in image {source_id} (or text too short)")
                            # Fallback to BLIP caption if no OCR text
                            if img_id in image_captions:
                                blip_caption = image_captions[img_id]
                                blip_chunk = Chunk(
                                    id=f"{img_id}::blip_caption",
                                    text=f"[Image Description from BLIP]: {blip_caption}",
                                    source=source_id,
                                    doc_type="image_caption"
                                )
                                chunks = [blip_chunk]
                                logger.info(f"✅ Using BLIP caption as fallback (no OCR text found)")
                            else:
                                chunks = []
                    except Exception as ocr_error:
                        logger.warning(f"⚠️ OCR extraction failed for {source_id}: {str(ocr_error)}")
                        # Fallback to BLIP caption if OCR fails
                        if img_id in image_captions:
                            blip_caption = image_captions[img_id]
                            blip_chunk = Chunk(
                                id=f"{img_id}::blip_caption",
                                text=f"[Image Description from BLIP]: {blip_caption}",
                                source=source_id,
                                doc_type="image_caption"
                            )
                            chunks = [blip_chunk]
                            logger.info(f"✅ Using BLIP caption as fallback (OCR failed)")
                        else:
                            chunks = []
                        
                except Exception as e:
                    error_msg = str(e)
                    if "cannot identify image file" in error_msg or "UnidentifiedImageError" in error_msg:
                        logger.warning(f"File {source_id} has image extension but is not a valid image file. Skipping.")
                        try:
                            logger.info(f"Attempting to process {source_id} as text file...")
                            elements = partition_any(p)
                            chunks = elements_to_text_chunks(elements, source=source_id)
                            if chunks:
                                logger.info(f"Successfully processed {source_id} as text file with {len(chunks)} chunk(s)")
                        except Exception as text_error:
                            logger.error(f"Failed to process {source_id} as text file: {str(text_error)}")
                    else:
                        logger.error(f"Error processing image {p}: {error_msg}", exc_info=True)
                        chunks = []
                        image_data_list = []
                
                # Skip OCR for speed - BLIP captions are sufficient
                # OCR is slow and optional, so we skip it to speed up ingestion
                # if chunks and any(c.doc_type == "image_caption" for c in chunks):
                #     try:
                #         elements = partition_any(p)
                #         ocr_chunks = elements_to_text_chunks(elements, source=source_id)
                #         chunks.extend(ocr_chunks)
                #     except Exception as e:
                #         logger.warning(f"OCR extraction failed for {p}: {str(e)}")
            else:
                # Text files - Direct text extraction, NO BLIP needed
                logger.info(f"📄 Processing text file: {source_id} (direct extraction, no BLIP)")
                elements = partition_any(p)
                chunks = elements_to_text_chunks(elements, source=source_id)
                if chunks:
                    logger.info(f"✅ Text file processed: {len(chunks)} chunk(s) extracted directly")
                else:
                    logger.warning(f"⚠️ No text chunks extracted from {source_id}")

            # Extract numeric facts (only if we have chunks)
            if chunks:
                logger.info(f"Extracting numeric facts from {len(chunks)} chunk(s) for {source_id}...")
                try:
                    facts = extract_numeric_facts(chunks)
                    logger.info(f"Extracted {len(facts)} fact(s) from {source_id}")
                except Exception as fact_error:
                    logger.warning(f"Error extracting facts from {source_id}: {str(fact_error)}")
                    facts = []
            else:
                facts = []
                logger.info(f"No chunks to extract facts from for {source_id}")
                
        except Exception as e:
            logger.error(f"❌ Unexpected error processing file {p}: {str(e)}", exc_info=True)
            # Ensure we always return something, even on error
            if chunks is None:
                chunks = []
            if facts is None:
                facts = []
            if image_data_list is None:
                image_data_list = []
            if image_captions is None:
                image_captions = {}
            if audio_transcript is None:
                audio_transcript = None
            if ocr_texts is None:
                ocr_texts = {}
            if pdf_text is None:
                pdf_text = None
        
        logger.info(f"🔚 Returning from _process_single_file for {source_id}: {len(chunks)} chunks, {len(image_data_list)} images, {len(facts)} facts, {len(image_captions)} captions, audio_transcript={'present' if audio_transcript else 'None'}, {len(ocr_texts)} OCR texts, pdf_text={'present' if pdf_text else 'None'}")
        return chunks, image_data_list, facts, image_captions, audio_transcript, ocr_texts, pdf_text

    def ingest_paths(self, paths: List[str]) -> Dict[str, Any]:
        logger.info(f"Starting parallel ingestion of {len(paths)} files")
        
        # Process files in parallel - collect all results first
        max_workers = min(len(paths), 4)  # Limit to 4 parallel workers
        logger.info(f"Using {max_workers} parallel workers for file processing")
        
        all_chunks = []  # Collect all chunks from all files
        all_image_data = []  # Collect all image data
        all_facts = []  # Collect all facts
        image_captions = {}  # Store captions for return
        audio_transcripts = {}  # Store audio transcripts for return
        ocr_texts = {}  # Store OCR texts for return
        pdf_texts = {}  # Store PDF texts for return (for sentiment analysis)
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all files for processing
            future_to_path = {executor.submit(self._process_single_file, p): p for p in paths}
            
            # Collect results as they complete (with timeout per file)
            import time
            start_time = time.time()
            max_file_time = 60  # Max 60 seconds per file
            file_start_times = {p: time.time() for p in paths}  # Track start time for each file
            
            for future in as_completed(future_to_path, timeout=max_file_time * len(paths)):
                p = future_to_path[future]
                file_start = file_start_times.get(p, time.time())
                logger.info(f"⏳ Waiting for result from {os.path.basename(p)}...")
                try:
                    # Get result with timeout
                    chunks, image_data_list, facts, file_image_captions, audio_transcript, file_ocr_texts, pdf_text = future.result(timeout=max_file_time)
                    logger.info(f"✅ Got result from {os.path.basename(p)}: {len(chunks)} chunks, {len(image_data_list)} images, audio_transcript={'present' if audio_transcript else 'None'}, {len(file_ocr_texts)} OCR texts, pdf_text={'present' if pdf_text else 'None'}")
                    
                    # Collect chunks for batch embedding later
                    if chunks:
                        all_chunks.extend(chunks)
                        logger.info(f"Collected {len(chunks)} chunk(s) from {os.path.basename(p)}")
                    
                    # Collect image data (now a list, can have multiple images from PDFs)
                    if image_data_list:
                        for img_data in image_data_list:
                            # Get caption for this specific image
                            img_id = img_data.get("id", "")
                            img_caption = file_image_captions.get(img_id)
                            all_image_data.append((img_data, p, img_caption))
                        logger.info(f"Collected {len(image_data_list)} image(s) from {os.path.basename(p)}")
                    
                    # Collect facts
                    if facts:
                        all_facts.extend(facts)
                        logger.info(f"Collected {len(facts)} fact(s) from {os.path.basename(p)}")
                    
                    # Store image captions (merge into main dict)
                    if file_image_captions:
                        image_captions.update(file_image_captions)
                        logger.info(f"Stored {len(file_image_captions)} image caption(s) from {os.path.basename(p)}")
                    
                    # Store audio transcripts
                    if audio_transcript:
                        source_id = os.path.basename(p)
                        audio_transcripts[source_id] = audio_transcript
                    
                    # Store OCR texts (merge into main dict)
                    if file_ocr_texts:
                        ocr_texts.update(file_ocr_texts)
                        logger.info(f"Stored {len(file_ocr_texts)} OCR text(s) from {os.path.basename(p)}")
                    
                    # Store PDF text for sentiment analysis (similar to audio transcripts)
                    if pdf_text:
                        source_id = os.path.basename(p)
                        pdf_texts[source_id] = pdf_text
                        logger.info(f"✅ Stored PDF text from {source_id} ({len(pdf_text)} chars) for sentiment analysis")
                    
                    elapsed = time.time() - file_start
                    logger.info(f"✅ Completed processing: {os.path.basename(p)} (took {elapsed:.2f}s)")
                    
                except TimeoutError:
                    elapsed = time.time() - file_start
                    logger.error(f"❌ File {os.path.basename(p)} processing timed out after {elapsed:.2f}s - skipping")
                except Exception as e:
                    elapsed = time.time() - file_start
                    logger.error(f"❌ Error processing file {os.path.basename(p)} after {elapsed:.2f}s: {str(e)}", exc_info=True)
            
            total_time = time.time() - start_time
            logger.info(f"All files processed in {total_time:.2f}s")
        
        # ⚡ PERFORMANCE ANALYSIS: Log timing for each stage
        import time
        post_processing_start = time.time()
        logger.info(f"⏱️  TIMING: File processing completed in {total_time:.2f}s")
        logger.info(f"⏱️  TIMING: Starting post-processing (embedding + storage)")
        
        # Prepare result with transcripts/captions (ready immediately)
        logger.info(f"📊 Audio transcripts ready: {list(audio_transcripts.keys())} with {sum(1 for t in audio_transcripts.values() if t)} non-empty")
        logger.info(f"📊 OCR texts ready: {list(ocr_texts.keys())} with {sum(1 for t in ocr_texts.values() if t)} non-empty")
        logger.info(f"📊 PDF texts ready: {list(pdf_texts.keys())} with {sum(1 for t in pdf_texts.values() if t)} non-empty")
        
        # Batch embed all text chunks at once (OPTIMIZED for speed)
        # NOTE: Audio transcripts are NOT included here - they skip vector DB storage
        text_ids, text_docs, text_meta, text_embs = [], [], [], []
        embed_time = 0
        if all_chunks:
            # Filter out any audio chunks (shouldn't be any, but just in case)
            non_audio_chunks = [c for c in all_chunks if c.doc_type != "audio"]
            if non_audio_chunks:
                text_ids = [c.id for c in non_audio_chunks]
                text_docs = [c.text for c in non_audio_chunks]
                text_meta = [{"source": c.source, "doc_type": c.doc_type} for c in non_audio_chunks]
                logger.info(f"📝 Embedding {len(text_docs)} chunk(s) (audio transcripts excluded - they skip vector DB)...")
                embed_start = time.time()
                text_embs = self.text_embedder.embed_texts(text_docs)  # Batch embedding
                embed_time = time.time() - embed_start
                logger.info(f"⏱️  TIMING: Embedding took {embed_time:.2f}s ({len(text_docs)} chunks)")
            else:
                logger.info(f"📝 No non-audio chunks to embed (all chunks were audio - skipping embedding)")
        
        # Process all image data (already embedded, just collecting)
        img_ids, img_meta, img_embs = [], [], []
        for image_data, p, image_caption in all_image_data:
            img_ids.append(image_data["id"])
            img_meta.append(image_data["meta"])
            img_embs.append(image_data["emb"])
        
        facts_to_insert = all_facts

        # Batch insert all data at once (OPTIMIZED)
        logger.info(f"💾 Storing in vector database...")
        store_start = time.time()
        if text_ids:
            self.index.add_texts(text_ids, text_docs, text_meta, text_embs)
        if img_ids:
            self.index.add_images(img_ids, img_meta, img_embs)
        if facts_to_insert:
            with sqlite3.connect(self.sqlite_path) as con:
                con.executemany("INSERT OR IGNORE INTO facts (source, key, value) VALUES (?, ?, ?)", facts_to_insert)
                con.commit()
        store_time = time.time() - store_start
        logger.info(f"⏱️  TIMING: Storage took {store_time:.2f}s")
        
        post_processing_time = time.time() - post_processing_start
        total_ingestion_time = time.time() - start_time
        
        # ⚡ PERFORMANCE SUMMARY
        logger.info(f"⏱️  ========== PERFORMANCE SUMMARY ==========")
        logger.info(f"⏱️  File Processing: {total_time:.2f}s")
        logger.info(f"⏱️  Embedding: {embed_time:.2f}s ({len(text_ids)} chunks)")
        logger.info(f"⏱️  Storage: {store_time:.2f}s")
        logger.info(f"⏱️  Post-processing Total: {post_processing_time:.2f}s")
        logger.info(f"⏱️  TOTAL INGESTION TIME: {total_ingestion_time:.2f}s")
        logger.info(f"⏱️  ===========================================")
        
        # Create filename-based mapping for backward compatibility with app.py
        # For PDFs with multiple images, combine all captions
        image_captions_by_filename = {}
        for img_data, p, img_caption in all_image_data:
            filename = os.path.basename(p)
            if img_caption:
                if filename not in image_captions_by_filename:
                    image_captions_by_filename[filename] = []
                image_captions_by_filename[filename].append(img_caption)
        
        # Convert lists to single strings for single images, keep lists for PDFs
        for filename, captions in image_captions_by_filename.items():
            if len(captions) == 1:
                image_captions_by_filename[filename] = captions[0]
            else:
                # For multiple images (PDFs), combine with separators
                image_captions_by_filename[filename] = " | ".join([f"Image {i+1}: {cap}" for i, cap in enumerate(captions)])
        
        # Create filename-based OCR texts mapping
        ocr_texts_by_filename = {}
        for img_id, ocr_text in ocr_texts.items():
            # Extract filename from image_id (format: "filename::pdf_image::1" or "filename::image")
            if "::" in img_id:
                filename = img_id.split("::")[0]
            else:
                filename = img_id
            
            if filename not in ocr_texts_by_filename:
                ocr_texts_by_filename[filename] = []
            ocr_texts_by_filename[filename].append(ocr_text)
        
        # Convert lists to single strings for single images, keep combined for multiple
        for filename, texts in ocr_texts_by_filename.items():
            if len(texts) == 1:
                ocr_texts_by_filename[filename] = texts[0]
            else:
                # For multiple images (PDFs), combine with separators
                ocr_texts_by_filename[filename] = " | ".join([f"Image {i+1}: {text}" for i, text in enumerate(texts)])
        
        # Return result with all data
        result = {
            "text_chunks": len(text_ids),
            "images": len(img_ids),
            "facts": len(facts_to_insert),
            "image_captions": image_captions_by_filename,  # Keyed by filename for app.py compatibility
            "audio_transcripts": audio_transcripts,
            "ocr_texts": ocr_texts_by_filename,  # Keyed by filename for app.py compatibility
            "pdf_texts": pdf_texts  # Keyed by filename for sentiment analysis
        }
        logger.info(f"📤 Returning result - Transcripts and PDF texts ready for sentiment analysis")
        return result

    # --------------- Retrieve ---------------
    def retrieve(self, query: str, top_k: int) -> Dict[str, Any]:
        # Optimize: Limit top_k for faster retrieval
        top_k = min(top_k, 5)
        
        text_q = self.text_embedder.embed_query(query)
        text_hits = self.index.query_texts(text_q, k=top_k)

        # Skip BLIP image search for speed - BLIP captions are already in text chunks
        # This eliminates the slow BLIP embedding step during retrieval
        image_hits = {"ids": [[]], "distances": [[]], "documents": [[]], "metadatas": [[]]}
        logger.debug(f"Retrieved {len(text_hits.get('ids', [[]])[0])} text result(s) (skipping separate image search for speed)")

        def fmt(res, modality: str):
            out = []
            for i in range(len(res.get("ids", [[]])[0])):
                out.append({
                    "id": res["ids"][0][i],
                    "score": float(res["distances"][0][i]) if "distances" in res else None,
                    "document": res.get("documents", [[None]])[0][i],
                    "metadata": res.get("metadatas", [[{}]])[0][i] | {"modality": modality},
                })
            return out

        combined = fmt(text_hits, "text") + fmt(image_hits, "image")
        
        # For each retrieved image, find its BLIP caption from text index (optimized - only check in combined results)
        # BLIP captions are already in text chunks, so they should be in combined results
        for r in combined:
            metadata = r.get("metadata") or {}
            if metadata.get("modality") == "image":
                source = metadata.get("source")
                # Look for BLIP caption in the same combined results (faster than separate query)
                for r2 in combined:
                    metadata2 = r2.get("metadata") or {}
                    if (metadata2.get("doc_type") == "image_caption" and 
                        metadata2.get("source") == source):
                        caption_text = r2.get("document", "")
                        # Extract the actual caption from the formatted text
                        if "[Image Description from BLIP]:" in caption_text:
                            caption_text = caption_text.split("[Image Description from BLIP]:", 1)[1].strip()
                        metadata["blip_caption"] = caption_text
                        break
        
        # Dedup by source, keep best score
        by_src: Dict[str, Dict[str, Any]] = {}
        for r in combined:
            src = (r.get("metadata") or {}).get("source") or r["id"]
            if src not in by_src or (r["score"] is not None and r["score"] < by_src[src].get("score", 1e9)):
                by_src[src] = r
        ranked = sorted(by_src.values(), key=lambda x: x.get("score", 0.0))[:top_k]
        return {"results": ranked}

    # --------------- Generate ---------------
    def answer(self, query: str, contexts: List[Dict[str, Any]], is_sentiment_analysis: bool = False) -> Dict[str, str]:
        """
        Generate answer from query and contexts.
        
        Args:
            query: The query string (full text for sentiment analysis)
            contexts: Retrieved contexts from vector DB (empty for direct sentiment analysis)
            is_sentiment_analysis: If True, this is a direct sentiment analysis query (skip retrieval was used)
        """
        if self.llm is None:
            return {
                "main_response": "LLM not configured. Please set up your API key in the .env file to enable question answering.",
                "sentiment_analysis": ""
            }
        
        # For direct sentiment analysis (skip_retrieval=True), use optimized prompt
        if is_sentiment_analysis:
            logger.info(f"⚡ Direct sentiment analysis mode - processing {len(query)} characters of text")
            
            # Load system prompt
            system_prompt = load_system_prompt()
            
            # For sentiment analysis, send the FULL text without truncation
            # Create optimized prompt for sentiment analysis that works for ANY content
            prompt = f"""{system_prompt}

## Text for Sentiment Analysis
Please analyze the following text for sentiment, emotional tone, and emotional indicators. Provide a comprehensive sentiment analysis regardless of the topic or content type.

Text to analyze:
{query}

## Your Response
Please provide:
1. Main Response: A summary of the text content and its key themes
2. Sentiment Analysis: A detailed sentiment analysis including:
   - Overall emotional tone (Positive/Neutral/Negative)
   - Key emotional indicators and their intensity
   - Emotional context and nuances
   - Notable emotional patterns or shifts
   - Mental health indicators (if applicable to the content)
   - Risk factors (if any concerning emotional patterns are detected)
   - Recommendations (if appropriate and relevant)

Note: Analyze sentiment for ANY type of content - whether it's depression-related, general conversation, academic text, or any other topic. The sentiment analysis should focus on the emotional tone and indicators present in the text.

Format your response with "--- SENTIMENT ANALYSIS ---" separating the main response from the detailed sentiment analysis."""
            
            logger.info(f"📤 Sending full text ({len(query)} chars) to LLM for sentiment analysis")
            
            try:
                # Use higher max_tokens for comprehensive sentiment analysis
                resp, usage_info = self.llm.generate_content(prompt, max_tokens=3000)
                logger.info(f"✅ Sentiment analysis generated - {usage_info.get('total_tokens', 0)} tokens used")
                
                # Parse the response to separate main content from sentiment analysis
                if "--- SENTIMENT ANALYSIS ---" in resp:
                    parts = resp.split("--- SENTIMENT ANALYSIS ---")
                    main_response = parts[0].strip()
                    sentiment_analysis = parts[1].strip() if len(parts) > 1 else ""
                else:
                    main_response = resp
                    sentiment_analysis = ""
                
                return {
                    "main_response": main_response,
                    "sentiment_analysis": sentiment_analysis
                }
            except Exception as e:
                logger.error(f"Sentiment analysis generation failed: {str(e)}")
                return {
                    "main_response": f"Sorry, I encountered an error while generating the sentiment analysis: {str(e)}",
                    "sentiment_analysis": ""
                }
        
        # Normal RAG flow (with contexts from vector DB)
        # Check for crisis situations first
        if is_crisis_query(query):
            logger.warning("Crisis query detected - providing emergency response with sentiment analysis")
            
            # Always perform sentiment analysis on crisis queries to understand emotional state
            # This helps provide comprehensive support even in crisis situations
            system_prompt = load_system_prompt()
            
            # Create a prompt that analyzes sentiment for crisis situations
            crisis_sentiment_prompt = f"""{system_prompt}

## Text for Sentiment Analysis (CRISIS SITUATION)
Please analyze the following text for sentiment, emotional tone, and emotional indicators. This is a CRISIS SITUATION that requires immediate attention, but sentiment analysis is still important to understand the emotional state.

Text to analyze:
{query}

## Your Response
Please provide:
1. Main Response: Acknowledgment that this is a crisis situation requiring immediate help
2. Sentiment Analysis: A detailed and sensitive sentiment analysis including:
   - Overall emotional tone (likely Negative/Critical)
   - Key emotional indicators and their severity
   - Emotional context and depth of distress
   - Notable emotional patterns (hopelessness, despair, etc.)
   - Risk level assessment based on emotional indicators
   - Emotional support recommendations

IMPORTANT: This is a crisis situation. The sentiment analysis should be sensitive, comprehensive, and help understand the depth of emotional distress while emphasizing the importance of immediate professional help.

Format your response with "--- SENTIMENT ANALYSIS ---" separating the main response from the detailed sentiment analysis."""
            
            try:
                logger.info(f"📤 Analyzing sentiment for crisis query ({len(query)} chars)")
                resp, usage_info = self.llm.generate_content(crisis_sentiment_prompt, max_tokens=2000)
                logger.info(f"✅ Sentiment analysis generated for crisis query - {usage_info.get('total_tokens', 0)} tokens used")
                
                # Parse the response
                if "--- SENTIMENT ANALYSIS ---" in resp:
                    parts = resp.split("--- SENTIMENT ANALYSIS ---")
                    main_response = parts[0].strip()
                    sentiment_analysis = parts[1].strip() if len(parts) > 1 else ""
                else:
                    main_response = resp
                    sentiment_analysis = ""
                
                # Prepend the crisis response with emergency resources
                crisis_msg = get_crisis_response()
                main_response = f"{crisis_msg}\n\n{main_response}"
                
                return {
                    "main_response": main_response,
                    "sentiment_analysis": sentiment_analysis
                }
            except Exception as e:
                logger.error(f"Sentiment analysis for crisis query failed: {str(e)}")
                # Fallback to basic crisis response (still provide help even if sentiment analysis fails)
                return {
                    "main_response": get_crisis_response(),
                    "sentiment_analysis": ""
                }
        
        # Check for off-topic questions
        if is_off_topic(query):
            logger.info("Off-topic query detected - providing redirection response with sentiment analysis")
            
            # Always perform sentiment analysis on the actual user query, even if off-topic
            # This ensures sentiment is analyzed for all content, not just depression-related
            system_prompt = load_system_prompt()
            
            # Create a prompt that analyzes sentiment for any content (not just depression-related)
            sentiment_prompt = f"""{system_prompt}

## Text for Sentiment Analysis
Please analyze the following text for sentiment, emotional tone, and emotional indicators. Provide a comprehensive sentiment analysis regardless of the topic.

Text to analyze:
{query}

## Your Response
Please provide:
1. Main Response: A brief acknowledgment that this topic is outside my specialization, but I can still analyze the sentiment
2. Sentiment Analysis: A detailed sentiment analysis including:
   - Overall emotional tone
   - Key emotional indicators
   - Emotional context and nuances
   - Any notable emotional patterns

Format your response with "--- SENTIMENT ANALYSIS ---" separating the main response from the detailed sentiment analysis."""
            
            try:
                logger.info(f"📤 Analyzing sentiment for off-topic query ({len(query)} chars)")
                resp, usage_info = self.llm.generate_content(sentiment_prompt, max_tokens=2000)
                logger.info(f"✅ Sentiment analysis generated for off-topic query - {usage_info.get('total_tokens', 0)} tokens used")
                
                # Parse the response
                if "--- SENTIMENT ANALYSIS ---" in resp:
                    parts = resp.split("--- SENTIMENT ANALYSIS ---")
                    main_response = parts[0].strip()
                    sentiment_analysis = parts[1].strip() if len(parts) > 1 else ""
                else:
                    main_response = resp
                    sentiment_analysis = ""
                
                # Prepend the redirect message to the main response
                redirect_msg = get_off_topic_response(query).split("--- SENTIMENT ANALYSIS ---")[0].strip()
                main_response = f"{redirect_msg}\n\n{main_response}"
                
                return {
                    "main_response": main_response,
                    "sentiment_analysis": sentiment_analysis
                }
            except Exception as e:
                logger.error(f"Sentiment analysis for off-topic query failed: {str(e)}")
                # Fallback to basic response
                off_topic_response = get_off_topic_response(query)
                if "--- SENTIMENT ANALYSIS ---" in off_topic_response:
                    parts = off_topic_response.split("--- SENTIMENT ANALYSIS ---")
                    return {
                        "main_response": parts[0].strip(),
                        "sentiment_analysis": parts[1].strip() if len(parts) > 1 else ""
                    }
                else:
                    return {
                        "main_response": off_topic_response,
                        "sentiment_analysis": ""
                    }
        
        logger.info("Generating answer using OpenAI for depression-related query")
        
        # Load system prompt
        system_prompt = load_system_prompt()
        
        # Build context from retrieved documents - limit to top 2 most relevant for faster response
        # This speeds up LLM response by reducing context size
        max_contexts = min(len(contexts), 2)  # Reduced from 3 to 2 for faster processing
        contexts = contexts[:max_contexts]
        
        ctx_blocks = []
        logger.debug(f"Building context from {len(contexts)} retrieved documents for LLM (limited to top {max_contexts})...")
        
        for c in contexts:
            src = (c.get("metadata") or {}).get("source", "")
            text = c.get("document") or ""
            metadata = c.get("metadata") or {}
            
            # Limit text length to prevent overly long contexts (max 400 chars per document for speed)
            if len(text) > 400:
                text = text[:400] + "..."
            
            if metadata.get("modality") == "image":
                # For images, use BLIP caption from metadata (already fetched in retrieve)
                image_source = metadata.get("source", "")
                blip_caption = metadata.get("blip_caption")
                
                if blip_caption:
                    # Clean the caption if it has prefix
                    if "[Image Description from BLIP]:" in blip_caption:
                        blip_caption = blip_caption.split("[Image Description from BLIP]:", 1)[1].strip()
                    text = f"[Image from {image_source}]: {blip_caption}"
                else:
                    # Use document text if available, otherwise minimal description
                    if text and "[Image Description from BLIP]:" in text:
                        blip_caption = text.split("[Image Description from BLIP]:", 1)[1].strip()
                        text = f"[Image from {image_source}]: {blip_caption}"
                    else:
                        text = f"[Image from {image_source} - visual content]"
            
            ctx_blocks.append(f"[Source: {src}]\n{text}")
        
        # Create the comprehensive prompt for OpenAI - simplified for faster processing
        context_section = f"\n\nRelevant Context:\n{chr(10).join(ctx_blocks)}" if ctx_blocks else ""
        
        prompt = f"""{system_prompt}

## Current Query
{query}
{context_section}

## Your Response
Please respond according to the protocols outlined above:"""
        
        # Minimal logging for speed
        logger.debug(f"Sending query to LLM with {len(ctx_blocks)} context document(s), total prompt length: {len(prompt)} chars")
        
        try:
            # Generate response with optimized prompt
            resp, usage_info = self.llm.generate_content(prompt)
            logger.debug(f"Answer generated successfully - {usage_info.get('total_tokens', 0)} tokens used")
            
            # Parse the response to separate main content from sentiment analysis
            if "--- SENTIMENT ANALYSIS ---" in resp:
                parts = resp.split("--- SENTIMENT ANALYSIS ---")
                main_response = parts[0].strip()
                sentiment_analysis = parts[1].strip() if len(parts) > 1 else ""
            else:
                main_response = resp
                sentiment_analysis = ""
            
            return {
                "main_response": main_response,
                "sentiment_analysis": sentiment_analysis
            }
        except Exception as e:
            logger.error(f"Answer generation failed: {str(e)}")
            return {
                "main_response": f"Sorry, I encountered an error while generating the answer: {str(e)}",
                "sentiment_analysis": ""
            }

    def query(self, query: str, k: int, skip_retrieval: bool = False) -> Dict[str, Any]:
        """
        Query the RAG pipeline.
        
        Args:
            query: The query string
            k: Number of results to retrieve
            skip_retrieval: If True, skip vector DB retrieval and go directly to sentiment analysis
                           (Useful for OCR text/audio transcripts that don't need retrieval)
        """
        if skip_retrieval:
            # Direct sentiment analysis - no vector DB retrieval needed
            # This is much faster for OCR text and audio transcripts
            logger.info(f"⚡ Skipping vector DB retrieval - direct sentiment analysis for {len(query)} characters")
            answer_data = self.answer(query, [], is_sentiment_analysis=True)  # Pass flag for sentiment analysis
            return {
                "main_response": answer_data["main_response"],
                "sentiment_analysis": answer_data["sentiment_analysis"],
                "contexts": []  # No contexts retrieved
            }
        else:
            # Normal RAG flow: retrieve from vector DB, then answer
            retrieved = self.retrieve(query, k)
            answer_data = self.answer(query, retrieved["results"], is_sentiment_analysis=False)
            return {
                "main_response": answer_data["main_response"],
                "sentiment_analysis": answer_data["sentiment_analysis"],
                "contexts": retrieved["results"]
            }

    def reset(self):
        self.index.reset()
        with sqlite3.connect(self.sqlite_path) as con:
            con.execute("DELETE FROM facts")
            con.commit()
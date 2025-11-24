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
from extractors import partition_any, elements_to_text_chunks, transcribe_audio, extract_numeric_facts, Chunk
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
    def _process_single_file(self, p: str) -> Tuple[List[Chunk], Dict[str, Any], List[Tuple[str, str, float]], str, str]:
        """Process a single file and return chunks, image data, facts, image caption, and audio transcript.
        Returns: (chunks, image_data, facts, image_caption, audio_transcript)
        - chunks: List of text chunks
        - image_data: Dict with 'id', 'meta', 'emb' if image, else None
        - facts: List of numeric facts
        - image_caption: BLIP caption string if image, else None
        - audio_transcript: Transcript text if audio file, else None
        """
        ext = Path(p).suffix.lower().lstrip(".")
        source_id = os.path.basename(p)
        chunks = []
        image_data = None
        facts = []
        image_caption = None
        audio_transcript = None

        logger.info(f"🚀 Processing file: {source_id} ({ext})")

        try:
            if ext in {"mp3", "wav", "m4a", "flac", "ogg"}:
                # Transcribe audio to text
                logger.info(f"=== Transcribing audio file: {source_id} ===")
                try:
                    transcript = transcribe_audio(
                        p, 
                        backend=os.getenv("OPENAI_WHISPER", settings.openai_whisper),
                        model_size=settings.whisper_model_size
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
                        
                        # Chunk the transcript like text files for better retrieval
                        logger.info(f"Starting to chunk audio transcript for {source_id}...")
                        try:
                            class DummyElement:
                                def __init__(self, text):
                                    self.text = text
                            
                            logger.info(f"Creating DummyElement with transcript length: {len(transcript)}")
                            dummy_element = DummyElement(transcript)
                            logger.info(f"DummyElement created, accessing .text attribute...")
                            test_text = dummy_element.text
                            logger.info(f"DummyElement.text accessible, length: {len(test_text)}")
                            
                            logger.info(f"Calling elements_to_text_chunks...")
                            transcript_chunks = elements_to_text_chunks(
                                [dummy_element], 
                                source=source_id,
                                chunk_size=1000,
                                overlap=150
                            )
                            logger.info(f"Chunking completed, got {len(transcript_chunks)} chunk(s)")
                            
                            for chunk in transcript_chunks:
                                chunk.doc_type = "audio"
                            chunks = transcript_chunks
                            audio_transcript = transcript  # Store full transcript for return
                            logger.info(f"✅ Audio transcript split into {len(chunks)} chunk(s) - ready for embedding")
                        except Exception as chunk_error:
                            logger.error(f"❌ Error chunking audio transcript: {str(chunk_error)}", exc_info=True)
                            chunks = []
                            audio_transcript = transcript  # Still store transcript even if chunking fails
                    else:
                        logger.warning(f"⚠️ Audio transcription returned empty result for {source_id}")
                        chunks = []
                        audio_transcript = None
                    
            elif ext in {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}:
                # Image processing
                try:
                    from PIL import Image
                    with Image.open(p) as img:
                        img.verify()
                    
                    # Generate image embedding (with thread lock for safety)
                    with self._blip_embedder_lock:
                        img_emb = self.blip_embedder.embed_images([p])[0]
                    image_data = {
                        "id": f"{source_id}::image",
                        "meta": {"source": source_id, "image_path": os.path.abspath(p)},
                        "emb": img_emb
                    }
                    
                    # Generate BLIP caption (with thread lock for safety)
                    # Note: Timeout is handled at the file processing level
                    logger.info(f"=== Generating BLIP caption for image: {source_id} ===")
                    try:
                        with self._blip_embedder_lock:
                            blip_caption = self.blip_embedder.generate_caption(p)
                        
                        logger.info(f"=== BLIP CAPTION GENERATED ===")
                        logger.info(f"Image: {source_id}")
                        logger.info(f"Full Caption: {blip_caption}")
                        logger.info(f"=== End of BLIP Caption ===")
                        
                        image_caption = blip_caption
                        blip_chunk = Chunk(
                            id=f"{source_id}::blip_caption",
                            text=f"[Image Description from BLIP]: {blip_caption}",
                            source=source_id,
                            doc_type="image_caption"
                        )
                        chunks = [blip_chunk]
                        logger.info(f"BLIP caption stored as text chunk")
                    except Exception as caption_error:
                        logger.error(f"Error generating BLIP caption for {p}: {str(caption_error)}", exc_info=True)
                        # Continue without caption - image embedding is still stored
                        chunks = []
                        image_caption = None
                        
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
                        image_data = None
                
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
            if image_data is None:
                image_data = None
            if image_caption is None:
                image_caption = None
            if audio_transcript is None:
                audio_transcript = None
        
        logger.info(f"🔚 Returning from _process_single_file for {source_id}: {len(chunks)} chunks, {len(facts)} facts, audio_transcript={'present' if audio_transcript else 'None'}")
        return chunks, image_data, facts, image_caption, audio_transcript

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
                    chunks, image_data, facts, image_caption, audio_transcript = future.result(timeout=max_file_time)
                    logger.info(f"✅ Got result from {os.path.basename(p)}: {len(chunks)} chunks, audio_transcript={'present' if audio_transcript else 'None'}")
                    
                    # Collect chunks for batch embedding later
                    if chunks:
                        all_chunks.extend(chunks)
                        logger.info(f"Collected {len(chunks)} chunk(s) from {os.path.basename(p)}")
                    
                    # Collect image data
                    if image_data:
                        all_image_data.append((image_data, p, image_caption))
                        logger.info(f"Collected image data from {os.path.basename(p)}")
                    
                    # Collect facts
                    if facts:
                        all_facts.extend(facts)
                        logger.info(f"Collected {len(facts)} fact(s) from {os.path.basename(p)}")
                    
                    # Store image captions
                    if image_caption:
                        source_id = os.path.basename(p)
                        image_captions[source_id] = image_caption
                    
                    # Store audio transcripts
                    if audio_transcript:
                        source_id = os.path.basename(p)
                        audio_transcripts[source_id] = audio_transcript
                    
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
        
        # Now batch process all collected data
        logger.info(f"Batch processing {len(all_chunks)} text chunk(s) and {len(all_image_data)} image(s)...")
        
        # Batch embed all text chunks at once (much faster than one-by-one)
        # NOTE: Text chunks (from .txt files, audio transcripts, or BLIP captions) use TextEmbedder, NOT BLIP
        text_ids, text_docs, text_meta, text_embs = [], [], [], []
        if all_chunks:
            text_ids = [c.id for c in all_chunks]
            text_docs = [c.text for c in all_chunks]
            text_meta = [{"source": c.source, "doc_type": c.doc_type} for c in all_chunks]
            logger.info(f"📝 Generating text embeddings for {len(text_docs)} chunk(s) using TextEmbedder (NOT BLIP)...")
            text_embs = self.text_embedder.embed_texts(text_docs)  # Batch embedding - no lock needed in main thread
            logger.info(f"✅ Text embeddings generated successfully (using TextEmbedder, not BLIP)")
        
        # Process all image data
        img_ids, img_meta, img_embs = [], [], []
        for image_data, p, image_caption in all_image_data:
            img_ids.append(image_data["id"])
            img_meta.append(image_data["meta"])
            img_embs.append(image_data["emb"])
        
        facts_to_insert = all_facts

        # Batch insert all data at once
        logger.info(f"Storing processed data in vector database...")
        if text_ids:
            logger.info(f"Adding {len(text_ids)} text chunk(s) to vector database...")
            self.index.add_texts(text_ids, text_docs, text_meta, text_embs)
        if img_ids:
            logger.info(f"Adding {len(img_ids)} image(s) to vector database...")
            self.index.add_images(img_ids, img_meta, img_embs)
        if facts_to_insert:
            logger.info(f"Storing {len(facts_to_insert)} numeric fact(s) in database...")
            with sqlite3.connect(self.sqlite_path) as con:
                con.executemany("INSERT OR IGNORE INTO facts (source, key, value) VALUES (?, ?, ?)", facts_to_insert)
                con.commit()
        
        logger.info(f"✅ Ingestion completed - Text chunks: {len(text_ids)}, Images: {len(img_ids)}, Facts: {len(facts_to_insert)}")
        logger.info(f"📊 Returning audio_transcripts: {list(audio_transcripts.keys())} with {sum(1 for t in audio_transcripts.values() if t)} non-empty transcript(s)")
        
        # Return ingestion results including image captions and audio transcripts
        result = {
            "text_chunks": len(text_ids),
            "images": len(img_ids),
            "facts": len(facts_to_insert),
            "image_captions": image_captions,
            "audio_transcripts": audio_transcripts
        }
        logger.info(f"📤 Returning ingestion result with {len(audio_transcripts)} audio transcript(s)")
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
    def answer(self, query: str, contexts: List[Dict[str, Any]]) -> Dict[str, str]:
        if self.llm is None:
            return {
                "main_response": "LLM not configured. Please set up your API key in the .env file to enable question answering.",
                "sentiment_analysis": ""
            }
        
        # Check for crisis situations first
        if is_crisis_query(query):
            logger.warning("Crisis query detected - providing emergency response")
            return {
                "main_response": get_crisis_response(),
                "sentiment_analysis": ""
            }
        
        # Check for off-topic questions
        if is_off_topic(query):
            logger.info("Off-topic query detected - providing redirection response")
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
        
        # Build context from retrieved documents - limit to top 3 most relevant
        # This speeds up LLM response by reducing context size
        max_contexts = min(len(contexts), 3)  # Limit to top 3 contexts for faster response
        contexts = contexts[:max_contexts]
        
        ctx_blocks = []
        logger.debug(f"Building context from {len(contexts)} retrieved documents for LLM (limited to top {max_contexts})...")
        
        for c in contexts:
            src = (c.get("metadata") or {}).get("source", "")
            text = c.get("document") or ""
            metadata = c.get("metadata") or {}
            
            # Limit text length to prevent overly long contexts (max 500 chars per document)
            if len(text) > 500:
                text = text[:500] + "..."
            
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
        
        # Log prompt summary (reduced logging for speed)
        logger.info(f"Sending query to LLM with {len(ctx_blocks)} context document(s), total prompt length: {len(prompt)} chars")
        
        try:
            # Generate response with optimized prompt
            resp, usage_info = self.llm.generate_content(prompt)
            logger.info("Answer generated successfully")
            
            # Log token usage summary at pipeline level
            logger.info(f"=== LLM Call Summary ===")
            logger.info(f"Query: {query[:100]}..." if len(query) > 100 else f"Query: {query}")
            logger.info(f"Context Documents Used: {len(contexts)}")
            logger.info(f"Token Usage - Input: {usage_info.get('prompt_tokens', 0)}, Output: {usage_info.get('completion_tokens', 0)}, Total: {usage_info.get('total_tokens', 0)}")
            logger.info(f"=== End LLM Call Summary ===")
            
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

    def query(self, query: str, k: int) -> Dict[str, Any]:
        retrieved = self.retrieve(query, k)
        answer_data = self.answer(query, retrieved["results"])
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
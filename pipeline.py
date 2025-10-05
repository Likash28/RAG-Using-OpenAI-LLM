from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import os
import sqlite3
from PIL import Image

from config import settings
from logging_config import get_logger
from embedder import TextEmbedder, CLIPMultimodal
from vectorstore import DualIndex
from extractors import partition_any, elements_to_text_chunks, transcribe_audio, extract_numeric_facts, Chunk
from gemini_llm import GeminiLLM
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
        self.clip_embedder = CLIPMultimodal(settings.clip_model_name)
        logger.info("CLIP embedder initialized")
        self.index = DualIndex(settings.chroma_dir)
        logger.info("Vector index initialized")
        self.sqlite_path = settings.sqlite_path
        self._ensure_sqlite()
        self.llm = self._init_llm()
        logger.info("RAG Pipeline initialization completed")

    def _ensure_sqlite(self):
        os.makedirs(os.path.dirname(self.sqlite_path) or ".", exist_ok=True)
        with sqlite3.connect(self.sqlite_path) as con:
            con.execute(
                "CREATE TABLE IF NOT EXISTS facts (source TEXT, key TEXT, value REAL, PRIMARY KEY(source, key, value))"
            )
            con.commit()

    def _init_llm(self):
        if settings.provider == "gemini" and settings.gemini_api_key:
            logger.info("Initializing Gemini LLM")
            return GeminiLLM(api_key=settings.gemini_api_key, model_name=settings.gemini_model)
        # Return None if no valid API key is provided
        logger.warning("No valid LLM provider configured")
        return None

    # --------------- Ingest ---------------
    def ingest_paths(self, paths: List[str]):
        logger.info(f"Starting ingestion of {len(paths)} files")
        text_ids, text_docs, text_meta, text_embs = [], [], [], []
        img_ids, img_meta, img_embs = [], [], []
        facts_to_insert = []

        for p in paths:
            logger.info(f"Processing file: {p}")
            ext = Path(p).suffix.lower().lstrip(".")
            source_id = os.path.basename(p)

            if ext in {"mp3", "wav", "m4a", "flac", "ogg"}:
                transcript = transcribe_audio(p, backend=os.getenv("OPENAI_WHISPER", settings.openai_whisper))
                if transcript:
                    chunks = [Chunk(id=f"{source_id}::audio::0", text=transcript, source=source_id, doc_type="audio")]
                else:
                    chunks = []
            elif ext in {"png", "jpg", "jpeg", "webp", "bmp", "tiff"}:
                # Image goes to vision index (CLIP) and also try OCR via unstructured
                try:
                    img_emb = self.clip_embedder.embed_images([p])[0]
                    img_ids.append(f"{source_id}::image")
                    img_meta.append({"source": source_id, "image_path": os.path.abspath(p)})
                    img_embs.append(img_emb)
                except Exception:
                    pass
                elements = partition_any(p)
                chunks = elements_to_text_chunks(elements, source=source_id)
            else:
                elements = partition_any(p)
                chunks = elements_to_text_chunks(elements, source=source_id)

            if not chunks:
                continue
            # Add text chunks
            ids = [c.id for c in chunks]
            docs = [c.text for c in chunks]
            metas = [{"source": c.source, "doc_type": c.doc_type} for c in chunks]
            embs = self.text_embedder.embed_texts(docs)

            text_ids.extend(ids)
            text_docs.extend(docs)
            text_meta.extend(metas)
            text_embs.extend(embs)

            # Numeric facts
            facts = extract_numeric_facts(chunks)
            facts_to_insert.extend(facts)

        if text_ids:
            self.index.add_texts(text_ids, text_docs, text_meta, text_embs)
        if img_ids:
            self.index.add_images(img_ids, img_meta, img_embs)
        if facts_to_insert:
            with sqlite3.connect(self.sqlite_path) as con:
                con.executemany("INSERT OR IGNORE INTO facts (source, key, value) VALUES (?, ?, ?)", facts_to_insert)
                con.commit()
        
        logger.info(f"Ingestion completed - Text chunks: {len(text_ids)}, Images: {len(img_ids)}, Facts: {len(facts_to_insert)}")

    # --------------- Retrieve ---------------
    def retrieve(self, query: str, top_k: int) -> Dict[str, Any]:
        text_q = self.text_embedder.embed_query(query)
        text_hits = self.index.query_texts(text_q, k=top_k)

        clip_q = self.clip_embedder.embed_text(query)
        image_hits = self.index.query_images_with_text(clip_q, k=max(2, top_k//2))

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
        
        logger.info("Generating answer using Gemini for depression-related query")
        
        # Load system prompt
        system_prompt = load_system_prompt()
        
        # Build context from retrieved documents
        ctx_blocks = []
        for c in contexts:
            src = (c.get("metadata") or {}).get("source", "")
            text = c.get("document") or ""
            if (c.get("metadata") or {}).get("modality") == "image":
                # For images, we only have a path; mention it as figure source
                text = f"[Figure available at {text}]"
            ctx_blocks.append(f"[Source: {src}]\n{text}")
        
        # Create the comprehensive prompt for Gemini
        context_section = f"\n\nContext from uploaded documents (use only if relevant):\n{chr(10).join(ctx_blocks)}" if ctx_blocks else ""
        
        prompt = f"""{system_prompt}

## Current Query
{query}
{context_section}

## Your Response
Please respond according to the protocols outlined above:"""
        
        try:
            resp = self.llm.generate_content(prompt)
            logger.info("Answer generated successfully")
            
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
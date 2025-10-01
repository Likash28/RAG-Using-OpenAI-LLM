from typing import List, Dict, Any, Optional
import os
import chromadb
from chromadb.config import Settings as ChromaSettings

class DualIndex:
    """Two Chroma collections: text_docs and image_docs."""
    def __init__(self, persist_dir: str):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        # No internal embedding function: we pass embeddings explicitly
        self.text = self.client.get_or_create_collection("text_docs", metadata={"hnsw:space":"cosine"})
        self.vision = self.client.get_or_create_collection("image_docs", metadata={"hnsw:space":"cosine"})

    # TEXT ops
    def add_texts(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
        self.text.add(ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings)

    def query_texts(self, query_embedding: List[float], k: int):
        return self.text.query(query_embeddings=[query_embedding], n_results=k)

    # IMAGE ops
    def add_images(self, ids: List[str], metadatas: List[Dict[str, Any]], embeddings: List[List[float]]):
        # store dummy documents as file paths if available in metadata
        docs = [m.get("image_path", "[image]") for m in metadatas]
        self.vision.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)

    def query_images_with_text(self, clip_text_embedding: List[float], k: int):
        return self.vision.query(query_embeddings=[clip_text_embedding], n_results=k)

    def reset(self):
        self.client.delete_collection("text_docs")
        self.client.delete_collection("image_docs")
        self.text = self.client.get_or_create_collection("text_docs", metadata={"hnsw:space":"cosine"})
        self.vision = self.client.get_or_create_collection("image_docs", metadata={"hnsw:space":"cosine"})

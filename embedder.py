from typing import List, Union
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np

class TextEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embs.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

class CLIPMultimodal:
    """Cross-modal CLIP. Can encode images and text into the same space."""
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_images(self, images: List[Union[str, Image.Image]]) -> List[List[float]]:
        pil = []
        for it in images:
            if isinstance(it, Image.Image):
                pil.append(it)
            else:
                pil.append(Image.open(it).convert("RGB"))
        embs = self.model.encode(pil, normalize_embeddings=True, convert_to_numpy=True)
        return embs.tolist()

    def embed_text(self, text: str) -> List[float]:
        embs = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)
        return embs[0].tolist()
from typing import List, Union, Optional
from PIL import Image
from sentence_transformers import SentenceTransformer
import numpy as np
import torch
from transformers import BlipProcessor, BlipModel, BlipForConditionalGeneration
from logging_config import get_logger

logger = get_logger("BLIPEmbedder")

class TextEmbedder:
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        embs = self.model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return embs.tolist()

    def embed_query(self, text: str) -> List[float]:
        return self.embed_texts([text])[0]

class BLIPMultimodal:
    """Cross-modal BLIP. Can encode images and text into the same space, and generate image captions."""
    def __init__(self, model_name: str = "Salesforce/blip-itm-base-coco"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipModel.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Load captioning model for generating text from images
        # Use image-captioning model for better caption generation
        try:
            caption_model_name = "Salesforce/blip-image-captioning-base"
            logger.info(f"Loading BLIP captioning model: {caption_model_name}")
            self.caption_processor = BlipProcessor.from_pretrained(caption_model_name)
            self.caption_model = BlipForConditionalGeneration.from_pretrained(caption_model_name).to(self.device)
            self.caption_model.eval()
            logger.info("BLIP captioning model loaded successfully")
        except Exception as e:
            # Fallback: use the same model for captioning if separate model fails
            logger.warning(f"Failed to load BLIP captioning model: {str(e)}")
            logger.warning("Falling back to ITM model (caption generation will be limited)")
            self.caption_processor = self.processor
            self.caption_model = None

    def embed_images(self, images: List[Union[str, Image.Image]]) -> List[List[float]]:
        """Embed images using BLIP's vision encoder."""
        pil_images = []
        for it in images:
            if isinstance(it, Image.Image):
                pil_images.append(it.convert("RGB"))
            else:
                pil_images.append(Image.open(it).convert("RGB"))
        
        embeddings = []
        with torch.no_grad():
            for img in pil_images:
                inputs = self.processor(images=img, return_tensors="pt").to(self.device)
                # Extract image features from vision model
                vision_outputs = self.model.vision_model(**inputs)
                image_features = vision_outputs.last_hidden_state
                # Use [CLS] token (first token) or mean pooling
                if len(image_features.shape) > 2:
                    # Use [CLS] token (first token)
                    image_features = image_features[:, 0, :]
                # Normalize embeddings
                image_features = image_features / image_features.norm(dim=-1, keepdim=True)
                embeddings.append(image_features.cpu().numpy().flatten().tolist())
        
        return embeddings

    def embed_text(self, text: str) -> List[float]:
        """Embed text using BLIP's text encoder."""
        try:
            with torch.no_grad():
                inputs = self.processor(text=text, return_tensors="pt", padding=True, truncation=True).to(self.device)
                
                # For BLIP ITM models, access the Q-Former which processes text
                # The Q-Former is typically accessible through the model
                text_features = None
                
                # Try method 1: Access Q-Former directly if available
                if hasattr(self.model, 'qformer'):
                    qformer_outputs = self.model.qformer(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None)
                    )
                    text_features = qformer_outputs.last_hidden_state
                    logger.debug("Extracted text features from Q-Former")
                
                # Try method 2: Use forward pass with dummy image (ITM models need both)
                if text_features is None:
                    dummy_image = Image.new('RGB', (224, 224), color='black')
                    image_inputs = self.processor(images=dummy_image, return_tensors="pt").to(self.device)
                    
                    outputs = self.model(
                        input_ids=inputs["input_ids"],
                        attention_mask=inputs.get("attention_mask", None),
                        pixel_values=image_inputs["pixel_values"]
                    )
                    
                    # Extract from Q-Former outputs in the model
                    if hasattr(outputs, 'qformer_outputs') and hasattr(outputs.qformer_outputs, 'last_hidden_state'):
                        text_features = outputs.qformer_outputs.last_hidden_state
                        logger.debug("Extracted text features from model outputs (Q-Former)")
                    elif hasattr(self.model, 'qformer'):
                        # Try accessing Q-Former through model
                        qformer_outputs = self.model.qformer(
                            input_ids=inputs["input_ids"],
                            attention_mask=inputs.get("attention_mask", None)
                        )
                        text_features = qformer_outputs.last_hidden_state
                        logger.debug("Extracted text features via model.qformer")
                
                if text_features is None:
                    raise ValueError("Could not extract text features from BLIP ITM model. Model structure may be different.")
                
                # Use [CLS] token (first token) or mean pooling
                if len(text_features.shape) > 2:
                    # Use [CLS] token (first token)
                    text_features = text_features[:, 0, :]
                
                # Normalize embeddings
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
                embedding = text_features.cpu().numpy().flatten().tolist()
                logger.debug(f"BLIP text embedding dimension: {len(embedding)}")
                return embedding
        except Exception as e:
            logger.error(f"Error embedding text with BLIP: {str(e)}", exc_info=True)
            # Re-raise to let pipeline handle fallback
            raise
    
    def generate_caption(self, image: Union[str, Image.Image]) -> str:
        """Generate a text caption from an image using BLIP."""
        try:
            # Load image if path provided
            if isinstance(image, str):
                img = Image.open(image).convert("RGB")
            else:
                img = image.convert("RGB")
            
            # Use captioning model if available
            if self.caption_model is not None:
                logger.info("Using BLIP captioning model to generate caption...")
                inputs = self.caption_processor(images=img, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    generated_ids = self.caption_model.generate(**inputs, max_length=50, num_beams=3)
                    caption = self.caption_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                    caption = caption.strip()
                    logger.info(f"BLIP generated caption: '{caption}'")
                    return caption
            else:
                # Fallback: return a placeholder since ITM model can't generate captions
                logger.warning("BLIP captioning model not available, using placeholder")
                return "[Image content - visual analysis available. Please describe what you see in the image.]"
        except Exception as e:
            logger.error(f"Error generating BLIP caption: {str(e)}", exc_info=True)
            return f"[Image processing error: {str(e)}]"
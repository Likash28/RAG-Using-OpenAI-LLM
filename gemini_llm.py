"""
Gemini LLM wrapper for the RAG pipeline
"""
import google.generativeai as genai
from typing import List, Dict, Any
from logging_config import get_logger

logger = get_logger("GeminiLLM")

class GeminiLLM:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash-exp"):
        """Initialize Gemini LLM"""
        self.api_key = api_key
        self.model_name = model_name
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        logger.info(f"Gemini LLM initialized with model: {model_name}")

    def generate_content(self, prompt: str) -> str:
        """Generate content using Gemini"""
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation failed: {str(e)}")
            raise

    def chat(self, messages: List[Dict[str, str]]) -> str:
        """Chat interface compatible with LangChain format"""
        try:
            # Convert messages to a single prompt
            prompt_parts = []
            for message in messages:
                if hasattr(message, 'content'):
                    content = message.content
                    if hasattr(message, 'type'):
                        if message.type == 'system':
                            prompt_parts.append(f"System: {content}")
                        elif message.type == 'human':
                            prompt_parts.append(f"Human: {content}")
                        else:
                            prompt_parts.append(content)
                    else:
                        prompt_parts.append(content)
                else:
                    prompt_parts.append(str(message))
            
            full_prompt = "\n".join(prompt_parts)
            response = self.model.generate_content(full_prompt)
            return response.text
        except Exception as e:
            logger.error(f"Gemini chat failed: {str(e)}")
            raise

    def __call__(self, messages: List[Any]) -> Any:
        """Make the class callable like LangChain models"""
        class Response:
            def __init__(self, text: str):
                self.content = text
                self.text = text
        
        result = self.chat(messages)
        return Response(result)
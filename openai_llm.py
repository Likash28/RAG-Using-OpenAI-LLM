"""
OpenAI LLM wrapper for the RAG pipeline
"""
from openai import OpenAI
from typing import List, Dict, Any, Tuple
from logging_config import get_logger

logger = get_logger("OpenAILLM")

class OpenAILLM:
    def __init__(self, api_key: str, model_name: str = "gpt-4o-mini"):
        """Initialize OpenAI LLM"""
        self.api_key = api_key
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)
        logger.info(f"OpenAI LLM initialized with model: {model_name}")

    def generate_content(self, prompt: str) -> Tuple[str, Dict[str, int]]:
        """Generate content using OpenAI. Returns (content, usage_info)"""
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7,
                max_tokens=1500,  # Reduced for faster response
                timeout=30.0  # 30 second timeout to prevent hanging
            )
            
            # Extract token usage information
            usage = response.usage
            usage_info = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            }
            
            # Log token usage
            logger.info(f"=== OpenAI Token Usage ===")
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Input (Prompt) Tokens: {usage_info['prompt_tokens']}")
            logger.info(f"Output (Completion) Tokens: {usage_info['completion_tokens']}")
            logger.info(f"Total Tokens: {usage_info['total_tokens']}")
            logger.info(f"=== End Token Usage ===")
            
            return response.choices[0].message.content, usage_info
        except Exception as e:
            logger.error(f"OpenAI generation failed: {str(e)}")
            raise

    def chat(self, messages: List[Dict[str, str]]) -> Tuple[str, Dict[str, int]]:
        """Chat interface compatible with LangChain format. Returns (content, usage_info)"""
        try:
            # Convert messages to OpenAI format
            openai_messages = []
            for message in messages:
                if isinstance(message, dict):
                    role = message.get("role", "user")
                    content = message.get("content", str(message))
                elif hasattr(message, 'content'):
                    role = getattr(message, 'role', 'user')
                    content = message.content
                else:
                    role = "user"
                    content = str(message)
                
                openai_messages.append({"role": role, "content": content})
            
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=openai_messages,
                temperature=0.7,
                max_tokens=1500,  # Reduced for faster response
                timeout=30.0  # 30 second timeout to prevent hanging
            )
            
            # Extract token usage information
            usage = response.usage
            usage_info = {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0
            }
            
            # Log token usage
            logger.info(f"=== OpenAI Token Usage ===")
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Input (Prompt) Tokens: {usage_info['prompt_tokens']}")
            logger.info(f"Output (Completion) Tokens: {usage_info['completion_tokens']}")
            logger.info(f"Total Tokens: {usage_info['total_tokens']}")
            logger.info(f"=== End Token Usage ===")
            
            return response.choices[0].message.content, usage_info
        except Exception as e:
            logger.error(f"OpenAI chat failed: {str(e)}")
            raise

    def __call__(self, messages: List[Any]) -> Any:
        """Make the class callable like LangChain models"""
        class Response:
            def __init__(self, text: str):
                self.content = text
                self.text = text
        
        result, _ = self.chat(messages)  # Ignore usage_info for LangChain compatibility
        return Response(result)


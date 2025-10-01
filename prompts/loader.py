"""
Prompt loader for the RAG application
"""
import os
from pathlib import Path
from logging_config import get_logger

logger = get_logger("PromptLoader")

def load_system_prompt() -> str:
    """Load the system prompt from file"""
    try:
        prompt_file = Path(__file__).parent / "system_prompt.txt"
        with open(prompt_file, 'r', encoding='utf-8') as f:
            prompt = f.read()
        logger.info("System prompt loaded successfully")
        return prompt
    except Exception as e:
        logger.error(f"Failed to load system prompt: {str(e)}")
        # Fallback prompt
        return """You are a compassionate mental health information assistant specializing in depression. 
Provide accurate, evidence-based information while maintaining professional boundaries. 
Always encourage consultation with qualified healthcare providers for medical advice.
If someone expresses thoughts of self-harm or suicide, immediately provide crisis resources and encourage immediate professional help."""

def load_crisis_detection_keywords() -> list:
    """Load keywords that indicate crisis situations"""
    return [
        "suicide", "kill myself", "end my life", "not worth living",
        "better off dead", "want to die", "self harm", "hurt myself",
        "can't go on", "give up", "hopeless", "no point",
        "suicidal", "ending it", "final solution", "escape"
    ]

def load_depression_keywords() -> list:
    """Load keywords that indicate depression-related topics"""
    return [
        "depression", "depressed", "sad", "sadness", "melancholy",
        "hopeless", "worthless", "empty", "numb", "mood",
        "anxiety", "anxious", "worry", "panic", "stress",
        "therapy", "counseling", "medication", "antidepressant",
        "symptoms", "treatment", "recovery", "coping",
        "mental health", "psychological", "emotional",
        "bipolar", "manic", "mood disorder", "dysthymia"
    ]

def load_off_topic_keywords() -> list:
    """Load keywords that indicate non-depression topics"""
    return [
        "weather", "sports", "politics", "news", "entertainment",
        "cooking", "travel", "shopping", "work", "job",
        "relationship", "dating", "marriage", "family",
        "technology", "computer", "programming", "science",
        "education", "school", "university", "academic",
        "finance", "money", "investment", "business"
    ]

def is_crisis_query(query: str) -> bool:
    """Check if the query indicates a crisis situation"""
    query_lower = query.lower()
    crisis_keywords = load_crisis_detection_keywords()
    
    for keyword in crisis_keywords:
        if keyword in query_lower:
            return True
    return False

def is_depression_related(query: str) -> bool:
    """Check if the query is depression-related"""
    query_lower = query.lower()
    depression_keywords = load_depression_keywords()
    
    for keyword in depression_keywords:
        if keyword in query_lower:
            return True
    return False

def is_off_topic(query: str) -> bool:
    """Check if the query is off-topic (non-depression)"""
    query_lower = query.lower()
    off_topic_keywords = load_off_topic_keywords()
    
    # Check if query contains off-topic keywords but no depression keywords
    has_off_topic = any(keyword in query_lower for keyword in off_topic_keywords)
    has_depression = is_depression_related(query)
    
    return has_off_topic and not has_depression

def get_crisis_response() -> str:
    """Get the standard crisis response"""
    return """I'm very concerned about what you're sharing. Your feelings are valid and you're not alone.

If you are having thoughts of suicide, please reach out for help immediately:

- National Suicide Prevention Lifeline: 988 (US) or 988lifeline.org
- Crisis Text Line: Text HOME to 741741  
- Emergency Services: 911

Please speak with a mental health professional or crisis counselor right now. There are people who care and want to help you through this difficult time.

This is an automated response for crisis situations. Please seek immediate professional help."""

def get_off_topic_response(query: str) -> str:
    """Get the standard off-topic response"""
    return f"""I appreciate your question about this topic. I'm specifically designed to help with depression-related questions and information. 

For questions about other topics, I'd recommend consulting appropriate resources or specialists in that field.

Is there anything about depression, mental health, or related concerns I can help you with instead?

--- SENTIMENT ANALYSIS ---

User Query Sentiment: Neutral - Seeking general information
Response Sentiment: Helpful but Redirecting - Maintaining focus on specialization
Justification: Chose helpful but redirecting tone to politely guide user to appropriate resources while maintaining professional boundaries"""

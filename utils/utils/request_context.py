"""
Request context utilities for logging and tracing
"""
import threading
from typing import Dict, Any, Optional

# Thread-local storage for request context
_context = threading.local()

def set_request_context(context: Dict[str, Any]) -> None:
    """Set request context in thread-local storage"""
    _context.data = context

def get_request_context() -> Dict[str, Any]:
    """Get request context from thread-local storage"""
    return getattr(_context, 'data', {})

def reset_request_context() -> None:
    """Reset request context in thread-local storage"""
    if hasattr(_context, 'data'):
        delattr(_context, 'data')

def get_context_value(key: str, default: Any = "N/A") -> Any:
    """Get a specific value from request context"""
    context = get_request_context()
    return context.get(key, default)

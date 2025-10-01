"""
Logging configuration for the RAG application
"""
import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
import uuid
from config import settings

# Create logs directory if it doesn't exist
log_dir = settings.log_dir
os.makedirs(log_dir, exist_ok=True)

# Configure root logger
root_logger = logging.getLogger()

# Disable noisy third-party loggers
logging.getLogger("httpcore.connection").disabled = True
logging.getLogger("httpcore.http11").disabled = True
logging.getLogger("openai._base_client").disabled = True
logging.getLogger("langsmith").disabled = True
logging.getLogger("langsmith._internal._serde").disabled = True
logging.getLogger("httpx").disabled = True
logging.getLogger("uvicorn.access").disabled = True
logging.getLogger("uvicorn.error").disabled = True

if settings.environment == "production":
    root_logger.setLevel(getattr(logging, settings.prod_log_level.upper()))
else:
    root_logger.setLevel(getattr(logging, settings.dev_log_level.upper()))

# Custom TimedRotatingFileHandler with immediate flush
class ImmediateFlushingFileHandler(TimedRotatingFileHandler):
    def __init__(self, filename, when='s', interval=1, backupCount=0, encoding=None, 
                 delay=False, utc=False, atTime=None):
        # Use the parent's initialization without overriding the mode
        super().__init__(filename, when, interval, backupCount, encoding, 
                         delay, utc, atTime)
        
    def emit(self, record):
        try:
            # Use the parent implementation for emit
            super().emit(record)
            # Just add the flush operation
            self.flush()
        except Exception:
            self.handleError(record)

# Log file setup
log_file_path = os.path.join(log_dir, "RAGApplication.log")
file_handler = ImmediateFlushingFileHandler(
    filename=log_file_path,
    when="midnight",
    interval=1,
    backupCount=4,
    encoding="utf-8"
)
file_handler.setLevel(logging.DEBUG)

# Console logging setup
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.ERROR)
root_logger.addHandler(stream_handler)

# Custom Request Formatter
class RequestFormatter(logging.Formatter):
    def format(self, record):
        # Set default values for request context
        record.cookie_id = getattr(record, 'cookie_id', "N/A")
        record.user_id = getattr(record, 'user_id', "N/A")
        record.url = getattr(record, 'url', "N/A")
        record.method = getattr(record, 'method', "N/A")
        record.request_id = getattr(record, 'request_id', "N/A")
        record.source_request_id = getattr(record, 'source_request_id', "N/A")
        return super().format(record)

# Improved log format with better readability
formatter = RequestFormatter(
    "%(asctime)s [%(user_id)s] [%(cookie_id)s] [%(source_request_id)s] [%(request_id)s] [%(levelname)s] [%(name)s]  [%(method)s %(url)s] %(message)s"
)
file_handler.setFormatter(formatter)
stream_handler.setFormatter(formatter)
root_logger.addHandler(file_handler)

# Initialize logging
def setup_logging():
    """Initialize logging configuration"""
    # This function can be called to ensure logging is set up
    pass

# Get logger for specific modules
def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(name)

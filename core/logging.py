import logging
import json
import sys
from datetime import datetime
from config.settings import get_settings
from fastapi import Request
from uuid import uuid4

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
        }
        if hasattr(record, "request_id"):
            log_data["request_id"] = record.request_id
        if hasattr(record, "workflow_id"):
            log_data["workflow_id"] = record.workflow_id
        if hasattr(record, "agent"):
            log_data["agent"] = record.agent
            
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_data)

def setup_logging():
    settings = get_settings()
    level = getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO)
    
    logger = logging.getLogger()
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
        
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JSONFormatter())
    logger.addHandler(handler)

def get_logger(name: str):
    return logging.getLogger(name)

# Context variables for logging
import contextvars
request_id_var = contextvars.ContextVar("request_id", default=None)
workflow_id_var = contextvars.ContextVar("workflow_id", default=None)

class ContextFilter(logging.Filter):
    def filter(self, record):
        req_id = request_id_var.get()
        wf_id = workflow_id_var.get()
        if req_id:
            record.request_id = req_id
        if wf_id:
            record.workflow_id = wf_id
        return True

logging.getLogger().addFilter(ContextFilter())

"""
Utility Functions
Logging and helper functions
"""
import logging
import os
from datetime import datetime
import json

def setup_logging(log_file='logs/run.log', level=logging.INFO):
    """
    Setup structured logging
    
    Args:
        log_file: Path to log file
        level: Logging level
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    
    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)
    
    return root_logger

def log_action(action, details):
    """
    Log an action with structured details
    
    Args:
        action: Action name
        details: Dict with action details
    """
    logger = logging.getLogger(__name__)
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'action': action,
        **details
    }
    logger.info(json.dumps(log_entry))


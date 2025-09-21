import logging
import os
from datetime import datetime
import uuid

def get_logger(name: str) -> logging.Logger:
    """
    Set up and return a logger with the given name and output file.
    """

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{uuid.uuid4().hex[:8]}"
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        log_filename = f"run_{run_id}.log"
        # Create file handler
        file_handler = logging.FileHandler(f"logs/{log_filename}", mode="w")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ))

        # Create console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter(
            "%(name)s - %(levelname)s - %(message)s"
        ))

        # Add handlers to logger
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        logger.run_id = run_id

    return logger
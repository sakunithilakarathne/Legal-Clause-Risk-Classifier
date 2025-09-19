import logging
import os

def get_logger(name: str, log_file: str) -> logging.Logger:
    """
    Set up and return a logger with the given name and output file.
    """
    # Ensure logs directory exists
    os.makedirs("logs", exist_ok=True)

    # Create a logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers
    if not logger.handlers:
        # Create file handler
        file_handler = logging.FileHandler(f"logs/{log_file}", mode="w")
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

    return logger
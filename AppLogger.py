
# logger.py

import logging

def get_logger(name):
    """
    Returns a logger with specified name.
    """

    logger = logging.getLogger(name)
    logger.setLevel("DEBUG")

    # Do not add handlers if they are already added
    if not logger.hasHandlers():
        LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        handler = logging.StreamHandler()
        formatter = logging.Formatter(LOG_FORMAT)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


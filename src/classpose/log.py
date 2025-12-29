import logging
import os
from pathlib import Path

CLASSPOSE_LOG_PATH = os.environ.get("CLASSPOSE_LOG_PATH", None)
formatter = logging.Formatter(
    fmt="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def get_logger(log_name: str):
    """
    Returns a logger that logs to console.

    Args:
        log_name (str): The name of the logger.

    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger(log_name)
    logging_level = os.environ.get("LOG_LEVEL", "INFO")
    logger.setLevel(logging_level)
    logger.propagate = False

    has_stream_handler = any(
        type(h) is logging.StreamHandler for h in logger.handlers
    )
    if not has_stream_handler:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    if CLASSPOSE_LOG_PATH:
        Path(CLASSPOSE_LOG_PATH).parent.mkdir(parents=True, exist_ok=True)
        add_file_handler(logger, CLASSPOSE_LOG_PATH)

    return logger


def add_file_handler(logger: logging.Logger, log_path: str) -> None:
    """
    Adds a file handler to the logger.

    Args:
        logger (logging.Logger): The logger to add the file handler to.
        log_path (str): The path to the log file.
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = logging.FileHandler(log_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

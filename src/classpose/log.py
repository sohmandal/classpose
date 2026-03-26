import logging
import os
from pathlib import Path

CLASSPOSE_LOG_PATH = os.environ.get("CLASSPOSE_LOG_PATH", None)
formatter = logging.Formatter(
    fmt="%(asctime)s,%(msecs)03d %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _resolve_stream_level() -> str:
    rank = int(os.environ.get("RANK", "0"))
    if rank > 0:
        return os.environ.get("LOG_LEVEL_NON_MAIN", "WARNING")
    return os.environ.get("LOG_LEVEL", "INFO")


def get_logger(log_name: str):
    """
    Returns a logger that logs to console.

    Args:
        log_name (str): The name of the logger.

    Returns:
        logging.Logger: The logger.
    """
    logger = logging.getLogger(log_name)
    logging_level = _resolve_stream_level()
    logger.setLevel(logging_level)
    logger.propagate = False

    stream_handlers = [
        handler
        for handler in logger.handlers
        if type(handler) is logging.StreamHandler
    ]
    if not stream_handlers:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(logging_level)
        logger.addHandler(stream_handler)
    else:
        for handler in stream_handlers:
            handler.setLevel(logging_level)
            handler.setFormatter(formatter)

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
    target_path = str(Path(log_path).resolve())
    for handler in logger.handlers:
        if not isinstance(handler, logging.FileHandler):
            continue
        if Path(handler.baseFilename).resolve() == Path(target_path):
            return

    file_handler = logging.FileHandler(target_path)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

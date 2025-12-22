import logging

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
    logger.setLevel(logging.INFO)

    if not logger.hasHandlers():
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    logger.propagate = False

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

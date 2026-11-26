import logging
from pathlib import Path


def setup_logger(name: str = "sentiment_analysis") -> logging.Logger:
    """Configure and return a project logger."""

    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Save logs to a file
    file_handler = logging.FileHandler(log_dir / "pipeline.log")
    file_handler.setFormatter(formatter)

    # Display logs in the terminal
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

from __future__ import annotations

import io
import logging
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent


def setup_script_logging(level: int = logging.INFO) -> io.StringIO:
    """Configure console + in-memory logging and return the memory stream."""

    log_stream = io.StringIO()
    logging.basicConfig(
        level=level,
        force=True,
        handlers=[
            logging.StreamHandler(),  # console (stderr)
            logging.StreamHandler(log_stream),  # in-memory stream for MLflow logging
        ],
    )
    return log_stream

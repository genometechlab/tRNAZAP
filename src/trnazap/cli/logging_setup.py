"""Logging for the trnazap CLI.

Every log line goes through ``tqdm.write`` so it is printed above any active progress
bars instead of being glued onto them. trnazap's own loggers log at INFO; third-party
libraries only at WARNING. Python ``warnings`` are routed through logging too.

Worker processes (``infer-multi``) do not write to the terminal themselves: they send
their records to the main process, which prints them (see :func:`setup_worker_logging`).
"""
from __future__ import annotations

import logging
import logging.handlers
import sys

from tqdm import tqdm

LOG_FORMAT = "%(asctime)s %(levelname)s: %(message)s"


class TqdmHandler(logging.Handler):
    """Write records with ``tqdm.write`` so they don't break progress bars."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            tqdm.write(self.format(record), file=sys.stderr)
        except Exception:
            self.handleError(record)


class _PrefixFilter(logging.Filter):
    def __init__(self, prefix: str):
        super().__init__()
        self.prefix = prefix

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = f"{self.prefix}{record.getMessage()}"
        record.args = None
        return True


def _install(handler: logging.Handler) -> None:
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(logging.WARNING)
    logging.getLogger("trnazap").setLevel(logging.INFO)
    logging.captureWarnings(True)


def setup_logging() -> None:
    """Configure logging for the CLI's main process."""
    handler = TqdmHandler()
    handler.setFormatter(logging.Formatter(LOG_FORMAT))
    _install(handler)


def setup_worker_logging(log_queue, prefix: str = "") -> None:
    """Configure a worker process to send its records to ``log_queue``.

    The main process prints them with a ``logging.handlers.QueueListener`` over its
    own handlers. ``prefix`` (e.g. ``"[GPU 1] "``) is prepended to every message.
    """
    handler = logging.handlers.QueueHandler(log_queue)
    if prefix:
        handler.addFilter(_PrefixFilter(prefix))
    _install(handler)

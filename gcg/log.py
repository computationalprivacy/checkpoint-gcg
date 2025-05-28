"""Logging utilities."""

import logging
import sys
from typing import Optional


def setup_logger(verbose: bool, log_file: Optional[str] = None) -> None:
    """Set up the logger.

    Args:
        verbose: Whether to log debug messages.
        log_file: Path to the log file. If None, logs to stdout.
    """
    # Set logging config
    logging.basicConfig(
        filename=log_file,
        filemode="a",
        format="[%(asctime)s - %(name)s - %(levelname)s]: %(message)s",
        level=logging.DEBUG if verbose else logging.INFO,
        force=True,
    )

    if log_file is None:
        # fallback to stdout if no file is specified
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(
            logging.Formatter("[%(asctime)s - %(name)s - %(levelname)s]: %(message)s")
        )
        logging.getLogger().addHandler(console)

    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("git").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("filelock").setLevel(logging.WARNING)
    logging.getLogger("wandb").setLevel(logging.WARNING)
    logging.getLogger("torch.distributed.nn.jit").setLevel(logging.WARNING)
    logging.getLogger("sentry_sdk").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("torch").setLevel(logging.WARNING)

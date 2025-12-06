"""
Logging utilities for the project.
"""

import logging
import os
import sys
from datetime import datetime
from .config import LOG_FORMAT, LOG_LEVEL, OUTPUTS_DIR


def setup_logger(name: str, log_file: str = None, level: str = LOG_LEVEL) -> logging.Logger:
    """
    Set up a logger with console and optional file handlers.

    Args:
        name: Name of the logger
        log_file: Optional path to log file
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)

    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))

    # Remove existing handlers to avoid duplicates
    logger.handlers = []

    # Create formatter
    formatter = logging.Formatter(LOG_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_default_logger(script_name: str) -> logging.Logger:
    """
    Get a default logger for a script with timestamped log file.

    Args:
        script_name: Name of the script (used in log filename)

    Returns:
        Configured logger instance
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(OUTPUTS_DIR, 'logs')
    log_file = os.path.join(log_dir, f"{script_name}_{timestamp}.log")

    return setup_logger(script_name, log_file)


class TrainingLogger:
    """
    Logger specifically for tracking training progress.
    """

    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self.epoch_metrics = []

    def log_epoch(self, epoch: int, train_loss: float, train_acc: float,
                  val_loss: float = None, val_acc: float = None):
        """Log metrics for an epoch."""
        msg = f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}"

        if val_loss is not None and val_acc is not None:
            msg += f" | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"

        self.logger.info(msg)

        # Store metrics
        metrics = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc
        }
        self.epoch_metrics.append(metrics)

    def log_best_model(self, epoch: int, metric_value: float, metric_name: str = "accuracy"):
        """Log when best model is saved."""
        self.logger.info(f"New best model saved at epoch {epoch} with {metric_name}: {metric_value:.4f}")

    def log_early_stopping(self, epoch: int, patience: int):
        """Log early stopping trigger."""
        self.logger.warning(f"Early stopping triggered at epoch {epoch} (patience: {patience})")

    def get_metrics_history(self):
        """Return all logged metrics."""
        return self.epoch_metrics

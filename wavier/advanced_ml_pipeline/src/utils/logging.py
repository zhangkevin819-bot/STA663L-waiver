"""
Advanced logging system with structured output, context managers,
and integration with MLflow for experiment tracking.
"""

import functools
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger


class LoggerConfig:
    """Centralized logger configuration."""
    
    @staticmethod
    def setup(
        level: str = "INFO",
        log_dir: Optional[Path] = None,
        serialize: bool = False,
        colorize: bool = True
    ) -> None:
        """Configure application-wide logging."""
        logger.remove()
        
        # Console handler with colors
        logger.add(
            sys.stdout,
            level=level,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
                "<level>{message}</level>"
            ),
            colorize=colorize,
        )
        
        # File handler with rotation
        if log_dir:
            log_dir.mkdir(parents=True, exist_ok=True)
            logger.add(
                log_dir / "app_{time}.log",
                level=level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                rotation="100 MB",
                retention="30 days",
                compression="zip",
                serialize=serialize,
            )


@contextmanager
def log_execution_time(operation: str, level: str = "INFO"):
    """Context manager to log execution time of operations."""
    start = time.perf_counter()
    logger.log(level, f"Starting: {operation}")
    
    try:
        yield
    except Exception as e:
        duration = time.perf_counter() - start
        logger.error(f"Failed: {operation} (took {duration:.3f}s): {e}")
        raise
    else:
        duration = time.perf_counter() - start
        logger.log(level, f"Completed: {operation} (took {duration:.3f}s)")


def log_function_call(func: Callable) -> Callable:
    """Decorator to automatically log function calls with args."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        func_name = f"{func.__module__}.{func.__qualname__}"
        
        # Format arguments
        args_repr = [repr(a)[:100] for a in args[:3]]
        kwargs_repr = [f"{k}={repr(v)[:50]}" for k, v in list(kwargs.items())[:3]]
        signature = ", ".join(args_repr + kwargs_repr)
        
        logger.debug(f"Calling {func_name}({signature})")
        
        try:
            result = func(*args, **kwargs)
            logger.debug(f"Completed {func_name}")
            return result
        except Exception as e:
            logger.exception(f"Exception in {func_name}: {e}")
            raise
    
    return wrapper


class MetricsLogger:
    """Logger for tracking ML metrics with MLflow integration."""
    
    def __init__(self, experiment_name: str, run_name: Optional[str] = None):
        self.experiment_name = experiment_name
        self.run_name = run_name
        self._metrics_buffer: dict[str, list] = {}
    
    def log_metric(
        self, 
        key: str, 
        value: float, 
        step: Optional[int] = None
    ) -> None:
        """Log a single metric value."""
        if key not in self._metrics_buffer:
            self._metrics_buffer[key] = []
        
        self._metrics_buffer[key].append((value, step))
        logger.info(f"Metric | {key}: {value:.6f}" + (f" @ step {step}" if step else ""))
    
    def log_metrics(
        self, 
        metrics: dict[str, float], 
        step: Optional[int] = None
    ) -> None:
        """Log multiple metrics at once."""
        for key, value in metrics.items():
            self.log_metric(key, value, step)
    
    def log_hyperparams(self, params: dict[str, Any]) -> None:
        """Log hyperparameters."""
        logger.info(f"Hyperparameters: {params}")
    
    def flush(self) -> None:
        """Flush buffered metrics to persistent storage."""
        logger.debug(f"Flushing {len(self._metrics_buffer)} metric streams")
        self._metrics_buffer.clear()


class ExperimentTracker:
    """Unified experiment tracking interface."""
    
    def __init__(self, backend: str = "mlflow"):
        self.backend = backend
        self._active_run = None
    
    def start_run(self, run_name: Optional[str] = None) -> None:
        """Start a new experiment run."""
        logger.info(f"Starting experiment run: {run_name or 'unnamed'}")
        self._active_run = run_name
    
    def end_run(self) -> None:
        """End the current experiment run."""
        if self._active_run:
            logger.info(f"Ending experiment run: {self._active_run}")
            self._active_run = None
    
    @contextmanager
    def run_context(self, run_name: Optional[str] = None):
        """Context manager for experiment runs."""
        self.start_run(run_name)
        try:
            yield self
        finally:
            self.end_run()


# Global logger instance
app_logger = logger.bind(context="application")

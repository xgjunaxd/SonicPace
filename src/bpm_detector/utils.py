"""Utility functions for audio analysis operations."""

import logging
from typing import Union

import numpy as np


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up and return a customized logger.
    
    Args:
        name: Name of the logger
        level: Logging level
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Only add handler if the logger doesn't already have one
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    logger.setLevel(level)
    return logger


def format_time(seconds: float) -> str:
    """Format seconds into a standard MM:SS string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., '03:45')
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    return f"{mins:02d}:{secs:02d}"


def format_time_ms(seconds: float) -> str:
    """Format seconds into MM:SS.ms string.
    
    Args:
        seconds: Time in seconds
        
    Returns:
        Formatted time string (e.g., '03:45.25')
    """
    mins = int(seconds // 60)
    secs = int(seconds % 60)
    ms = int((seconds % 1) * 100)
    return f"{mins:02d}:{secs:02d}.{ms:02d}"


def safe_divide(a: Union[float, np.ndarray], b: Union[float, np.ndarray], default: float = 0.0) -> Union[float, np.ndarray]:
    """Safely divide numbers or arrays without raising ZeroDivisionError.
    
    Args:
        a: Numerator
        b: Denominator
        default: Default value if denominator is 0
        
    Returns:
        Result of division or default value
    """
    if isinstance(b, (int, float)):
        if b == 0 or np.isnan(b):
            return default
        return a / b
        
    if isinstance(b, np.ndarray):
        # Prevent division by zero warnings
        old_settings = np.seterr(divide='ignore', invalid='ignore')
        result = np.divide(a, b, out=np.full_like(a, default, dtype=float), where=b != 0)
        np.seterr(**old_settings)
        return result
        
    return default


def normalize_array(arr: np.ndarray) -> np.ndarray:
    """Normalize a numpy array to the 0-1 range.
    
    Args:
        arr: Input array
        
    Returns:
        Normalized array
    """
    if len(arr) == 0:
        return arr
        
    arr_min = np.min(arr)
    arr_max = np.max(arr)
    
    if arr_max == arr_min:
        return np.zeros_like(arr)
        
    return (arr - arr_min) / (arr_max - arr_min)


def is_supported_audio_format(filename: str) -> bool:
    """Check if the file extension is a supported audio format.
    
    Args:
        filename: Path or name of the file
        
    Returns:
        True if supported, False otherwise
    """
    supported_extensions = {'.wav', '.mp3', '.flac', '.ogg', '.m4a', '.aac'}
    ext = filename.lower()
    return any(ext.endswith(supported_ext) for supported_ext in supported_extensions)
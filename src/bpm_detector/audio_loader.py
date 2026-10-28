"""Audio loading, validation, and preprocessing module."""

import os
from typing import Optional, Tuple

import librosa
import numpy as np

from .exceptions import AudioLoadError, UnsupportedFormatError
from .utils import is_supported_audio_format


class AudioLoader:
    """Handles loading, validating, and preprocessing audio files."""

    def __init__(self, target_sr: int = 22050, mono: bool = True, normalize: bool = True):
        """Initialize the audio loader.

        Args:
            target_sr: Target sample rate for resampling
            mono: Whether to convert audio to mono
            normalize: Whether to peak-normalize the audio signal
        """
        self.target_sr = target_sr
        self.mono = mono
        self.normalize = normalize

    def load(
        self,
        file_path: str,
        offset: float = 0.0,
        duration: Optional[float] = None,
    ) -> Tuple[np.ndarray, int]:
        """Load and preprocess an audio file.

        Args:
            file_path: Path to the audio file
            offset: Start reading after this time (in seconds)
            duration: Only load up to this much audio (in seconds)

        Returns:
            Tuple of (audio signal array, sample rate)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        if not is_supported_audio_format(file_path):
            raise UnsupportedFormatError(f"Unsupported audio format for file: {file_path}")

        try:
            y, sr = librosa.load(
                file_path,
                sr=self.target_sr,
                mono=self.mono,
                offset=offset,
                duration=duration,
            )
        except Exception as e:
            raise AudioLoadError(f"Error loading audio file {file_path}: {e}") from e

        if len(y) == 0:
            raise AudioLoadError(f"Loaded audio file is empty: {file_path}")

        if self.normalize:
            y = self.normalize_signal(y)

        return y, int(sr)

    @staticmethod
    def normalize_signal(y: np.ndarray) -> np.ndarray:
        """Peak-normalize an audio signal to [-1.0, 1.0]."""
        max_val = np.max(np.abs(y))
        if max_val > 0:
            return y / max_val
        return y

    @staticmethod
    def trim_silence(y: np.ndarray, top_db: float = 30.0) -> np.ndarray:
        """Trim leading and trailing silence from an audio signal."""
        trimmed, _ = librosa.effects.trim(y, top_db=top_db)
        return trimmed
"""Core BPM detection module."""

import os
from typing import Any, Dict, List, Optional, Tuple, Union

import librosa
import numpy as np

from .beat_tracker import BeatTracker
from .exceptions import AudioLoadError


class BPMDetector:
    """Primary detector for estimating BPM (tempo) from audio signals and files."""

    def __init__(
        self,
        sr: int = 22050,
        hop_length: int = 512,
        min_bpm: float = 60.0,
        max_bpm: float = 200.0,
    ):
        """Initialize the BPM detector.

        Args:
            sr: Target sample rate for audio loading
            hop_length: Hop length for onset and beat analysis
            min_bpm: Minimum expected BPM
            max_bpm: Maximum expected BPM
        """
        self.sr = sr
        self.hop_length = hop_length
        self.min_bpm = min_bpm
        self.max_bpm = max_bpm
        self.beat_tracker = BeatTracker(hop_length=hop_length)

    def detect_from_file(self, file_path: str) -> Tuple[float, float]:
        """Detect BPM and confidence from an audio file.

        Args:
            file_path: Path to the audio file

        Returns:
            Tuple of (BPM, confidence percentage 0-100)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        try:
            y, sr = librosa.load(file_path, sr=self.sr, mono=True)
        except Exception as e:
            raise AudioLoadError(f"Failed to load audio file {file_path}: {e}") from e

        return self.detect(y, int(sr))

    def detect(self, y: np.ndarray, sr: Optional[int] = None) -> Tuple[float, float]:
        """Detect BPM and confidence from a raw audio signal array.

        Args:
            y: Audio time series
            sr: Sample rate (defaults to self.sr)

        Returns:
            Tuple of (BPM, confidence percentage 0-100)
        """
        sample_rate = sr or self.sr
        if len(y) == 0:
            return 0.0, 0.0

        track_results = self.beat_tracker.track(y, sample_rate)
        bpm = float(track_results.get("tempo", 0.0))
        confidence = float(track_results.get("tempo_confidence", 0.0)) * 100.0

        # Fold or double tempo into the target [min_bpm, max_bpm] range
        if bpm > 0:
            while bpm < self.min_bpm:
                bpm *= 2.0
            while bpm > self.max_bpm:
                bpm /= 2.0

        return round(bpm, 2), round(confidence, 2)

    def detect_detailed(self, file_path: str) -> Dict[str, Any]:
        """Perform detailed BPM detection including beat timestamps and candidates.

        Args:
            file_path: Path to the audio file

        Returns:
            Dictionary with detailed tempo and beat information
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")

        y, sr = librosa.load(file_path, sr=self.sr, mono=True)
        track_results = self.beat_tracker.track(y, int(sr))
        candidates = self.beat_tracker.get_tempo_candidates(y, int(sr))

        bpm, confidence = self.detect(y, int(sr))

        return {
            "bpm": bpm,
            "confidence": confidence,
            "raw_tempo": track_results["tempo"],
            "beat_times": track_results["beat_times"],
            "onset_times": track_results["onset_times"],
            "candidates": [
                {"bpm": c[0], "strength": c[1]} for c in candidates
            ],
        }
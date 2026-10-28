"""Beat and tempo tracking module."""

from typing import Any, Dict, Tuple, Union

import librosa
import numpy as np


class BeatTracker:
    """Tracks beats, estimates tempo, and detects onsets."""

    def __init__(self, hop_length: int = 512):
        """Initialize beat tracker.
        
        Args:
            hop_length: Hop length for feature extraction
        """
        self.hop_length = hop_length

    def track(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Perform beat and tempo tracking on the audio signal.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary containing tempo, beats, and onset information
        """
        # Calculate onset envelope
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        
        # Track beats
        tempo_array, beats = librosa.beat.beat_track(
            onset_envelope=onset_env, 
            sr=sr, 
            hop_length=self.hop_length
        )
        
        # Handle librosa return types safely
        tempo = float(tempo_array[0]) if isinstance(tempo_array, np.ndarray) else float(tempo_array)
        
        # Convert beat frames to times
        beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=self.hop_length)
        
        # Detect actual onsets
        onsets = librosa.onset.onset_detect(
            onset_envelope=onset_env, 
            sr=sr, 
            hop_length=self.hop_length
        )
        onset_times = librosa.frames_to_time(onsets, sr=sr, hop_length=self.hop_length)
        
        return {
            'tempo': tempo,
            'beat_frames': beats.tolist() if isinstance(beats, np.ndarray) else beats,
            'beat_times': beat_times.tolist() if isinstance(beat_times, np.ndarray) else beat_times,
            'onset_frames': onsets.tolist() if isinstance(onsets, np.ndarray) else onsets,
            'onset_times': onset_times.tolist() if isinstance(onset_times, np.ndarray) else onset_times,
            'tempo_confidence': self._estimate_tempo_confidence(onset_env, beats)
        }

    def get_tempo_candidates(self, y: np.ndarray, sr: int, max_candidates: int = 5) -> list[Tuple[float, float]]:
        """Get multiple tempo candidates with their relative strengths.
        
        Args:
            y: Audio signal
            sr: Sample rate
            max_candidates: Maximum number of candidates to return
            
        Returns:
            List of tuples containing (tempo, strength)
        """
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        prior = librosa.beat.tempo(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length, aggregate=None)
        
        # Get unique tempos and their counts (as a basic strength metric)
        tempos, counts = np.unique(np.round(prior), return_counts=True)
        
        # Sort by strength (count)
        sorted_indices = np.argsort(counts)[::-1]
        
        candidates = []
        total_counts = np.sum(counts)
        
        for i in range(min(max_candidates, len(sorted_indices))):
            idx = sorted_indices[i]
            tempo = float(tempos[idx])
            strength = float(counts[idx] / total_counts)
            candidates.append((tempo, strength))
            
        return candidates

    def _estimate_tempo_confidence(self, onset_env: np.ndarray, beats: np.ndarray) -> float:
        """Estimate how confident the tracker is in the detected tempo/beats."""
        if len(beats) < 4:
            return 0.0
            
        # Check variance of inter-beat intervals (IBI)
        ibis = np.diff(beats)
        if len(ibis) == 0 or np.mean(ibis) == 0:
            return 0.0
            
        # A steady tempo will have low variance in IBI
        cv = np.std(ibis) / np.mean(ibis)  # Coefficient of variation
        
        # Map CV to a 0.0 - 1.0 confidence score (heuristically)
        # CV of 0 = 100% confidence, CV > 0.2 = ~0% confidence
        confidence = max(0.0, 1.0 - (cv * 5.0))
        
        return float(confidence)
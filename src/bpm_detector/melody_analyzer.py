"""Melody extraction and analysis module."""

from typing import Any, Dict

import librosa
import numpy as np

from .music_theory import NOTE_NAMES


class MelodyAnalyzer:
    """Analyzes melodic characteristics, pitch tracking, and range."""

    def __init__(self, hop_length: int = 512):
        """Initialize melody analyzer.
        
        Args:
            hop_length: Hop length for analysis
        """
        self.hop_length = hop_length

    def analyze(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Perform complete melody analysis.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary containing melodic characteristics
        """
        # Pitch tracking using PYIN
        f0, voiced_flag, voiced_probs = librosa.pyin(
            y, 
            fmin=float(librosa.note_to_hz('C2')), 
            fmax=float(librosa.note_to_hz('C7')), 
            sr=sr,
            hop_length=self.hop_length
        )

        # Handle case where no melody is detected
        if not np.any(voiced_flag):
            return self._get_empty_results()

        # Extract voiced frequencies
        voiced_f0 = f0[voiced_flag]
        
        # Calculate melodic range
        fmin_detected = float(np.min(voiced_f0))
        fmax_detected = float(np.max(voiced_f0))
        
        # Convert frequencies to MIDI notes for range calculations
        midi_min = librosa.hz_to_midi(fmin_detected)
        midi_max = librosa.hz_to_midi(fmax_detected)
        range_octaves = (midi_max - midi_min) / 12.0
        
        # Calculate melody coverage (percentage of frames with a detected pitch)
        melody_coverage = float(np.sum(voiced_flag) / len(voiced_flag))
        
        return {
            'melody_present': True,
            'melody_coverage': melody_coverage,
            'melodic_range': {
                'range_octaves': float(range_octaves),
                'lowest_note_name': librosa.hz_to_note(fmin_detected),
                'highest_note_name': librosa.hz_to_note(fmax_detected),
                'vocal_range_category': self._categorize_vocal_range(fmin_detected, fmax_detected)
            },
        }

    def _categorize_vocal_range(self, fmin: float, fmax: float) -> str:
        """Categorize the detected pitch range into standard vocal ranges."""
        # Extremely simplified vocal range estimation
        if fmin < 100:
            return "Bass/Baritone"
        elif fmax > 800:
            return "Soprano"
        elif fmax > 400:
            return "Alto/Tenor"
        return "Mid-Range"

    def _get_empty_results(self) -> Dict[str, Any]:
        """Return default structure when no melody is present."""
        return {
            'melody_present': False,
            'melody_coverage': 0.0,
            'melodic_range': {
                'range_octaves': 0.0,
                'lowest_note_name': 'Unknown',
                'highest_note_name': 'Unknown',
                'vocal_range_category': 'Unknown'
            },
        }
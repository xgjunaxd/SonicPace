"""Rhythm and time signature analysis module."""

from typing import Any, Dict

import librosa
import numpy as np


class RhythmAnalyzer:
    """Analyzes rhythmic characteristics, time signature, and groove."""

    def __init__(self, hop_length: int = 512):
        """Initialize rhythm analyzer.
        
        Args:
            hop_length: Hop length for analysis
        """
        self.hop_length = hop_length

    def analyze(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Perform complete rhythm analysis.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary containing rhythm characteristics
        """
        # Onset detection for rhythmic features
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        
        # Tempo and beat tracking
        tempo_array, beats = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length)
        tempo = float(tempo_array[0]) if isinstance(tempo_array, np.ndarray) else float(tempo_array)

        # Analyze time signature and groove
        time_signature = self._detect_time_signature(onset_env, beats)
        groove_type, swing_ratio = self._analyze_groove(onset_env, beats)
        syncopation = self._calculate_syncopation(onset_env, beats)
        complexity = self._calculate_rhythmic_complexity(onset_env)

        return {
            'tempo': tempo,
            'time_signature': time_signature,
            'groove_type': groove_type,
            'syncopation_level': syncopation,
            'rhythmic_complexity': complexity,
            'swing_ratio': swing_ratio
        }

    def _detect_time_signature(self, onset_env: np.ndarray, beats: np.ndarray) -> str:
        """Estimate time signature based on beat emphasis."""
        if len(beats) < 8:
            return "4/4"  # Default fallback
            
        # Get onset strengths at beat frames
        beat_strengths = onset_env[beats]
        
        # Simple autocorrelation to find meter (looking for 3/4 vs 4/4)
        if len(beat_strengths) > 12:
            # Normalize
            beat_strengths = beat_strengths - np.mean(beat_strengths)
            if np.std(beat_strengths) > 0:
                beat_strengths = beat_strengths / np.std(beat_strengths)
                
            auto_corr = np.correlate(beat_strengths, beat_strengths, mode='full')
            auto_corr = auto_corr[len(auto_corr)//2:]
            
            # Compare strength at lag 3 vs lag 4
            if len(auto_corr) > 4:
                lag3 = auto_corr[3]
                lag4 = auto_corr[4]
                if lag3 > lag4 * 1.2:
                    return "3/4"
        
        return "4/4"

    def _analyze_groove(self, onset_env: np.ndarray, beats: np.ndarray) -> tuple[str, float]:
        """Analyze groove type and swing ratio."""
        # Simplified placeholder for groove detection
        return "straight", 0.5

    def _calculate_syncopation(self, onset_env: np.ndarray, beats: np.ndarray) -> float:
        """Calculate level of syncopation (off-beat emphasis)."""
        if len(beats) < 2:
            return 0.0
            
        # Find onsets that occur exactly between beats
        off_beat_energy = 0.0
        on_beat_energy = 0.0
        
        for i in range(len(beats) - 1):
            start, end = beats[i], beats[i+1]
            midpoint = (start + end) // 2
            
            # Check region around beats vs region around midpoints
            on_beat_energy += np.sum(onset_env[start:start+2])
            off_beat_energy += np.sum(onset_env[midpoint-1:midpoint+2])
            
        total = on_beat_energy + off_beat_energy + 1e-8
        syncopation = off_beat_energy / total
        
        return min(1.0, float(syncopation * 2.0))

    def _calculate_rhythmic_complexity(self, onset_env: np.ndarray) -> float:
        """Estimate overall rhythmic complexity based on onset variance."""
        if len(onset_env) == 0:
            return 0.0
        variance = np.var(onset_env)
        # Normalize to 0-1 range roughly
        return min(1.0, float(variance) / 10.0)

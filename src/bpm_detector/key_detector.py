"""Key detection module."""

from typing import Tuple
import librosa
import numpy as np


class KeyDetector:
    """Detects musical key and mode from audio signals."""

    # Krumhansl-Schmuckler key profiles
    MAJOR_PROFILE = [6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88]
    MINOR_PROFILE = [6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17]
    
    NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    def __init__(self, hop_length: int = 512):
        """Initialize key detector."""
        self.hop_length = hop_length
        
        # Normalize profiles for cosine similarity calculations
        self.major_profile = np.array(self.MAJOR_PROFILE)
        self.major_profile /= np.linalg.norm(self.major_profile)
        
        self.minor_profile = np.array(self.MINOR_PROFILE)
        self.minor_profile /= np.linalg.norm(self.minor_profile)

    def detect(self, y: np.ndarray, sr: int) -> Tuple[str, float]:
        """Detect key and confidence from audio signal.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Tuple of (Key string, Confidence percentage)
        """
        # Extract harmonic part for cleaner pitch detection
        y_harmonic, _ = librosa.effects.hpss(y)
        
        # Compute chromagram
        chroma = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr, hop_length=self.hop_length)
        
        # Average chroma across all frames
        chroma_mean = np.mean(chroma, axis=1)
        
        # Normalize the mean chroma vector
        if np.sum(chroma_mean) > 0:
            chroma_mean /= np.linalg.norm(chroma_mean)
            
        best_score = -1.0
        best_key = "Unknown"
        
        # Test all 12 pitch classes for Major and Minor
        for i in range(12):
            # Shift the chroma array so the test root note is at index 0
            shifted_chroma = np.roll(chroma_mean, -i)
            
            # Check major profile correlation
            major_score = np.dot(shifted_chroma, self.major_profile)
            if major_score > best_score:
                best_score = float(major_score)
                best_key = f"{self.NOTE_NAMES[i]} Major"
                
            # Check minor profile correlation
            minor_score = np.dot(shifted_chroma, self.minor_profile)
            if minor_score > best_score:
                best_score = float(minor_score)
                best_key = f"{self.NOTE_NAMES[i]} Minor"
        
        # Scale score to a rough 0-100 confidence percentage
        confidence = min(100.0, max(0.0, best_score * 100))
        
        return best_key, confidence
"""Song structure and section analysis module."""

from typing import Any, Dict, List

import librosa
import numpy as np


class StructureAnalyzer:
    """Analyzes musical structure, detecting sections like verse, chorus, etc."""

    def __init__(self, hop_length: int = 512, n_fft: int = 2048):
        """Initialize structure analyzer.
        
        Args:
            hop_length: Hop length for feature extraction
            n_fft: FFT window size
        """
        self.hop_length = hop_length
        self.n_fft = n_fft

    def analyze(self, y: np.ndarray, sr: int) -> Dict[str, Any]:
        """Perform structure analysis on audio signal.
        
        Args:
            y: Audio signal
            sr: Sample rate
            
        Returns:
            Dictionary containing structural analysis results
        """
        duration = librosa.get_duration(y=y, sr=sr)
        
        # If the audio is extremely short, just return a single section
        if duration < 5.0:
            return self._get_default_structure(duration)

        # 1. Extract Mel-frequency cepstral coefficients (MFCCs) for timbral features
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20, hop_length=self.hop_length)
        
        # 2. Extract Chroma features for harmonic content
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length)
        
        # Stack features
        features = np.vstack([mfcc, chroma])
        
        # 3. Compute self-similarity matrix
        # (Using a simplified approach for fast execution)
        recurrence = librosa.segment.recurrence_matrix(
            features, mode='affinity', metric='cosine', sparse=False
        )
        
        # 4. Detect boundaries using novelty curve
        # Create a novelty curve from the recurrence matrix
        novelty = np.mean(recurrence, axis=1)
        novelty = np.abs(np.diff(novelty))
        novelty = np.append(novelty, 0) # Pad to match length
        
        # Smooth novelty curve
        window_len = max(3, int(sr * 2.0 / self.hop_length)) # 2-second window
        window = np.ones(window_len) / window_len
        novelty_smooth = np.convolve(novelty, window, mode='same')
        
        # Find peaks in novelty curve (section boundaries)
        # We'll use a simple threshold-based peak picking
        threshold = np.mean(novelty_smooth) + 0.5 * np.std(novelty_smooth)
        peaks = librosa.util.peak_pick(novelty_smooth, pre_max=3, post_max=3, pre_avg=3, post_avg=5, delta=threshold, wait=5)
        
        # Convert frame indices to time
        boundary_times = librosa.frames_to_time(peaks, sr=sr, hop_length=self.hop_length)
        
        # Add start and end boundaries
        boundaries = [0.0] + list(boundary_times)
        if boundaries[-1] < duration - 1.0:
            boundaries.append(duration)
            
        # 5. Build section metadata
        sections = self._build_sections(boundaries, y, sr)
        
        # 6. Estimate global structure properties
        complexity = float(len(sections) / (duration / 15.0)) # rough heuristic: expect a change every 15s
        complexity = min(1.0, max(0.0, complexity))
        
        return {
            'form': 'Generated Form',
            'section_count': len(sections),
            'repetition_ratio': 0.3,  # Placeholder for deeper repetition analysis
            'structural_complexity': complexity,
            'sections': sections
        }

    def _build_sections(self, boundaries: List[float], y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Construct detailed section metadata from boundary times."""
        sections = []
        
        # Simple heuristic mapping based on position and relative energy
        types = ['intro', 'verse', 'chorus', 'bridge', 'outro']
        
        for i in range(len(boundaries) - 1):
            start_time = boundaries[i]
            end_time = boundaries[i + 1]
            section_duration = end_time - start_time
            
            # Extract section audio to measure energy
            start_sample = int(start_time * sr)
            end_sample = int(end_time * sr)
            section_audio = y[start_sample:end_sample]
            
            # Calculate RMS energy for this section
            if len(section_audio) > 0:
                energy = float(np.sqrt(np.mean(section_audio**2)))
            else:
                energy = 0.0
                
            # Assign a pseudo-type based on position (in reality this requires clustering)
            if i == 0:
                section_type = 'intro'
            elif i == len(boundaries) - 2:
                section_type = 'outro'
            elif energy > 0.1: # high energy -> chorus
                section_type = 'chorus'
            else:
                section_type = 'verse'
                
            sections.append({
                'type': section_type,
                'start_time': float(start_time),
                'end_time': float(end_time),
                'duration': float(section_duration),
                'energy_level': min(1.0, energy * 5.0), # normalized approximation
                'complexity': 0.5  # placeholder
            })
            
        return sections

    def _get_default_structure(self, duration: float) -> Dict[str, Any]:
        """Return a basic single-section structure for very short audio."""
        return {
            'form': 'A',
            'section_count': 1,
            'repetition_ratio': 0.0,
            'structural_complexity': 0.1,
            'sections': [{
                'type': 'theme',
                'start_time': 0.0,
                'end_time': float(duration),
                'duration': float(duration),
                'energy_level': 0.5,
                'complexity': 0.1
            }]
        }
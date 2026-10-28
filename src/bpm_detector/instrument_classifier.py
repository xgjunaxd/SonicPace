"""Instrument classification module."""

from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np


class InstrumentClassifier:
    """Classifies instruments present in audio signals."""

    INSTRUMENT_RANGES = {
        'bass': {'low': 20.0, 'high': 250.0},
        'kick': {'low': 20.0, 'high': 100.0},
        'snare': {'low': 150.0, 'high': 1000.0},
        'hihat': {'low': 5000.0, 'high': 15000.0},
        'vocal': {'low': 150.0, 'high': 3000.0},
        'guitar': {'low': 80.0, 'high': 1200.0},
        'piano': {'low': 27.0, 'high': 4000.0},
    }

    def __init__(self, hop_length: int = 512, n_fft: int = 2048):
        """Initialize instrument classifier."""
        self.hop_length = hop_length
        self.n_fft = n_fft

    def classify_instruments(self, y: np.ndarray, sr: int) -> List[Dict[str, Any]]:
        """Classify instruments present in the audio."""
        stft = librosa.stft(y, hop_length=self.hop_length, n_fft=self.n_fft)
        magnitude = np.abs(stft)
        freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
        
        harmonic, percussive = librosa.effects.hpss(y)
        harm_stft = np.abs(librosa.stft(harmonic, hop_length=self.hop_length, n_fft=self.n_fft))
        perc_stft = np.abs(librosa.stft(percussive, hop_length=self.hop_length, n_fft=self.n_fft))

        instruments = []
        for inst, freq_range in self.INSTRUMENT_RANGES.items():
            confidence = self._calculate_instrument_confidence(
                magnitude, freqs, (freq_range['low'], freq_range['high']),
                instrument=inst, harmonic=harm_stft, percussive=perc_stft
            )
            if confidence > 0.2:
                instruments.append({
                    'instrument': inst,
                    'confidence': float(confidence),
                    'prominence': self._calculate_instrument_prominence(
                        magnitude, freqs, low_freq=freq_range['low'], high_freq=freq_range['high']
                    )
                })
        
        # Sort by confidence and filter
        sorted_instruments = sorted(instruments, key=lambda x: x['confidence'], reverse=True)
        return self._filter_redundant_instruments(sorted_instruments)

    def _calculate_instrument_confidence(
        self,
        magnitude: np.ndarray,
        freqs: np.ndarray,
        freq_range: Optional[Tuple[float, float]] = None,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
        instrument: Optional[str] = None,
        harmonic: Optional[np.ndarray] = None,
        percussive: Optional[np.ndarray] = None,
        spectral_shape=None,
    ) -> float:
        """Calculate confidence for instrument presence."""
        low = freq_range[0] if freq_range else (low_freq or 0.0)
        high = freq_range[1] if freq_range else (high_freq or 22050.0)
        
        freq_mask = (freqs >= low) & (freqs <= high)
        if not np.any(freq_mask):
            return 0.0
            
        total_energy = np.sum(magnitude) + 1e-8
        band_energy = np.sum(magnitude[freq_mask, :])
        base_conf = float(band_energy / total_energy)
        
        if instrument in ['kick', 'snare', 'hihat'] and percussive is not None:
            perc_energy = np.sum(percussive[freq_mask, :])
            base_conf = float((base_conf + (perc_energy / total_energy)) / 2.0)
        elif instrument in ['vocal', 'piano', 'guitar', 'bass'] and harmonic is not None:
            harm_energy = np.sum(harmonic[freq_mask, :])
            base_conf = float((base_conf + (harm_energy / total_energy)) / 2.0)
            
        return min(1.0, base_conf * 2.5)

    def _calculate_instrument_prominence(
        self,
        magnitude: np.ndarray,
        freqs: np.ndarray,
        freq_range: Optional[Tuple[float, float]] = None,
        low_freq: Optional[float] = None,
        high_freq: Optional[float] = None,
    ) -> float:
        """Calculate instrument prominence in the mix."""
        low = freq_range[0] if freq_range else (low_freq or 0.0)
        high = freq_range[1] if freq_range else (high_freq or 22050.0)
        freq_mask = (freqs >= low) & (freqs <= high)
        
        if not np.any(freq_mask):
            return 0.0
            
        band_max = np.max(magnitude[freq_mask, :])
        overall_max = np.max(magnitude) + 1e-8
        return float(band_max / overall_max)

    def _filter_redundant_instruments(self, instruments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out redundant instrument detections."""
        return instruments[:5]  # Keep top 5 most confident instruments
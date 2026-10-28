"""Custom exceptions for the BPM and music analysis package."""

class BPMDetectorError(Exception):
    """Base exception for all BPM detector errors."""
    pass


class AudioLoadError(BPMDetectorError):
    """Raised when an audio file cannot be loaded or is corrupted."""
    pass


class UnsupportedFormatError(AudioLoadError):
    """Raised when the audio file format is not supported."""
    pass


class AnalysisError(BPMDetectorError):
    """Raised when an error occurs during audio analysis."""
    pass


class FeatureExtractionError(AnalysisError):
    """Raised when audio features (MFCC, Chroma, etc.) cannot be extracted."""
    pass


class ParallelProcessingError(BPMDetectorError):
    """Raised when a parallel execution pool fails to complete its tasks."""
    pass


class ConfigurationError(BPMDetectorError):
    """Raised when the analyzer is provided with invalid configuration parameters."""
    pass

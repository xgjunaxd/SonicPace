"""BPM Detector and Music Analysis Package."""

from .bpm_detector import BPMDetector
from .music_analyzer import AudioAnalyzer
from .parallel_analyzer import SmartParallelAudioAnalyzer
from .key_detector import KeyDetector
from .rhythm_analyzer import RhythmAnalyzer
from .melody_analyzer import MelodyAnalyzer
from .structure_analyzer import StructureAnalyzer
from .similarity_engine import SimilarityEngine
from .audio_loader import AudioLoader
from .config import AnalysisConfig, DEFAULT_CONFIG
from .exporters import ResultExporter
from .exceptions import (
    BPMDetectorError,
    AudioLoadError,
    UnsupportedFormatError,
    AnalysisError,
)

__version__ = "1.2.0"
__all__ = [
    "BPMDetector",
    "AudioAnalyzer",
    "SmartParallelAudioAnalyzer",
    "KeyDetector",
    "RhythmAnalyzer",
    "MelodyAnalyzer",
    "StructureAnalyzer",
    "SimilarityEngine",
    "AudioLoader",
    "AnalysisConfig",
    "DEFAULT_CONFIG",
    "ResultExporter",
    "BPMDetectorError",
    "AudioLoadError",
    "UnsupportedFormatError",
    "AnalysisError",
]

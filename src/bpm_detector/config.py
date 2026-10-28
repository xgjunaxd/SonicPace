"""Configuration management module for BPM Detector."""

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

from .exceptions import ConfigurationError


@dataclass
class AnalysisConfig:
    """Configuration parameters for audio analysis and BPM detection."""

    sr: int = 22050
    hop_length: int = 512
    n_fft: int = 2048
    min_bpm: float = 60.0
    max_bpm: float = 200.0
    detect_key: bool = True
    comprehensive: bool = False
    auto_parallel: bool = True
    max_workers: Optional[int] = None
    normalize_audio: bool = True
    trim_silence: bool = False
    supported_formats: List[str] = field(
        default_factory=lambda: [".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"]
    )

    def __post_init__(self) -> None:
        """Validate configuration parameters after initialization."""
        self.validate()

    def validate(self) -> None:
        """Validate that all configuration parameters are within acceptable ranges."""
        if self.sr <= 0:
            raise ConfigurationError(f"Sample rate (sr) must be positive, got {self.sr}")
        if self.hop_length <= 0:
            raise ConfigurationError(f"hop_length must be positive, got {self.hop_length}")
        if self.n_fft <= 0 or self.n_fft < self.hop_length:
            raise ConfigurationError(
                f"n_fft ({self.n_fft}) must be positive and >= hop_length ({self.hop_length})"
            )
        if self.min_bpm <= 0 or self.max_bpm <= 0 or self.min_bpm >= self.max_bpm:
            raise ConfigurationError(
                f"Invalid BPM range: min_bpm ({self.min_bpm}) must be < max_bpm ({self.max_bpm})"
            )
        if self.max_workers is not None and self.max_workers < 1:
            raise ConfigurationError(f"max_workers must be >= 1, got {self.max_workers}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to a dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "AnalysisConfig":
        """Create an AnalysisConfig instance from a dictionary."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)


# Default singleton configuration instance
DEFAULT_CONFIG = AnalysisConfig()
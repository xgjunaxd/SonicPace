"""Music theory constants and utilities."""

from typing import Dict, List

# Standard chromatic scale note names
NOTE_NAMES: List[str] = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

# Consonance ratings for musical intervals (0-11 semitones)
# 1.0 is highly consonant (unison, octave), 0.1 is highly dissonant (minor second, tritone)
CONSONANCE_RATINGS: Dict[int, float] = {
    0: 1.0,   # Unison (Perfect consonance)
    1: 0.1,   # Minor Second (Sharp dissonance)
    2: 0.3,   # Major Second (Mild dissonance)
    3: 0.6,   # Minor Third (Imperfect consonance)
    4: 0.7,   # Major Third (Imperfect consonance)
    5: 0.8,   # Perfect Fourth (Perfect consonance)
    6: 0.1,   # Tritone (Sharp dissonance)
    7: 0.9,   # Perfect Fifth (Perfect consonance)
    8: 0.5,   # Minor Sixth (Imperfect consonance)
    9: 0.6,   # Major Sixth (Imperfect consonance)
    10: 0.3,  # Minor Seventh (Mild dissonance)
    11: 0.1   # Major Seventh (Sharp dissonance)
}
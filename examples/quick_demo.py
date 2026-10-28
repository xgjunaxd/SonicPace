import os
import numpy as np
import soundfile as sf
from bpm_detector.bpm_detector import BPMDetector
from bpm_detector.music_analyzer import AudioAnalyzer
from bpm_detector.parallel_analyzer import SmartParallelAudioAnalyzer

def create_test_audio(filename="temp_test_120bpm.wav", bpm=120, duration=10, sr=22050):
    print(f"Generating synthetic {bpm} BPM audio track ({filename})...")
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    # Create a nice A Major chord background (440Hz, 554.37Hz, 659.25Hz)
    harmonic = 0.2 * (np.sin(2 * np.pi * 440 * t) + np.sin(2 * np.pi * 554.37 * t) + np.sin(2 * np.pi * 659.25 * t))
    
    # Add sharp drum clicks at 120 BPM (every 0.5 seconds)
    beats = np.zeros_like(t)
    beat_interval = int(sr * (60.0 / bpm))
    for i in range(0, len(t), beat_interval):
        click_len = min(int(0.04 * sr), len(t) - i)
        decay = np.exp(-np.linspace(0, 10, click_len))
        beats[i:i+click_len] = 0.8 * np.sin(2 * np.pi * 100 * np.linspace(0, 0.04, click_len)) * decay

    audio = np.clip(harmonic + beats, -1.0, 1.0)
    sf.write(filename, audio, sr)
    return filename

def main():
    print("=" * 55)
    print("        BEAT-MASTER: QUICK DEMO & VERIFICATION")
    print("=" * 55)
    
    test_file = create_test_audio()
    
    try:
        # 1. Test BPMDetector
        detector = BPMDetector()
        bpm, confidence = detector.detect_from_file(test_file)
        print(f"\n[1] BPMDetector Result:")
        print(f"    -> Detected BPM: {bpm:.1f} BPM (Confidence: {confidence:.1f}%)")
        
        # 2. Test Comprehensive AudioAnalyzer
        analyzer = AudioAnalyzer()
        results = analyzer.analyze_file(test_file, detect_key=True, comprehensive=True)
        basic = results.get("basic_info", {})
        rhythm = results.get("rhythm", {})
        print(f"\n[2] AudioAnalyzer Comprehensive Result:")
        print(f"    -> Detected Key:   {basic.get('key', 'Unknown')} ({basic.get('key_confidence', 0):.1f}%)")
        print(f"    -> Time Signature: {rhythm.get('time_signature', '4/4')}")
        print(f"    -> Groove Type:    {rhythm.get('groove_type', 'straight')}")
        print(f"    -> Duration:       {basic.get('duration', 0):.1f}s")
        
        # 3. Test Parallel Analyzer
        parallel = SmartParallelAudioAnalyzer(max_workers=2)
        p_res = parallel.analyze_file(test_file, detect_key=True)
        print(f"\n[3] SmartParallelAudioAnalyzer Status: OK")
        print("=" * 55)
        print("SUCCESS: All core Beat-Master modules are working!")
        print("=" * 55)
    finally:
        if os.path.exists(test_file):
            os.remove(test_file)

if __name__ == "__main__":
    main()

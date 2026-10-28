"""Analyze a real audio track with Beat-Master."""

import os
from bpm_detector.music_analyzer import AudioAnalyzer

def main():
    # Put the path to your actual MP3 or WAV file here:
    # Example: r"C:\Users\risha\Music\my_favorite_song.mp3"
    file_path = r"C:/path/to/your/song.mp3"
    
    if not os.path.exists(file_path):
        print(f"Error: Could not find file at '{file_path}'")
        print("Please update the 'file_path' variable in analyze_track.py with your real music file path.")
        return

    print(f"Analyzing '{os.path.basename(file_path)}'...")
    
    # Initialize the comprehensive analyzer
    analyzer = AudioAnalyzer()
    
    # Run full comprehensive analysis (BPM, Key, Rhythm, Structure)
    results = analyzer.analyze_file(file_path, detect_key=True, comprehensive=True)
    
    # Print out a clean report
    print("\n" + "=" * 50)
    print("           BEAT-MASTER ANALYSIS REPORT")
    print("=" * 50)
    
    basic = results.get("basic_info", {})
    print(f"🎵 BPM:          {basic.get('bpm', 0):.1f} (Confidence: {basic.get('bpm_confidence', 0):.1f}%)")
    print(f"🎹 Musical Key:  {basic.get('key', 'Unknown')} (Confidence: {basic.get('key_confidence', 0):.1f}%)")
    print(f"⏱️ Duration:     {basic.get('duration', 0):.1f} seconds")
    
    rhythm = results.get("rhythm", {})
    if rhythm:
        print(f"🥁 Time Sign:    {rhythm.get('time_signature', '4/4')}")
        print(f"🎸 Groove Type:  {rhythm.get('groove_type', 'straight')}")
        print(f"🔥 Syncopation:  {rhythm.get('syncopation_level', 0):.2f}")
        
    structure = results.get("structure", {})
    if structure:
        print(f"🎼 Song Form:    {structure.get('form', 'Unknown')}")
        print(f"📂 Sections:     {structure.get('section_count', 0)} sections detected")
        
    print("=" * 50)

if __name__ == "__main__":
    main()
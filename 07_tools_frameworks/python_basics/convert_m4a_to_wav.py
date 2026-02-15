import sys
import os
from pydub import AudioSegment

def convert_m4a_to_wav(m4a_path, wav_path, target_sample_rate=16000):
    # Load the m4a file
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    # Convert to mono and set frame rate
    audio = audio.set_channels(1).set_frame_rate(target_sample_rate)
    # Export as wav
    audio.export(wav_path, format="wav")
    print(f"Converted {m4a_path} to {wav_path} ({target_sample_rate} Hz, mono)")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python convert_m4a_to_wav.py input.m4a output.wav")
        sys.exit(1)
    m4a_path = sys.argv[1]
    wav_path = sys.argv[2]
    convert_m4a_to_wav(m4a_path, wav_path)

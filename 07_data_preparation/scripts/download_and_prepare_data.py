"""
Script to download a small sample from the LibriTTS dataset and preprocess it for the voice cloning pipeline.
"""
import os
import requests
import zipfile
from pathlib import Path
import librosa
import soundfile as sf

data_dir = Path("data/libritts_sample")
data_dir.mkdir(parents=True, exist_ok=True)

# Download a small sample (dev-clean subset)
url = "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"  # fallback: LJ Speech if LibriTTS sample is too large
libritts_url = "https://www.openslr.org/resources/60/dev-clean.tar.gz"
libritts_tar = data_dir / "dev-clean.tar.gz"
ljspeech_tar = data_dir / "LJSpeech-1.1.tar.bz2"

if not libritts_tar.exists():
    print("Downloading LibriTTS dev-clean sample...")
    r = requests.get(libritts_url, stream=True)
    with open(libritts_tar, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print("Download complete.")
else:
    print("LibriTTS sample already downloaded.")

# Extract the tar.gz file
import tarfile
if libritts_tar.exists():
    print("Extracting LibriTTS dev-clean sample...")
    with tarfile.open(libritts_tar, "r:gz") as tar:
        tar.extractall(path=data_dir)
    print("Extraction complete.")
else:
    print("LibriTTS tar file not found.")

# Preprocess: resample and normalize a few audio files
import glob
audio_files = glob.glob(str(data_dir / "LibriTTS" / "dev-clean" / "**" / "*.wav"), recursive=True)
preprocessed_dir = data_dir / "preprocessed"
preprocessed_dir.mkdir(exist_ok=True)

for wav_path in audio_files[:5]:  # Limit to 5 files for demo
    y, sr = librosa.load(wav_path, sr=16000)
    y = librosa.util.normalize(y)
    out_path = preprocessed_dir / Path(wav_path).name
    sf.write(out_path, y, 16000)
    print(f"Preprocessed: {out_path}")

print("Preprocessing complete. First 5 audio files are resampled and normalized.")

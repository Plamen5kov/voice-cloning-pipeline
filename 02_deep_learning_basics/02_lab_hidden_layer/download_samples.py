"""
Download and prepare audio samples for male vs female voice classification.
Uses LibriSpeech dataset with known speaker genders.
"""

import os
import urllib.request
import tarfile
import soundfile as sf
import librosa
import numpy as np
from pathlib import Path

# LibriSpeech dev-clean subset URL
LIBRISPEECH_URL = "https://www.openslr.org/resources/12/dev-clean.tar.gz"

# Known speaker IDs and their genders from LibriSpeech metadata
# Format: speaker_id: gender ('M' or 'F')
SPEAKER_GENDERS = {
    # Female speakers
    '1272': 'F', '1462': 'F', '1673': 'F', '1919': 'F', '1988': 'F',
    '1993': 'F', '2035': 'F', '2078': 'F', '2086': 'F', '2277': 'F',
    '2412': 'F', '2428': 'F', '2803': 'F', '2902': 'F', '3000': 'F',
    '3081': 'F', '3235': 'F', '3486': 'F', '3526': 'F', '3576': 'F',
    '3607': 'F', '3664': 'F', '3703': 'F', '3723': 'F', '3728': 'F',
    '3807': 'F', '3879': 'F', '3982': 'F', '4051': 'F', '4088': 'F',
    '4160': 'F', '4362': 'F', '4397': 'F', '4446': 'F', '4507': 'F',
    '4640': 'F', '4680': 'F', '4830': 'F', '4853': 'F', '4970': 'F',
    '4992': 'F', '5022': 'F', '5105': 'F', '5120': 'F', '5163': 'F',
    '5192': 'F', '5339': 'F', '5390': 'F', '5393': 'F', '5456': 'F',
    '5463': 'F', '5514': 'F', '5536': 'F', '5556': 'F', '5639': 'F',
    '5649': 'F', '5678': 'F', '5683': 'F', '5703': 'F', '5750': 'F',
    # Male speakers
    '1089': 'M', '1188': 'M', '1246': 'M', '1334': 'M', '1355': 'M',
    '1447': 'M', '1502': 'M', '1578': 'M', '1624': 'M', '1701': 'M',
    '174': 'M', '1743': 'M', '1867': 'M', '1926': 'M', '1963': 'M',
    '1984': 'M', '2002': 'M', '2007': 'M', '2078': 'M', '2094': 'M',
    '2136': 'M', '2159': 'M', '2182': 'M', '2196': 'M', '2289': 'M',
    '2300': 'M', '2391': 'M', '2416': 'M', '251': 'M', '2691': 'M',
    '2836': 'M', '2893': 'M', '2952': 'M', '2961': 'M', '3017': 'M',
    '3112': 'M', '3170': 'M', '3214': 'M', '3242': 'M', '3259': 'M',
    '3296': 'M', '3310': 'M', '3374': 'M', '3436': 'M', '3440': 'M',
    '3575': 'M', '3586': 'M', '3752': 'M', '3853': 'M', '3857': 'M',
    '3883': 'M', '5338': 'M', '5895': 'M', '6829': 'M', '6925': 'M',
    '7021': 'M', '7067': 'M', '7078': 'M', '7113': 'M', '7127': 'M',
    '7176': 'M', '7190': 'M', '7226': 'M', '7278': 'M', '7302': 'M',
    '7312': 'M', '7367': 'M', '777': 'M', '7794': 'M', '7850': 'M',
    '8063': 'M', '8088': 'M', '8297': 'M', '8455': 'M', '8468': 'M',
    '8629': 'M', '908': 'M'
}


def download_and_extract_librispeech(data_dir="data"):
    """Download and extract LibriSpeech dev-clean subset."""
    os.makedirs(data_dir, exist_ok=True)
    
    tar_path = os.path.join(data_dir, "dev-clean.tar.gz")
    extract_dir = data_dir
    
    # Check if already downloaded
    if os.path.exists(os.path.join(data_dir, "LibriSpeech", "dev-clean")):
        print("✓ LibriSpeech dev-clean already downloaded")
        return os.path.join(data_dir, "LibriSpeech", "dev-clean")
    
    # Download
    if not os.path.exists(tar_path):
        print(f"Downloading LibriSpeech dev-clean (~350MB)...")
        print("This may take several minutes...")
        try:
            urllib.request.urlretrieve(LIBRISPEECH_URL, tar_path)
            print("✓ Download complete")
        except Exception as e:
            print(f"✗ Download failed: {e}")
            return None
    
    # Extract
    print("Extracting...")
    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(extract_dir)
        print("✓ Extraction complete")
        os.remove(tar_path)  # Clean up tar file
    except Exception as e:
        print(f"✗ Extraction failed: {e}")
        return None
    
    return os.path.join(data_dir, "LibriSpeech", "dev-clean")


def trim_audio(audio, sr, duration=3.0):
    """Trim or pad audio to specified duration."""
    target_length = int(duration * sr)
    
    if len(audio) > target_length:
        # Trim from middle
        start = (len(audio) - target_length) // 2
        return audio[start:start + target_length]
    elif len(audio) < target_length:
        # Pad with zeros
        padding = target_length - len(audio)
        return np.pad(audio, (0, padding), mode='constant')
    return audio


def collect_audio_samples(librispeech_dir, gender, num_samples, duration=3.0, sr=22050):
    """Collect audio samples for a specific gender."""
    samples = []
    
    # Get speaker IDs for this gender
    speaker_ids = [sid for sid, g in SPEAKER_GENDERS.items() if g == gender]
    
    # First pass: try to collect from available speakers
    print(f"  Searching for {gender} speakers...")
    available_speakers = []
    
    for speaker_id in speaker_ids:
        speaker_dir = Path(librispeech_dir) / speaker_id
        if speaker_dir.exists():
            available_speakers.append(speaker_id)
    
    print(f"  Found {len(available_speakers)} available speakers")
    
    if len(available_speakers) == 0:
        print(f"  Warning: No speakers found for gender {gender}")
        return samples
    
    # Collect multiple samples per speaker to reach target
    # Keep cycling through speakers until we have enough samples
    sample_index = 0
    max_iterations = num_samples * 2  # Prevent infinite loop
    iteration = 0
    
    while len(samples) < num_samples and iteration < max_iterations:
        iteration += 1
        
        for speaker_id in available_speakers:
            if len(samples) >= num_samples:
                break
            
            speaker_dir = Path(librispeech_dir) / speaker_id
            
            # Get all FLAC files for this speaker
            flac_files = sorted(list(speaker_dir.rglob("*.flac")))
            
            # Calculate how many samples we've already taken from this speaker
            start_idx = (iteration - 1) * 2  # Take 2 files per iteration per speaker
            files_to_process = flac_files[start_idx:start_idx + 2]
            
            if not files_to_process:
                continue  # This speaker has no more files
            
            for flac_file in files_to_process:
                if len(samples) >= num_samples:
                    break
                
                try:
                    # Load audio
                    audio, orig_sr = librosa.load(str(flac_file), sr=sr)
                    
                    # Trim to duration
                    audio = trim_audio(audio, sr, duration)
                    
                    samples.append(audio)
                    
                except Exception as e:
                    continue  # Skip files that can't be loaded
    
    print(f"  Collected {len(samples)} samples")
    
    return samples[:num_samples]


def save_samples(samples, output_dir, label_name):
    """Save audio samples to directory."""
    os.makedirs(output_dir, exist_ok=True)
    
    for i, audio in enumerate(samples):
        filename = f"{label_name}_{i:04d}.wav"
        filepath = os.path.join(output_dir, filename)
        sf.write(filepath, audio, 22050)
    
    print(f"✓ Saved {len(samples)} {label_name} samples to {output_dir}")


def main():
    """Main function to download and prepare dataset."""
    print("=" * 60)
    print("Audio Dataset Preparation: Male vs Female Voice Classification")
    print("=" * 60)
    
    # Configuration
    train_samples_per_class = 120
    test_samples_per_class = 40
    duration = 3.0  # seconds
    sr = 22050  # sample rate
    
    data_dir = "data"
    
    # Download LibriSpeech
    print("\n[1/4] Downloading LibriSpeech dataset...")
    librispeech_dir = download_and_extract_librispeech(data_dir)
    
    if librispeech_dir is None:
        print("\n✗ Failed to download dataset. Please check your internet connection.")
        return
    
    # Collect female voice samples
    print("\n[2/4] Collecting female voice samples...")
    female_train = collect_audio_samples(librispeech_dir, 'F', train_samples_per_class, duration, sr)
    female_test = collect_audio_samples(librispeech_dir, 'F', test_samples_per_class, duration, sr)
    
    if len(female_train) < train_samples_per_class:
        print(f"  ⚠ Could only collect {len(female_train)}/{train_samples_per_class} train samples")
    if len(female_test) < test_samples_per_class:
        print(f"  ⚠ Could only collect {len(female_test)}/{test_samples_per_class} test samples")
    
    # Collect male voice samples
    print("\n[3/4] Collecting male voice samples...")
    male_train = collect_audio_samples(librispeech_dir, 'M', train_samples_per_class, duration, sr)
    male_test = collect_audio_samples(librispeech_dir, 'M', test_samples_per_class, duration, sr)
    
    if len(male_train) < train_samples_per_class:
        print(f"  ⚠ Could only collect {len(male_train)}/{train_samples_per_class} train samples")
    if len(male_test) < test_samples_per_class:
        print(f"  ⚠ Could only collect {len(male_test)}/{test_samples_per_class} test samples")
    
    # Save samples
    print("\n[4/4] Saving samples...")
    save_samples(female_train, os.path.join(data_dir, "train", "female"), "female")
    save_samples(female_test, os.path.join(data_dir, "test", "female"), "female")
    save_samples(male_train, os.path.join(data_dir, "train", "male"), "male")
    save_samples(male_test, os.path.join(data_dir, "test", "male"), "male")
    
    # Summary
    print("\n" + "=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"Training samples:   {len(female_train)} female + {len(male_train)} male = {len(female_train) + len(male_train)} total")
    print(f"Test samples:       {len(female_test)} female + {len(male_test)} male = {len(female_test) + len(male_test)} total")
    print(f"Sample duration:    {duration} seconds")
    print(f"Sample rate:        {sr} Hz")
    print("\nNext steps:")
    print("1. Run: python test_setup.py")
    print("2. Open: audio_classification_nn.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    main()

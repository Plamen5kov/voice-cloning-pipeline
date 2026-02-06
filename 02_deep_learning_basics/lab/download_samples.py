"""
Download real audio samples for the speech vs music classification lab.

This script downloads samples from:
- Speech: LibriSpeech (free audiobook dataset)
- Music: GTZAN Dataset (music genre classification dataset)
"""

import urllib.request
import os
import tarfile
import zipfile
from pathlib import Path
import shutil
import librosa
import soundfile as sf
import numpy as np
import subprocess


def download_file_wget(url, destination):
    """Download using wget command (fallback method)."""
    try:
        print(f"Trying wget for {url}...")
        result = subprocess.run(
            ['wget', '-O', str(destination), url],
            check=True,
            capture_output=False
        )
        print("Download complete!")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"wget failed: {e}")
        return False


def download_file(url, destination):
    """Download a file with progress indicator."""
    print(f"Downloading {url}...")
    
    def progress_hook(count, block_size, total_size):
        if total_size > 0:
            percent = int(count * block_size * 100 / total_size)
            print(f"\rProgress: {percent}%", end='')
        else:
            mb_downloaded = count * block_size / (1024 * 1024)
            print(f"\rDownloaded: {mb_downloaded:.1f}MB", end='')
    
    try:
        urllib.request.urlretrieve(url, destination, progress_hook)
        print("\nDownload complete!")
        return True
    except Exception as e:
        print(f"\nPython download failed: {e}")
        print("Trying wget as fallback...")
        return download_file_wget(url, destination)


def extract_archive(archive_path, extract_to):
    """Extract tar.gz or zip archive."""
    print(f"Extracting {archive_path}...")
    
    archive_str = str(archive_path)
    if archive_str.endswith('.tar.gz'):
        with tarfile.open(archive_path, 'r:gz') as tar:
            tar.extractall(extract_to)
    elif archive_str.endswith('.zip'):
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    
    print("Extraction complete!")


def convert_and_copy_audio(source_file, dest_file, target_sr=22050, duration=3.0):
    """
    Load, convert to target format, and save audio file.
    
    Arguments:
    source_file -- path to source audio file
    dest_file -- path to destination .wav file
    target_sr -- target sample rate (default: 22050)
    duration -- target duration in seconds (default: 3.0)
    """
    try:
        # Load audio
        y, sr = librosa.load(source_file, sr=target_sr, duration=duration)
        
        # Pad or trim to exact duration
        target_length = int(target_sr * duration)
        if len(y) < target_length:
            y = np.pad(y, (0, target_length - len(y)), mode='constant')
        else:
            y = y[:target_length]
        
        # Save as WAV
        sf.write(dest_file, y, target_sr)
        return True
    except Exception as e:
        print(f"Error processing {source_file}: {e}")
        return False


def download_librispeech_samples(data_dir, num_train=60, num_test=20):
    """
    Download speech samples from LibriSpeech dev-clean dataset.
    
    Arguments:
    data_dir -- base data directory
    num_train -- number of training samples to extract
    num_test -- number of test samples to extract
    """
    print("\n" + "="*60)
    print("DOWNLOADING SPEECH SAMPLES FROM LIBRISPEECH")
    print("="*60)
    
    data_dir = Path(data_dir)
    temp_dir = data_dir / 'temp'
    temp_dir.mkdir(exist_ok=True)
    
    # Use LibriSpeech dev-clean (small subset, ~330MB)
    url = "https://www.openslr.org/resources/12/dev-clean.tar.gz"
    archive_path = temp_dir / "dev-clean.tar.gz"
    
    # Download
    if not archive_path.exists():
        download_file(url, archive_path)
    else:
        print(f"Archive already exists at {archive_path}")
    
    # Extract
    extract_to = temp_dir / "librispeech"
    if not extract_to.exists():
        extract_archive(archive_path, extract_to)
    else:
        print(f"Already extracted to {extract_to}")
    
    # Find all .flac files
    librispeech_root = extract_to / "LibriSpeech" / "dev-clean"
    audio_files = list(librispeech_root.rglob("*.flac"))
    
    print(f"\nFound {len(audio_files)} speech files")
    print(f"Converting {num_train} for training and {num_test} for testing...")
    
    # Create output directories
    train_speech_dir = data_dir / 'train' / 'speech'
    test_speech_dir = data_dir / 'test' / 'speech'
    train_speech_dir.mkdir(parents=True, exist_ok=True)
    test_speech_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert training samples
    count = 0
    for i, audio_file in enumerate(audio_files[:num_train]):
        dest_file = train_speech_dir / f"speech_{i:03d}.wav"
        if convert_and_copy_audio(audio_file, dest_file):
            count += 1
            if count % 10 == 0:
                print(f"Processed {count}/{num_train} training samples...")
    
    print(f"✓ Created {count} training speech samples")
    
    # Convert test samples
    count = 0
    for i, audio_file in enumerate(audio_files[num_train:num_train+num_test]):
        dest_file = test_speech_dir / f"speech_{i:03d}.wav"
        if convert_and_copy_audio(audio_file, dest_file):
            count += 1
    
    print(f"✓ Created {count} test speech samples")
    
    # Cleanup
    print("\nCleaning up temporary files...")
    shutil.rmtree(temp_dir)
    print("✓ Speech samples ready!")


def create_test_music_samples(data_dir, num_train=60, num_test=20):
    """
    Create simple test music samples by transforming speech.
    This is ONLY for testing the lab structure - replace with real music!
    
    Arguments:
    data_dir -- base data directory
    num_train -- number of training samples
    num_test -- number of test samples
    """
    print("\n" + "="*60)
    print("CREATING TEST MUSIC SAMPLES")
    print("="*60)
    print("\n⚠️  WARNING: Creating synthetic test samples")
    print("These are NOT real music - just modified speech for testing!")
    print("For real training, please add actual music files.\n")
    
    data_dir = Path(data_dir)
    
    # Check if we have speech samples to transform
    train_speech_dir = data_dir / 'train' / 'speech'
    test_speech_dir = data_dir / 'test' / 'speech'
    
    if not train_speech_dir.exists() or len(list(train_speech_dir.glob('*.wav'))) == 0:
        print("❌ No speech samples found!")
        print("Please run option 1 first to download speech samples.")
        return
    
    train_music_dir = data_dir / 'train' / 'music'
    test_music_dir = data_dir / 'test' / 'music'
    train_music_dir.mkdir(parents=True, exist_ok=True)
    test_music_dir.mkdir(parents=True, exist_ok=True)
    
    sr = 22050
    
    # Get available speech files
    speech_files = list(train_speech_dir.glob('*.wav'))
    
    print(f"Creating {num_train} training music samples...")
    # Create training samples by modifying speech
    for i in range(min(num_train, len(speech_files))):
        try:
            # Load speech
            y, _ = librosa.load(speech_files[i % len(speech_files)], sr=sr, duration=3.0)
            
            # Apply transformations to make it sound different
            # 1. Pitch shift down
            y_music = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4)
            # 2. Add harmonics
            y_harmonic, _ = librosa.effects.hpss(y_music)
            # 3. Time stretch
            y_final = librosa.effects.time_stretch(y_harmonic, rate=0.9)
            
            # Ensure correct length
            target_length = int(sr * 3.0)
            if len(y_final) < target_length:
                y_final = np.pad(y_final, (0, target_length - len(y_final)))
            else:
                y_final = y_final[:target_length]
            
            output_file = train_music_dir / f'music_{i:03d}.wav'
            sf.write(output_file, y_final, sr)
            
            if (i + 1) % 10 == 0:
                print(f"  Created {i + 1}/{num_train} samples...")
        except Exception as e:
            print(f"Error creating sample {i}: {e}")
    
    print(f"✓ Created {min(num_train, len(speech_files))} training samples")
    
    # Create test samples
    test_speech_files = list(test_speech_dir.glob('*.wav'))
    print(f"Creating {num_test} test music samples...")
    
    for i in range(min(num_test, len(test_speech_files))):
        try:
            y, _ = librosa.load(test_speech_files[i % len(test_speech_files)], sr=sr, duration=3.0)
            y_music = librosa.effects.pitch_shift(y, sr=sr, n_steps=-4)
            y_harmonic, _ = librosa.effects.hpss(y_music)
            y_final = librosa.effects.time_stretch(y_harmonic, rate=0.9)
            
            target_length = int(sr * 3.0)
            if len(y_final) < target_length:
                y_final = np.pad(y_final, (0, target_length - len(y_final)))
            else:
                y_final = y_final[:target_length]
            
            output_file = test_music_dir / f'music_{i:03d}.wav'
            sf.write(output_file, y_final, sr)
        except Exception as e:
            print(f"Error creating test sample {i}: {e}")
    
    print(f"✓ Created {min(num_test, len(test_speech_files))} test samples")
    print("\n⚠️  IMPORTANT: Replace these with real music files!")
    print("   Add .wav files to:")
    print(f"   - {train_music_dir.absolute()}")
    print(f"   - {test_music_dir.absolute()}")
    print("✓ Test samples ready!")
    """
    Download and prepare music samples from GTZAN dataset.
    
    The GTZAN dataset contains 1000 music tracks across 10 genres.
    This function downloads it and extracts the requested number of samples.
    
    Arguments:
    data_dir -- base data directory
    num_train -- number of training samples (default: 60)
    num_test -- number of test samples (default: 20)
    """
    print("\n" + "="*60)
    print("DOWNLOADING MUSIC SAMPLES FROM GTZAN DATASET")
    print("="*60)
    print("\nGTZAN Dataset: 1000 music tracks, 10 genres")
    print("Download size: ~1.2GB")
    print(f"Will extract {num_train} training + {num_test} test samples\n")
    
    data_dir = Path(data_dir)
    temp_dir = data_dir / 'temp_music'
    temp_dir.mkdir(exist_ok=True)
    
    # Try multiple mirror URLs
    urls = [
        "http://opihi.cs.uvic.ca/sound/genres.tar.gz",
        "https://github.com/mdeff/fma/releases/download/gtzan/genres.tar.gz",
    ]
    
    archive_path = temp_dir / "genres.tar.gz"
    
    # Download if not already present
    if not archive_path.exists():
        print("Attempting to download GTZAN dataset...")
        print("This may take several minutes (~1.2GB download)...\n")
        
        download_success = False
        for i, url in enumerate(urls):
            print(f"\nTrying mirror {i+1}/{len(urls)}: {url}")
            if download_file(url, archive_path):
                download_success = True
                break
            else:
                print(f"Mirror {i+1} failed, trying next...")
                if archive_path.exists():
                    archive_path.unlink()  # Remove partial download
        
        if not download_success:
            print("\n" + "="*60)
            print("AUTOMATIC DOWNLOAD FAILED")
            print("="*60)
            print("\nPlease download the GTZAN dataset manually:\n")
            print("Option 1 - Direct Download:")
            print("  wget http://opihi.cs.uvic.ca/sound/genres.tar.gz")
            print(f"  mv genres.tar.gz {temp_dir.absolute()}/\n")
            print("Option 2 - Kaggle:")
            print("  Visit: https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification")
            print(f"  Download and place genres.tar.gz in: {temp_dir.absolute()}/\n")
            print("Then run this script again.")
            return
    else:
        print(f"✓ Archive already exists at {archive_path}")
    
    if archive_path.exists():
        print(f"✓ Found archive at {archive_path}")
        
        # Extract
        extract_to = temp_dir / "gtzan"
        if not extract_to.exists():
            extract_archive(archive_path, extract_to)
        
        # Find all audio files
        genres_dir = extract_to / "genres"
        if not genres_dir.exists():
            # Try alternative structure
            genres_dir = extract_to
        
        audio_files = list(genres_dir.rglob("*.wav")) + list(genres_dir.rglob("*.au"))
        
        if len(audio_files) == 0:
            print("\n❌ No audio files found in archive!")
            print("Please check the archive structure and try again.")
            return
        
        print(f"\n✓ Found {len(audio_files)} music files")
        print(f"Converting {num_train} for training and {num_test} for testing...")
        
        # Shuffle for variety
        import random
        random.shuffle(audio_files)
        
        # Create output directories
        train_music_dir = data_dir / 'train' / 'music'
        test_music_dir = data_dir / 'test' / 'music'
        train_music_dir.mkdir(parents=True, exist_ok=True)
        test_music_dir.mkdir(parents=True, exist_ok=True)
        
        # Convert training samples
        count = 0
        for i, audio_file in enumerate(audio_files[:num_train]):
            dest_file = train_music_dir / f"music_{i:03d}.wav"
            if convert_and_copy_audio(audio_file, dest_file):
                count += 1
                if count % 10 == 0:
                    print(f"Processed {count}/{num_train} training samples...")
        
        print(f"✓ Created {count} training music samples")
        
        # Convert test samples
        count = 0
        for i, audio_file in enumerate(audio_files[num_train:num_train+num_test]):
            dest_file = test_music_dir / f"music_{i:03d}.wav"
            if convert_and_copy_audio(audio_file, dest_file):
                count += 1
        
        print(f"✓ Created {count} test music samples")
        
        # Cleanup
        print("\nCleaning up temporary files...")
        shutil.rmtree(temp_dir)
        print("✓ Music samples ready!")
        
    else:
        print(f"\n⚠ Archive not found at {archive_path}")
        print("\nPlease download the GTZAN dataset following the instructions above,")
        print(f"place genres.tar.gz in {temp_dir.absolute()}/")
        print("and run this script again.")





def main():
    """Main function to download all samples."""
    print("\n" + "="*60)
    print("AUDIO SAMPLE DOWNLOADER FOR SPEECH VS MUSIC LAB")
    print("="*60)
    
    # Configuration
    data_dir = Path('data')
    num_train_per_class = 60
    num_test_per_class = 20
    
    print(f"\nData directory: {data_dir.absolute()}")
    print(f"Training samples per class: {num_train_per_class}")
    print(f"Test samples per class: {num_test_per_class}")
    print(f"Total: {(num_train_per_class + num_test_per_class) * 2} audio files")
    
    # Ask user what to download
    print("\n" + "="*60)
    print("DOWNLOAD OPTIONS")
    print("="*60)
    print("1. Download speech samples (LibriSpeech - ~330MB, automatic)")
    print("2. Create test music samples (modified speech - for testing only)")
    print("3. Download speech + create test music (quick start)")
    print("4. Show instructions for adding real music files")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    
    if choice == '1':
        download_librispeech_samples(data_dir, num_train_per_class, num_test_per_class)
    elif choice == '2':
        create_test_music_samples(data_dir, num_train_per_class, num_test_per_class)
    elif choice == '3':
        download_librispeech_samples(data_dir, num_train_per_class, num_test_per_class)
        create_test_music_samples(data_dir, num_train_per_class, num_test_per_class)
    elif choice == '4':
        print("\n" + "="*60)
        print("HOW TO ADD REAL MUSIC FILES")
        print("="*60)
        print("\n1. Collect music files (.wav, .mp3, .flac, etc.)")
        print(f"   Need {num_train_per_class} training + {num_test_per_class} test files")
        print("\n2. Recommended sources:")
        print("   - Your personal music library")
        print("   - Free Music Archive: https://freemusicarchive.org/")
        print("   - YouTube Audio Library: https://youtube.com/audiolibrary")
        print("   - GTZAN (if accessible): http://marsyas.info/downloads/datasets.html")
        print("\n3. Place files in:")
        print(f"   Training: {(data_dir / 'train' / 'music').absolute()}/")
        print(f"   Testing:  {(data_dir / 'test' / 'music').absolute()}/")
        print("\n4. Files will be automatically converted to 3-second .wav format")
        print("   when you run the notebook.\n")
        return
    else:
        print("Invalid choice!")
        return
    
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    
    # Count files
    train_speech = len(list((data_dir / 'train' / 'speech').glob('*.wav'))) if (data_dir / 'train' / 'speech').exists() else 0
    train_music = len(list((data_dir / 'train' / 'music').glob('*.wav'))) if (data_dir / 'train' / 'music').exists() else 0
    test_speech = len(list((data_dir / 'test' / 'speech').glob('*.wav'))) if (data_dir / 'test' / 'speech').exists() else 0
    test_music = len(list((data_dir / 'test' / 'music').glob('*.wav'))) if (data_dir / 'test' / 'music').exists() else 0
    
    print(f"\nTraining samples:")
    print(f"  Speech: {train_speech}")
    print(f"  Music:  {train_music}")
    print(f"\nTest samples:")
    print(f"  Speech: {test_speech}")
    print(f"  Music:  {test_music}")
    
    total = train_speech + train_music + test_speech + test_music
    print(f"\nTotal audio files: {total}")
    
    if train_speech > 0 and train_music > 0 and test_speech > 0 and test_music > 0:
        print("\n✓ Dataset is complete and ready!")
        print("\nNext step: Run audio_classification_logreg.ipynb notebook")
    elif train_speech > 0 or test_speech > 0:
        print("\n⚠ You have speech samples but no music samples yet.")
        print("Run this script again and choose option 2 to download music.")
    elif train_music > 0 or test_music > 0:
        print("\n⚠ You have music samples but no speech samples yet.")
        print("Run this script again and choose option 1 to download speech.")
    else:
        print("\n⚠ No samples downloaded yet.")
        print("Run this script again to download the dataset.")


if __name__ == "__main__":
    main()

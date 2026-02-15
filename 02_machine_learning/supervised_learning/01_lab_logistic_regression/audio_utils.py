"""
Audio utilities for loading and preprocessing audio data for logistic regression.
"""

import numpy as np
import librosa
import os
from pathlib import Path


def load_audio_file(filepath, sr=22050, duration=3.0, n_mels=128, n_fft=2048, hop_length=512):
    """
    Load an audio file and convert it to a mel-spectrogram.
    
    Arguments:
    filepath -- path to the audio file
    sr -- sample rate (default: 22050 Hz)
    duration -- duration to load in seconds (default: 3.0 seconds)
    n_mels -- number of mel bands (default: 128)
    n_fft -- FFT window size (default: 2048)
    hop_length -- hop length for STFT (default: 512)
    
    Returns:
    mel_spec -- mel-spectrogram as numpy array
    """
    # Load audio file
    y, _ = librosa.load(filepath, sr=sr, duration=duration)
    
    # Pad or trim to exact duration
    target_length = int(sr * duration)
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    else:
        y = y[:target_length]
    
    # Convert to mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=y, 
        sr=sr, 
        n_mels=n_mels,
        n_fft=n_fft,
        hop_length=hop_length
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1] range
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_norm


def load_dataset(data_path='data', sr=22050, duration=3.0, n_mels=128):
    """
    Load audio dataset from directory structure and convert to spectrograms.
    
    Expected directory structure:
    data/
        train/
            speech/ (label = 1)
            music/  (label = 0)
        test/
            speech/
            music/
    
    Arguments:
    data_path -- path to the data directory
    sr -- sample rate
    duration -- duration of each audio clip
    n_mels -- number of mel bands
    
    Returns:
    train_set_x_orig -- training spectrograms (m_train, n_mels, time_steps)
    train_set_y -- training labels (1, m_train)
    test_set_x_orig -- test spectrograms (m_test, n_mels, time_steps)
    test_set_y -- test labels (1, m_test)
    classes -- array of class names
    """
    data_path = Path(data_path)
    
    # Define classes
    classes = np.array([b"music", b"speech"])
    
    # Load training data
    train_speech_path = data_path / 'train' / 'speech'
    train_music_path = data_path / 'train' / 'music'
    
    train_spectrograms = []
    train_labels = []
    
    # Load speech samples (label = 1)
    if train_speech_path.exists():
        for audio_file in sorted(train_speech_path.glob('*.wav')):
            try:
                mel_spec = load_audio_file(str(audio_file), sr=sr, duration=duration, n_mels=n_mels)
                train_spectrograms.append(mel_spec)
                train_labels.append(1)
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
    
    # Load music samples (label = 0)
    if train_music_path.exists():
        for audio_file in sorted(train_music_path.glob('*.wav')):
            try:
                mel_spec = load_audio_file(str(audio_file), sr=sr, duration=duration, n_mels=n_mels)
                train_spectrograms.append(mel_spec)
                train_labels.append(0)
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
    
    # Load test data
    test_speech_path = data_path / 'test' / 'speech'
    test_music_path = data_path / 'test' / 'music'
    
    test_spectrograms = []
    test_labels = []
    
    # Load speech samples (label = 1)
    if test_speech_path.exists():
        for audio_file in sorted(test_speech_path.glob('*.wav')):
            try:
                mel_spec = load_audio_file(str(audio_file), sr=sr, duration=duration, n_mels=n_mels)
                test_spectrograms.append(mel_spec)
                test_labels.append(1)
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
    
    # Load music samples (label = 0)
    if test_music_path.exists():
        for audio_file in sorted(test_music_path.glob('*.wav')):
            try:
                mel_spec = load_audio_file(str(audio_file), sr=sr, duration=duration, n_mels=n_mels)
                test_spectrograms.append(mel_spec)
                test_labels.append(0)
            except Exception as e:
                print(f"Error loading {audio_file}: {e}")
    
    # Convert to numpy arrays
    if len(train_spectrograms) == 0:
        print("WARNING: No training data found! Please add audio files to data/train/speech/ and data/train/music/")
        # Return dummy data for demonstration
        train_set_x_orig = np.zeros((10, n_mels, 130))
        train_set_y = np.array([[1, 1, 1, 1, 1, 0, 0, 0, 0, 0]])
    else:
        train_set_x_orig = np.array(train_spectrograms)
        train_set_y = np.array([train_labels])
    
    if len(test_spectrograms) == 0:
        print("WARNING: No test data found! Please add audio files to data/test/speech/ and data/test/music/")
        # Return dummy data for demonstration
        test_set_x_orig = np.zeros((4, n_mels, 130))
        test_set_y = np.array([[1, 1, 0, 0]])
    else:
        test_set_x_orig = np.array(test_spectrograms)
        test_set_y = np.array([test_labels])
    
    return train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes


def generate_sample_audio(output_path='data', num_train_speech=50, num_train_music=50, 
                          num_test_speech=15, num_test_music=15):
    """
    Generate sample audio files for testing (simple tones and noise).
    This is just for demonstration - replace with real audio data.
    
    Arguments:
    output_path -- base path for data directory
    num_train_speech -- number of training speech samples
    num_train_music -- number of training music samples
    num_test_speech -- number of test speech samples
    num_test_music -- number of test music samples
    """
    import soundfile as sf
    
    output_path = Path(output_path)
    sr = 22050
    duration = 3.0
    
    print("Generating sample audio files...")
    print("NOTE: These are synthetic sounds for testing only.")
    print("For real training, replace with actual speech and music files!\n")
    
    # Generate training speech (higher frequency tones with variations)
    for i in range(num_train_speech):
        t = np.linspace(0, duration, int(sr * duration))
        freq = 200 + np.random.randint(0, 300)  # Simulate speech frequencies
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        # Add harmonics
        audio += 0.15 * np.sin(2 * np.pi * freq * 2 * t)
        audio += 0.1 * np.sin(2 * np.pi * freq * 3 * t)
        # Add some noise
        audio += 0.05 * np.random.randn(len(t))
        
        output_file = output_path / 'train' / 'speech' / f'speech_{i:03d}.wav'
        sf.write(output_file, audio, sr)
    
    # Generate training music (multiple frequencies, more harmonic)
    for i in range(num_train_music):
        t = np.linspace(0, duration, int(sr * duration))
        # Chord-like structure
        freq1 = 220 + np.random.randint(-20, 20)
        freq2 = freq1 * 1.5  # Fifth
        freq3 = freq1 * 2.0  # Octave
        
        audio = 0.2 * np.sin(2 * np.pi * freq1 * t)
        audio += 0.2 * np.sin(2 * np.pi * freq2 * t)
        audio += 0.15 * np.sin(2 * np.pi * freq3 * t)
        # Add rhythm
        envelope = np.abs(np.sin(2 * np.pi * 2 * t))
        audio = audio * envelope
        
        output_file = output_path / 'train' / 'music' / f'music_{i:03d}.wav'
        sf.write(output_file, audio, sr)
    
    # Generate test speech
    for i in range(num_test_speech):
        t = np.linspace(0, duration, int(sr * duration))
        freq = 200 + np.random.randint(0, 300)
        audio = 0.3 * np.sin(2 * np.pi * freq * t)
        audio += 0.15 * np.sin(2 * np.pi * freq * 2 * t)
        audio += 0.1 * np.sin(2 * np.pi * freq * 3 * t)
        audio += 0.05 * np.random.randn(len(t))
        
        output_file = output_path / 'test' / 'speech' / f'speech_{i:03d}.wav'
        sf.write(output_file, audio, sr)
    
    # Generate test music
    for i in range(num_test_music):
        t = np.linspace(0, duration, int(sr * duration))
        freq1 = 220 + np.random.randint(-20, 20)
        freq2 = freq1 * 1.5
        freq3 = freq1 * 2.0
        
        audio = 0.2 * np.sin(2 * np.pi * freq1 * t)
        audio += 0.2 * np.sin(2 * np.pi * freq2 * t)
        audio += 0.15 * np.sin(2 * np.pi * freq3 * t)
        envelope = np.abs(np.sin(2 * np.pi * 2 * t))
        audio = audio * envelope
        
        output_file = output_path / 'test' / 'music' / f'music_{i:03d}.wav'
        sf.write(output_file, audio, sr)
    
    print(f"Generated {num_train_speech + num_train_music} training samples")
    print(f"Generated {num_test_speech + num_test_music} test samples")
    print("\nReplace these with real audio files for actual training!")

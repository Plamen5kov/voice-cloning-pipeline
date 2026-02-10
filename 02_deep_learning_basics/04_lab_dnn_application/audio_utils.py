"""
Audio Utilities for Deep Neural Network Application

This module provides helper functions for loading and processing audio data.
"""

import numpy as np
import os
import librosa


def load_and_process_audio(audio_path, sr=22050, duration=3, n_mels=128):
    """
    Load an audio file and convert it to a mel-spectrogram
    
    Arguments:
    audio_path -- path to the audio file
    sr -- sample rate (default: 22050 Hz)
    duration -- duration in seconds (default: 3 seconds)
    n_mels -- number of mel frequency bins (default: 128)
    
    Returns:
    mel_spec -- mel-spectrogram as numpy array
    """
    # Load audio file
    y, sr = librosa.load(audio_path, sr=sr, duration=duration)
    
    # Pad if shorter than duration
    target_length = sr * duration
    if len(y) < target_length:
        y = np.pad(y, (0, target_length - len(y)), mode='constant')
    
    # Compute mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    
    # Convert to dB scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 255] range (similar to image pixel values)
    mel_spec_normalized = ((mel_spec_db - mel_spec_db.min()) / 
                          (mel_spec_db.max() - mel_spec_db.min()) * 255)
    
    return mel_spec_normalized


def load_audio_dataset(data_dir='data', sr=22050, duration=3, n_mels=128):
    """
    Load the voice gender dataset from directory structure
    
    Expected directory structure:
    data/
        train/
            male/
            female/
        test/
            male/
            female/
    
    Arguments:
    data_dir -- root data directory
    sr -- sample rate
    duration -- audio duration in seconds
    n_mels -- number of mel frequency bins
    
    Returns:
    train_x_orig -- training spectrograms (m_train, n_mels, time_steps)
    train_y -- training labels (1, m_train) - 0=male, 1=female
    test_x_orig -- test spectrograms (m_test, n_mels, time_steps)
    test_y -- test labels (1, m_test) - 0=male, 1=female
    classes -- list of class names
    """
    
    classes = ['male', 'female']
    
    # Initialize lists
    train_spectrograms = []
    train_labels = []
    test_spectrograms = []
    test_labels = []
    
    # Load training data
    train_dir = os.path.join(data_dir, 'train')
    if os.path.exists(train_dir):
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(train_dir, class_name)
            if os.path.exists(class_dir):
                for audio_file in os.listdir(class_dir):
                    if audio_file.endswith(('.wav', '.mp3', '.flac')):
                        audio_path = os.path.join(class_dir, audio_file)
                        try:
                            spec = load_and_process_audio(audio_path, sr, duration, n_mels)
                            train_spectrograms.append(spec)
                            train_labels.append(class_idx)
                        except Exception as e:
                            print(f"Error loading {audio_path}: {e}")
    
    # Load test data
    test_dir = os.path.join(data_dir, 'test')
    if os.path.exists(test_dir):
        for class_idx, class_name in enumerate(classes):
            class_dir = os.path.join(test_dir, class_name)
            if os.path.exists(class_dir):
                for audio_file in os.listdir(class_dir):
                    if audio_file.endswith(('.wav', '.mp3', '.flac')):
                        audio_path = os.path.join(class_dir, audio_file)
                        try:
                            spec = load_and_process_audio(audio_path, sr, duration, n_mels)
                            test_spectrograms.append(spec)
                            test_labels.append(class_idx)
                        except Exception as e:
                            print(f"Error loading {audio_path}: {e}")
    
    # If no data found, create dummy data for testing
    if len(train_spectrograms) == 0 or len(test_spectrograms) == 0:
        print("Warning: No audio data found. Creating dummy dataset for testing.")
        print("Please add audio files to data/train/ and data/test/ directories.")
        
        # Create dummy data
        m_train = 100
        m_test = 20
        time_steps = int(sr * duration / 512) + 1  # Approximate time steps
        
        train_spectrograms = np.random.randn(m_train, n_mels, time_steps) * 50 + 128
        train_labels = np.random.randint(0, len(classes), m_train)
        test_spectrograms = np.random.randn(m_test, n_mels, time_steps) * 50 + 128
        test_labels = np.random.randint(0, len(classes), m_test)
    else:
        # Convert to numpy arrays
        train_spectrograms = np.array(train_spectrograms)
        test_spectrograms = np.array(test_spectrograms)
    
    # Convert labels to (1, m) shape
    train_y = np.array(train_labels).reshape(1, -1)
    test_y = np.array(test_labels).reshape(1, -1)
    
    print(f"Loaded {len(train_spectrograms)} training samples")
    print(f"Loaded {len(test_spectrograms)} test samples")
    print(f"Classes: {classes}")
    
    return train_spectrograms, train_y, test_spectrograms, test_y, classes

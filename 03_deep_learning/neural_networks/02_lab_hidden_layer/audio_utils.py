"""
Audio utilities for loading and preprocessing audio data.
"""

import os
import numpy as np
import librosa
import soundfile as sf


def load_audio_file(filepath, sr=22050, n_mels=128, duration=3.0):
    """
    Load an audio file and convert to mel-spectrogram.
    
    Args:
        filepath: Path to audio file
        sr: Sample rate
        n_mels: Number of mel bands
        duration: Expected duration in seconds
    
    Returns:
        mel_spectrogram: Normalized mel-spectrogram of shape (n_mels, time_steps)
    """
    # Load audio
    audio, _ = librosa.load(filepath, sr=sr, duration=None)
    
    # Ensure exact length by padding or trimming
    target_length = int(duration * sr)
    if len(audio) > target_length:
        # Trim from center
        start = (len(audio) - target_length) // 2
        audio = audio[start:start + target_length]
    elif len(audio) < target_length:
        # Pad with zeros
        padding = target_length - len(audio)
        audio = np.pad(audio, (0, padding), mode='constant')
    
    # Compute mel-spectrogram with fixed parameters
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        n_fft=2048,
        hop_length=512
    )
    
    # Convert to log scale (dB)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize to [0, 1]
    mel_spec_normalized = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min() + 1e-8)
    
    return mel_spec_normalized


def load_dataset(data_dir="data", sr=22050, n_mels=128):
    """
    Load the complete dataset (train and test sets).
    
    Args:
        data_dir: Root directory containing train/ and test/ folders
        sr: Sample rate
        n_mels: Number of mel bands
    
    Returns:
        train_x: Training features of shape (n_features, m_train)
        train_y: Training labels of shape (1, m_train) - 0 for female, 1 for male
        test_x: Test features of shape (n_features, m_test)
        test_y: Test labels of shape (1, m_test)
        classes: List of class names ['female', 'male']
    """
    
    classes = ['female', 'male']
    
    # Load training data
    print("Loading training data...")
    train_x_list = []
    train_y_list = []
    
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, "train", class_name)
        
        if not os.path.exists(class_dir):
            raise ValueError(f"Training directory not found: {class_dir}")
        
        audio_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.wav')])
        
        for audio_file in audio_files:
            filepath = os.path.join(class_dir, audio_file)
            
            # Load and preprocess
            mel_spec = load_audio_file(filepath, sr=sr, n_mels=n_mels)
            
            # Flatten to 1D feature vector
            features = mel_spec.flatten()
            
            train_x_list.append(features)
            train_y_list.append(label_idx)
        
        print(f"  Loaded {len(audio_files)} {class_name} samples")
    
    # Convert to numpy arrays
    train_x = np.array(train_x_list).T  # Shape: (n_features, m_train)
    train_y = np.array(train_y_list).reshape(1, -1)  # Shape: (1, m_train)
    
    # Load test data
    print("Loading test data...")
    test_x_list = []
    test_y_list = []
    
    for label_idx, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, "test", class_name)
        
        if not os.path.exists(class_dir):
            raise ValueError(f"Test directory not found: {class_dir}")
        
        audio_files = sorted([f for f in os.listdir(class_dir) if f.endswith('.wav')])
        
        for audio_file in audio_files:
            filepath = os.path.join(class_dir, audio_file)
            
            # Load and preprocess
            mel_spec = load_audio_file(filepath, sr=sr, n_mels=n_mels)
            
            # Flatten to 1D feature vector
            features = mel_spec.flatten()
            
            test_x_list.append(features)
            test_y_list.append(label_idx)
        
        print(f"  Loaded {len(audio_files)} {class_name} samples")
    
    # Convert to numpy arrays
    test_x = np.array(test_x_list).T  # Shape: (n_features, m_test)
    test_y = np.array(test_y_list).reshape(1, -1)  # Shape: (1, m_test)
    
    print(f"\nDataset loaded successfully!")
    print(f"Training set: {train_x.shape}")
    print(f"Test set: {test_x.shape}")
    
    return train_x, train_y, test_x, test_y, classes


def sigmoid(z):
    """
    Compute the sigmoid of z.
    
    Args:
        z: A scalar or numpy array of any size
    
    Returns:
        s: sigmoid(z)
    """
    s = 1 / (1 + np.exp(-z))
    return s

"""
Utility functions for TTS scripts.
Common functionality to reduce code duplication and improve maintainability.
"""

import torch
from TTS.api import TTS
import os
from pathlib import Path
from typing import Optional


def get_device():
    """
    Get the best available device (CUDA or CPU).
    
    Returns:
        str: 'cuda' if available, otherwise 'cpu'
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    return device


def load_tts_model(model_name="tts_models/multilingual/multi-dataset/xtts_v2", device=None):
    """
    Load a TTS model and move it to the specified device.
    
    Args:
        model_name: Name of the TTS model to load
        device: Device to use ('cuda' or 'cpu'). If None, auto-detect.
    
    Returns:
        TTS: Loaded TTS model
    """
    if device is None:
        device = get_device()
    
    print(f"Loading {model_name}...")
    tts = TTS(model_name=model_name).to(device)
    return tts


def ensure_directory(path):
    """
    Create directory if it doesn't exist.
    
    Args:
        path: Directory path to create
    
    Returns:
        Path: Path object of the created/existing directory
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def generate_output_path(output_dir, filename, create_dir=True):
    """
    Generate an output file path and optionally create the directory.
    
    Args:
        output_dir: Output directory path
        filename: Name of the output file
        create_dir: Whether to create the directory if it doesn't exist
    
    Returns:
        str: Full output file path
    """
    if create_dir:
        ensure_directory(output_dir)
    return os.path.join(output_dir, filename)


def synthesize_speech(tts, text, output_file, speaker=None, speaker_wav=None, language="en"):
    """
    Synthesize speech from text using either a speaker name or speaker WAV file.
    
    Args:
        tts: TTS model instance
        text: Text to synthesize
        output_file: Output file path
        speaker: Speaker name (for predefined speakers)
        speaker_wav: Path to speaker WAV file (for voice cloning)
        language: Language code (default: 'en')
    
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        if speaker_wav and os.path.isfile(speaker_wav):
            # Voice cloning mode
            tts.tts_to_file(
                text=text,
                speaker_wav=speaker_wav,
                language=language,
                file_path=output_file
            )
        elif speaker:
            # Predefined speaker mode
            tts.tts_to_file(
                text=text,
                speaker=speaker,
                language=language,
                file_path=output_file
            )
        else:
            raise ValueError("Either 'speaker' or 'speaker_wav' must be provided")
        
        return True
    except Exception as e:
        print(f"Error during synthesis: {e}")
        return False


def sanitize_filename(name, max_length=50):
    """
    Convert a string into a safe filename.
    
    Args:
        name: String to convert
        max_length: Maximum length of the filename
    
    Returns:
        str: Sanitized filename
    """
    # Replace spaces and special characters
    safe_name = name.replace(' ', '_').lower()
    # Remove any character that isn't alphanumeric, underscore, or hyphen
    safe_name = ''.join(c for c in safe_name if c.isalnum() or c in ('_', '-'))
    # Truncate if too long
    return safe_name[:max_length]


def print_progress(current, total, message=""):
    """
    Print a progress indicator.
    
    Args:
        current: Current item number
        total: Total number of items
        message: Optional message to display
    """
    prefix = f"[{current}/{total}]"
    if message:
        print(f"{prefix} {message}")
    else:
        print(prefix)


def print_summary(successful, failed, output_dir):
    """
    Print a summary of batch operations.
    
    Args:
        successful: Number of successful operations
        failed: Number of failed operations
        output_dir: Output directory path
    """
    print(f"\n{'='*60}")
    print(f"Operation complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Output directory: {output_dir}")
    print(f"{'='*60}")

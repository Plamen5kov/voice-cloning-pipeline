# Speech & Audio Processing - Learning Guide

## 🎯 Module Overview

Master the fundamentals of digital audio processing, feature extraction, and audio analysis essential for voice cloning and speech synthesis projects.

## 📚 What You'll Learn

- Audio fundamentals (sampling, frequency, amplitude)
- Waveform and spectrogram analysis
- Feature extraction (MFCCs, mel-spectrograms)
- Audio preprocessing and augmentation
- Speaker diarization and separation
- Audio quality enhancement

## 🎓 Learning Objectives

- [ ] Load and visualize audio waveforms
- [ ] Compute and interpret spectrograms
- [ ] Extract MFCC features for ML models
- [ ] Normalize and preprocess audio
- [ ] Remove noise and enhance quality
- [ ] Split and segment audio files

## 📝 Key Concepts

### Audio Fundamentals
- **Sampling Rate**: 16kHz (speech), 44.1kHz (music), 24kHz (TTS)
- **Bit Depth**: 16-bit standard, 24/32-bit for production
- **Channels**: Mono (1) for speech, Stereo (2) for music

### Feature Extraction
- **Waveform**: Raw audio amplitude over time
- **Spectrogram**: Frequency content over time (2D visualization)
- **Mel-Spectrogram**: Perceptually-weighted frequency representation
- **MFCCs**: Compact representation for speech (13-40 coefficients)

## 🚀 Exercises & Tasks

### Task 1: Audio Visualization
- Load WAV file with librosa
- Plot waveform
- Compute and display spectrogram
- Analyze frequency content

### Task 2: MFCC Extraction
- Extract MFCC features
- Visualize as heatmap
- Save features for ML training

### Task 3: Audio Preprocessing
- Resample to 24kHz
- Convert to mono
- Normalize volume
- Trim silence

### Task 4: Noise Reduction
- Identify background noise
- Apply spectral subtraction
- Compare before/after

### Task 5: Audio Segmentation
- Split long audio into chunks
- Detect voice activity
- Save segments with metadata

## 📊 Success Criteria

- ✅ Can visualize and interpret audio features
- ✅ Extract MFCCs for ML pipelines
- ✅ Preprocess audio to consistent format
- ✅ Improve audio quality programmatically
- ✅ Understand sampling rate tradeoffs

## 🔧 Required Libraries

```bash
pip install librosa
pip install soundfile
pip install scipy
pip install matplotlib
pip install noisereduce
```

## 🔗 Next Steps

→ **[05_nlp](../05_nlp/)** for text processing that pairs with audio

**Time Estimate**: 5-7 hours

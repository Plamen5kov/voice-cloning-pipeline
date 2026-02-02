# Data Preparation - Learning Guide

## 🎯 Module Overview

Learn professional data preparation techniques for voice cloning. High-quality, well-organized data is the foundation of successful ML projects.

## 📚 What You'll Learn

- Recording voice samples properly
- Audio preprocessing pipelines
- Dataset organization and structure
- Data labeling and metadata
- Audio-text alignment
- Quality control and validation

## 🎓 Learning Objectives

- [ ] Record clean voice samples
- [ ] Build preprocessing pipeline
- [ ] Organize data in standard format
- [ ] Create metadata files
- [ ] Validate audio-text pairs
- [ ] Handle LibriTTS and custom datasets

## 📝 Key Concepts

### Data Quality Factors
- **Audio Quality**: 24kHz+, low noise, consistent volume
- **Text Quality**: Accurate transcription, proper punctuation
- **Diversity**: Various phonemes, intonations, contexts
- **Quantity**: Minimum 30 minutes for voice cloning

### Standard Dataset Structure
```
dataset/
├── metadata.csv          # Audio filename, text, speaker
├── wavs/
│   ├── sample001.wav
│   ├── sample002.wav
│   └── ...
└── transcripts/
    ├── sample001.txt
    └── ...
```

### Recording Best Practices
- **Environment**: Quiet room, minimal echo
- **Equipment**: Good microphone (USB or XLR)
- **Distance**: 6-12 inches from mic
- **Consistency**: Same setup for all recordings

## 🚀 Exercises & Tasks

### Task 1: Record Voice Samples
- Set up quiet recording environment
- Record 10 short sentences (5-10 seconds each)
- Record 10 longer passages (30-60 seconds each)
- Save as WAV, 24kHz, mono

**Learning Point**: Consistency is key for voice cloning

### Task 2: Build Preprocessing Pipeline (`scripts/download_and_prepare_data.py`)
Create a script to:
- Trim silence from start/end
- Normalize volume to -20dB
- Resample to 24kHz
- Convert to mono
- Validate duration (3-30 seconds)

**Learning Point**: Automated preprocessing ensures consistency

### Task 3: Download LibriTTS
- Download dev-clean subset
- Extract to `data/libritts_sample/`
- Explore structure
- Count samples and speakers

**Learning Point**: Use public datasets to supplement custom data

### Task 4: Create Metadata File
Build CSV with:
- Filename
- Speaker ID
- Transcription text
- Duration
- Quality score

**Learning Point**: Metadata enables filtering and analysis

### Task 5: Audio-Text Validation
Write validation script:
- Check all audio files exist
- Verify transcriptions are non-empty
- Validate audio quality (sampling rate, channels)
- Detect problematic samples
- Generate quality report

## 📊 Success Criteria

- ✅ 20+ clean voice recordings
- ✅ Automated preprocessing pipeline
- ✅ Organized dataset structure
- ✅ Complete metadata file
- ✅ Validation passing 100%

## 🔧 Required Libraries

```bash
pip install librosa
pip install soundfile
pip install pydub
pip install pandas
```

## 📋 Quality Checklist

For each recording:
- [ ] No background noise
- [ ] Consistent volume level
- [ ] Clear pronunciation
- [ ] Natural pacing (not too fast/slow)
- [ ] 24kHz sampling rate
- [ ] Mono channel
- [ ] 3-30 second duration
- [ ] Accurate transcription

## 🔗 Next Steps

→ **[08_model_training_finetuning](../08_model_training_finetuning/)** to train on your data

## 💡 Data Collection Tips

1. **Record in batches**: Maintain consistent voice/mood
2. **Read naturally**: Don't sound robotic
3. **Include variety**: Questions, statements, different emotions
4. **Mark bad takes**: Note which recordings to exclude
5. **Back up immediately**: Don't lose hours of work

## 📖 Additional Resources

- [LibriTTS Dataset](https://www.openslr.org/60/)
- [LJ Speech Dataset](https://keithito.com/LJ-Speech-Dataset/)
- [Audio Recording Best Practices](https://www.audacityteam.org/about/features/recording/)

## ⚠️ Common Issues

**Issue**: Inconsistent background noise
**Solution**: Use same room and time of day for all recordings

**Issue**: Clipping (distortion from too loud)
**Solution**: Keep input levels at -12dB to -6dB peak

**Issue**: Mouth sounds (clicks, pops)
**Solution**: Stay hydrated, use pop filter, edit in post

---

**Time Estimate**: 10-15 hours (including recording time)

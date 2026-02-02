# Model Training & Fine-tuning - Learning Guide

## 🎯 Module Overview

Learn to train and fine-tune TTS models on your voice data. This is where your prepared data becomes a personalized voice synthesis system.

## 📚 What You'll Learn

- TTS model architectures (Tacotron, FastSpeech, XTTS)
- Training pipeline setup
- Hyperparameter tuning
- Monitoring training progress
- Handling overfitting
- Model checkpointing and saving

## 🎓 Learning Objectives

- [ ] Configure training environment
- [ ] Prepare data for TTS training
- [ ] Train Tacotron or XTTS model
- [ ] Monitor loss curves
- [ ] Fine-tune pre-trained model
- [ ] Save and version models

## 📝 Key Concepts

### TTS Model Types
- **Tacotron 2**: Seq2seq with attention (slower, high quality)
- **FastSpeech**: Non-autoregressive (faster, good quality)
- **XTTS**: Multi-lingual, voice cloning (modern, versatile)

### Training Process
1. **Data Loading**: Batch audio-text pairs
2. **Forward Pass**: Model predicts mel-spectrogram
3. **Loss Computation**: Compare prediction to ground truth
4. **Backpropagation**: Update model weights
5. **Validation**: Test on held-out data

### Key Hyperparameters
- **Batch Size**: 16-32 (depends on GPU memory)
- **Learning Rate**: 1e-4 to 1e-3 (with warmup)
- **Epochs**: 100-500 (until convergence)
- **Mel Bins**: 80 (standard)

## 🚀 Exercises & Tasks

### Task 1: Install Training Framework
```bash
pip install TTS
pip install coqui-tts
```

Configure GPU and verify CUDA

### Task 2: Prepare Training Data
- Format dataset in Coqui TTS structure
- Create train/validation split (90/10)
- Generate phoneme alignments
- Validate data loading

### Task 3: Configure Training
- Choose model architecture
- Set hyperparameters
- Configure logging and checkpointing
- Estimate training time

### Task 4: Start Training
- Initialize model or load pre-trained weights
- Train for initial epochs
- Monitor loss curves
- Save checkpoints every N steps

### Task 5: Fine-tune Pre-trained Model
- Load XTTS pre-trained model
- Freeze encoder layers
- Train only speaker embedding
- Compare to training from scratch

### Task 6: Validate and Test
- Generate samples during training
- Listen for quality improvements
- Detect overfitting (train vs val loss)
- Select best checkpoint

## 📊 Success Criteria

- ✅ Training runs without errors
- ✅ Loss decreases consistently
- ✅ Generated audio is intelligible
- ✅ Voice sounds similar to training data
- ✅ Model generalizes to new text

## 🎯 Training Checklist

Pre-training:
- [ ] Data preprocessed and validated
- [ ] GPU memory sufficient for batch size
- [ ] Config file reviewed
- [ ] Baseline model tested

During training:
- [ ] Monitor loss curves (should decrease)
- [ ] Generate samples every 1000 steps
- [ ] Check for overfitting
- [ ] Adjust learning rate if needed

Post-training:
- [ ] Test on diverse text samples
- [ ] Compare checkpoints
- [ ] Document hyperparameters
- [ ] Save best model

## 🔧 Required Setup

```bash
# GPU with 8GB+ VRAM recommended
# CUDA 11.8+
pip install TTS
pip install tensorboard  # For monitoring
```

## 📈 Interpreting Training Metrics

| Metric | Good | Warning |
|--------|------|---------|
| Loss | Steadily decreasing | Plateauing early |
| Val Loss | Tracks train loss | Much higher than train |
| Audio Quality | Improves over time | Remains noisy |
| Inference Speed | <1s for sentence | >5s for sentence |

## 🔗 Next Steps

→ **[09_generative_ai](../09_generative_ai/)** for advanced generative techniques

## 💡 Training Tips

1. **Start small**: Test on 10 samples before full dataset
2. **Use pre-trained models**: Fine-tuning is faster than training from scratch
3. **Monitor early**: Catch issues in first 100 steps
4. **Save often**: Don't lose hours of training to crashes
5. **Listen regularly**: Quality metrics don't tell the full story

## 📖 Additional Resources

- [Coqui TTS Training Guide](https://tts.readthedocs.io/en/latest/tutorial_for_nervous_beginners.html)
- [XTTS Fine-tuning](https://github.com/coqui-ai/TTS/wiki/Fine-Tuning)
- [Tacotron 2 Paper](https://arxiv.org/abs/1712.05884)

## ⚠️ Common Issues

**Issue**: CUDA out of memory
**Solution**: Reduce batch size or use gradient accumulation

**Issue**: Loss not decreasing
**Solution**: Check data quality, lower learning rate, increase warmth

**Issue**: Robotic voice output
**Solution**: Train longer, increase model capacity, improve data quality

---

**Time Estimate**: 15-25 hours (including training time)

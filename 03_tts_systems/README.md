# Text-to-Speech (TTS) Systems Exercises

This folder is for experiments with TTS systems like Coqui TTS and Bark. Store scripts, configs, and audio outputs here.

## Setup

### PyTorch Installation
PyTorch 2.10.0 with CUDA 13.0 is installed globally in pyenv Python 3.11.9:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

**Note**: This machine has an NVIDIA Blackwell GB10 GPU (compute capability 12.1). PyTorch 2.10+ with CUDA 13.0 is required for GPU support.

### Dependencies
```bash
pip install transformers==4.49.0  # Required for XTTS compatibility
```

## Suggested contents:
- TTS demo scripts
- Batch conversion scripts
- Audio output files
- Voice/style comparison results

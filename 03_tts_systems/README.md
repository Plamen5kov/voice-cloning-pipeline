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
pip install torchcodec --index-url https://download.pytorch.org/whl/cu130
sudo apt install -y ffmpeg  # Required by torchcodec
```

## Scripts

### Utility Module
- **tts_utils.py** - Common utility functions used by all scripts:
  - `get_device()` - Auto-detect CUDA/CPU
  - `load_tts_model()` - Load and initialize TTS models
  - `synthesize_speech()` - Unified speech synthesis
  - `ensure_directory()` - Safe directory creation
  - `generate_output_path()` - Path generation with auto-creation
  - `sanitize_filename()` - Safe filename generation
  - `print_progress()` - Progress indicators
  - `print_summary()` - Batch operation summaries

### Demo Scripts
- **tts_basic_demo.py** - Basic TTS demonstration
- **tts_xtts_ana_demo.py** - Demo with Ana Florence speaker
- **tts_xtts_my_voice_demo.py** - Voice cloning demo with your voice sample
- **tts_voice_comparison.py** - Compare 8 different speaker voices
- **tts_batch_convert.py** - Batch convert multiple paragraphs to audio

## Usage Examples

### Single Voice Generation
```bash
python tts_xtts_ana_demo.py
```

### Voice Cloning
```bash
# Requires my_voice_sample.wav in the current directory
python tts_xtts_my_voice_demo.py
```

### Voice Comparison
```bash
python tts_voice_comparison.py
# Output: voice_comparison/ folder with 8 different voices
```

### Batch Conversion
```bash
python tts_batch_convert.py
# Output: batch_output_default/ and batch_output_cloned/ folders
```

## Output Directories
- `voice_comparison/` - Voice comparison samples
- `batch_output_default/` - Batch conversions with default speaker
- `batch_output_cloned/` - Batch conversions with cloned voice
- Individual output files: `output_xtts_demo_*.wav`

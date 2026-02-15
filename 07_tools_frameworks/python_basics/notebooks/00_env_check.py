import sys
print('Python version:', sys.version)

import torch
print('Torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())

import torchaudio
print('Torchaudio version:', torchaudio.__version__)

import librosa
print('Librosa version:', librosa.__version__)

try:
    import bark
    print('Bark installed')
except ImportError:
    print('Bark not installed')

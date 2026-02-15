#!/bin/bash
# Script to convert all .m4a files in data/raw_voice_samples to .wav (mono, 16kHz) in data/processed_voice_samples
#
# Usage: bash scripts/convert_m4a_to_wav.sh
#
# What it does:
# - Finds all .m4a files in data/raw_voice_samples
# - Converts each to .wav format using ffmpeg
# - Output files are mono (1 channel), 24kHz sample rate, 16-bit PCM
# - Saves converted files in data/processed_voice_samples
#
# ffmpeg parameters explained:
#   -i <input>      : Input file (.m4a)
#   -ar 24000       : Set output sample rate to 24,000 Hz (Bark requirement)
#   -ac 1           : Set output to mono (single audio channel)
#   -c:a pcm_s16le  : Use 16-bit signed PCM encoding (uncompressed, widely supported)

RAW_DIR="data/raw_voice_samples"
PROCESSED_DIR="data/processed_voice_samples"

mkdir -p "$PROCESSED_DIR"

for f in "$RAW_DIR"/*.m4a; do
    fname=$(basename "$f" .m4a)
    out="$PROCESSED_DIR/$fname.wav"
    # Convert to mono, 24kHz, 16-bit PCM WAV
    ffmpeg -i "$f" -ar 24000 -ac 1 -c:a pcm_s16le "$out"

    echo "Converted $f to $out"
done

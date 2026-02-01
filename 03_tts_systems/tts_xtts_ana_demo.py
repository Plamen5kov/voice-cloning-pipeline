import torch
from TTS.api import TTS

def main():
    # Try CUDA with PyTorch 2.10 + CUDA 13.0
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # List available models
    print("Available models:")
    print(TTS().list_models())
    # Init the XTTS v2 expressive model
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    # Use the default Ana Florence speaker
    text = "I just want to hear how the voice would sound saying this. Honey, give me the toilet paper!"
    default_speaker = "Ana Florence"
    out_path = "output_xtts_demo_ana.wav"
    print(f"Generating for default speaker: {default_speaker}")
    tts.tts_to_file(
        text=text,
        speaker=default_speaker,
        language="en",
        file_path=out_path
    )
    print(f"Audio saved to {out_path} using speaker: {default_speaker}")

if __name__ == "__main__":
    main()

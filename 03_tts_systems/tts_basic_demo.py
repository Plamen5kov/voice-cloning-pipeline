import torch
from TTS.api import TTS

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # List available models
    print("Available models:")
    print(TTS().list_models())
    # Init a basic English TTS model
    tts = TTS(model_name="tts_models/en/ljspeech/glow-tts").to(device)
    # Synthesize speech
    tts.tts_to_file(text="Hello, this is a basic Coqui TTS demo.", file_path="output_basic_demo.wav")
    print("Audio saved to output_basic_demo.wav")

if __name__ == "__main__":
    main()

import torch
from TTS.api import TTS

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    # List available models
    print("Available models:")
    print(TTS().list_models())
    # Init the XTTS v2 expressive model
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2").to(device)
    # Use the user's own sample for voice cloning
    text = "I just want to hear how the voice would sound saying this. Honey, give me the toilet paper!"
    speaker_wav = "my_voice_sample.wav"
    out_path = "output_xtts_demo_cloned.wav"
    print(f"Generating for your voice sample: {speaker_wav}")
    tts.tts_to_file(
        text=text,
        speaker_wav=speaker_wav,
        language="en",
        file_path=out_path
    )
    print(f"Audio saved to {out_path} using your voice sample.")

if __name__ == "__main__":
    main()

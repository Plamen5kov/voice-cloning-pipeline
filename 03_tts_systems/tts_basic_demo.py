from tts_utils import get_device, load_tts_model

def main():
    # Get device and load model
    device = get_device()
    tts = load_tts_model(model_name="tts_models/en/ljspeech/glow-tts", device=device)
    
    # Synthesize speech (basic model doesn't support speaker parameter)
    text = "Hello, this is a basic Coqui TTS demo."
    output_file = "output_basic_demo.wav"
    
    print(f"Generating speech...")
    tts.tts_to_file(text=text, file_path=output_file)
    print(f"✓ Audio saved to {output_file}")

if __name__ == "__main__":
    main()


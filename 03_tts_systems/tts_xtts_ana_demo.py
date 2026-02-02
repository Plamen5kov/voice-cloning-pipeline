from tts_utils import get_device, load_tts_model, synthesize_speech

def main():
    # Get device and load model
    device = get_device()
    tts = load_tts_model(device=device)
    
    # Configuration
    text = "I just want to hear how the voice would sound saying this. Honey, give me the toilet paper!"
    speaker = "Ana Florence"
    output_file = "output_xtts_demo_ana.wav"
    
    # Generate speech
    print(f"Generating speech with speaker: {speaker}")
    success = synthesize_speech(tts, text, output_file, speaker=speaker)
    
    if success:
        print(f"✓ Audio saved to {output_file}")

if __name__ == "__main__":
    main()

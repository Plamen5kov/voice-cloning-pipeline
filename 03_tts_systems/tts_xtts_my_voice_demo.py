from tts_utils import get_device, load_tts_model, synthesize_speech

def main():
    # Get device and load model
    device = get_device()
    tts = load_tts_model(device=device)
    
    # Configuration
    text = "I just want to hear how the voice would sound saying this. Honey, give me the toilet paper!"
    speaker_wav = "my_voice_sample.wav"
    output_file = "output_xtts_demo_cloned.wav"
    
    # Generate speech with voice cloning
    print(f"Generating speech with your voice sample: {speaker_wav}")
    success = synthesize_speech(tts, text, output_file, speaker_wav=speaker_wav)
    
    if success:
        print(f"✓ Audio saved to {output_file} using your voice sample")

if __name__ == "__main__":
    main()

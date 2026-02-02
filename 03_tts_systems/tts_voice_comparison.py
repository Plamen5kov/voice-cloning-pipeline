from tts_utils import (
    get_device, 
    load_tts_model, 
    synthesize_speech, 
    generate_output_path,
    sanitize_filename,
    print_progress
)

def main():
    # Get device and load model
    device = get_device()
    tts = load_tts_model(device=device)
    
    # Configuration
    text = "Hello! This is a demonstration of different voice styles. Each speaker has unique characteristics and tone."
    output_dir = "voice_comparison"
    
    # List of available speakers to compare
    speakers = [
        "Claribel Dervla",
        "Daisy Studious", 
        "Gracie Wise",
        "Tammie Ema",
        "Ana Florence",
        "Annmarie Nele",
        "Asya Anara",
        "Brenda Stern"
    ]
    
    print(f"\nGenerating speech with {len(speakers)} different speakers...")
    print(f"Text: '{text}'\n")
    
    for i, speaker in enumerate(speakers, 1):
        filename = f"speaker_{sanitize_filename(speaker)}.wav"
        output_file = generate_output_path(output_dir, filename)
        
        print_progress(i, len(speakers), f"Generating: {speaker}...")
        success = synthesize_speech(tts, text, output_file, speaker=speaker)
        
        if success:
            print(f"  ✓ Saved to {output_file}")
        else:
            print(f"  ✗ Failed")
    
    print(f"\n✅ Voice comparison complete! Check the '{output_dir}' folder.")
    print(f"Listen to each file to compare the different voice styles.")

if __name__ == "__main__":
    main()

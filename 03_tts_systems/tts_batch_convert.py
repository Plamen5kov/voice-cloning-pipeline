from tts_utils import (
    get_device,
    load_tts_model,
    synthesize_speech,
    generate_output_path,
    print_progress,
    print_summary
)
import os

def batch_convert_paragraphs(paragraphs, output_dir="batch_output", speaker="Ana Florence", speaker_wav=None, language="en"):
    """
    Convert multiple paragraphs to separate audio files.
    
    Args:
        paragraphs: List of text strings to convert
        output_dir: Directory to save output files
        speaker: Speaker name for predefined speakers
        speaker_wav: Path to speaker WAV file for voice cloning
        language: Language code (en, es, fr, de, etc.)
    """
    # Get device and load model
    device = get_device()
    tts = load_tts_model(device=device)
    
    print(f"\nBatch converting {len(paragraphs)} paragraphs...")
    print(f"Speaker: {speaker_wav if speaker_wav else speaker}")
    print(f"Output directory: {output_dir}\n")
    
    successful = 0
    failed = 0
    
    for i, text in enumerate(paragraphs, 1):
        filename = f"paragraph_{i:03d}.wav"
        output_file = generate_output_path(output_dir, filename)
        
        print_progress(i, len(paragraphs), f"Converting paragraph {i}...")
        print(f"  Text preview: {text[:80]}...")
        
        success = synthesize_speech(
            tts, text, output_file, 
            speaker=speaker,
            speaker_wav=speaker_wav,
            language=language
        )
        
        if success:
            print(f"  ✓ Saved to {output_file}\n")
            successful += 1
        else:
            print(f"  ✗ Failed\n")
            failed += 1
    
    print_summary(successful, failed, output_dir)

def main():
    # Example: Convert paragraphs from a book excerpt
    paragraphs = [
        "In the beginning, the universe was created. This has made a lot of people very angry and been widely regarded as a bad move.",
        
        "The story so far: In the beginning the Universe was created. This has made a lot of people very angry and been widely regarded as a bad move.",
        
        "Many were increasingly of the opinion that they'd all made a big mistake in coming down from the trees in the first place. And some said that even the trees had been a bad move, and that no one should ever have left the oceans.",
        
        "And then, one Thursday, nearly two thousand years after one man had been nailed to a tree for saying how great it would be to be nice to people for a change, a girl sitting on her own in a small cafe in Rickmansworth suddenly realized what it was that had been going wrong all this time.",
        
        "The answer to this is very simple. It was a joke. It had to be a number, an ordinary, smallish number, and I chose that one. Binary representations, base thirteen, Tibetan monks are all complete nonsense. I sat at my desk, stared into the garden and thought '42 will do'. I typed it out. End of story."
    ]
    
    # Convert using default speaker
    batch_convert_paragraphs(
        paragraphs=paragraphs,
        output_dir="batch_output_default",
        speaker="Ana Florence"
    )
    
    # Example: Use your own voice (if my_voice_sample.wav exists)
    if os.path.exists("my_voice_sample.wav"):
        print("\n" + "="*60)
        print("Now converting with YOUR cloned voice...")
        print("="*60 + "\n")
        
        batch_convert_paragraphs(
            paragraphs=paragraphs[:2],  # Just first 2 for demo
            output_dir="batch_output_cloned",
            speaker_wav="my_voice_sample.wav"
        )

if __name__ == "__main__":
    main()

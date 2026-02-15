import os
import soundfile as sf

def convert_folder_to_wav(src_folder, dst_folder):
    os.makedirs(dst_folder, exist_ok=True)
    for fname in os.listdir(src_folder):
        if fname.endswith('.flac'):
            flac_path = os.path.join(src_folder, fname)
            wav_path = os.path.join(dst_folder, fname.replace('.flac', '.wav'))
            data, samplerate = sf.read(flac_path)
            sf.write(wav_path, data, samplerate)
            print(f"Converted {fname} to {wav_path}")

convert_folder_to_wav('male_samples', 'male_samples_wav')
convert_folder_to_wav('female_samples', 'female_samples_wav')

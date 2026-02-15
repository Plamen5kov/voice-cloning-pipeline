import os
import shutil
import json

# Path to the dev-clean directory
DEV_CLEAN_DIR = "data/LibriSpeech/dev-clean"
# Path to the speakers metadata file (update if needed)
SPEAKERS_FILE = "data/LibriSpeech/SPEAKERS.TXT"

# Output directories
MALE_DIR = "male_samples"
FEMALE_DIR = "female_samples"

os.makedirs(MALE_DIR, exist_ok=True)
os.makedirs(FEMALE_DIR, exist_ok=True)

# Parse speakers.txt to build gender mapping
SPEAKER_GENDERS = {}
with open(SPEAKERS_FILE, "r") as f:
    for line in f:
        if line.strip() and not line.startswith("#"):
            parts = line.strip().split("|")
            if len(parts) >= 3:
                speaker_id = parts[0].strip()
                gender = parts[1].strip()
                SPEAKER_GENDERS[speaker_id] = gender

male_ids = []
female_ids = []

# Traverse dev-clean directory
for speaker_id in os.listdir(DEV_CLEAN_DIR):
    speaker_path = os.path.join(DEV_CLEAN_DIR, speaker_id)
    if not os.path.isdir(speaker_path):
        continue
    gender = SPEAKER_GENDERS.get(speaker_id)
    if gender == "M":
        male_ids.append(speaker_id)
    elif gender == "F":
        female_ids.append(speaker_id)
    else:
        continue  # skip unknown gender
    # Copy all audio files for this speaker
    for chapter_id in os.listdir(speaker_path):
        chapter_path = os.path.join(speaker_path, chapter_id)
        if not os.path.isdir(chapter_path):
            continue
        for fname in os.listdir(chapter_path):
            if fname.endswith(".flac"):
                src = os.path.join(chapter_path, fname)
                if gender == "M":
                    shutil.copy(src, os.path.join(MALE_DIR, f"{speaker_id}_{chapter_id}_{fname}"))
                elif gender == "F":
                    shutil.copy(src, os.path.join(FEMALE_DIR, f"{speaker_id}_{chapter_id}_{fname}"))

# Save the resulting speaker IDs for traceability
with open("male_speaker_ids.json", "w") as f:
    json.dump(male_ids, f, indent=2)
with open("female_speaker_ids.json", "w") as f:
    json.dump(female_ids, f, indent=2)

print(f"Done. Male speakers: {len(male_ids)}, Female speakers: {len(female_ids)}.")

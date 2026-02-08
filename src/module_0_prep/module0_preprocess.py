import os
import glob
import librosa
import soundfile as sf
import numpy as np

# Configuration
INPUT_DIR = r"C:\Users\Khushi\Downloads\Track_1_SD_DevData_1\Track_1_SD_DevData_1\Hindi\data\wav"
OUTPUT_DIR = os.path.join("data", "processed_audio")
TARGET_SR = 16000
TARGET_RMS = 0.1

def preprocess_audio(input_dir, output_dir, target_sr=16000, target_rms=0.1):
    """
    Module 0: Preprocesses raw audio files by resampling, converting to mono,
    trimming silence, and normalizing RMS.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Check if input directory exists
    if not os.path.exists(input_dir):
        print(f"Error: Input directory does not exist: {input_dir}")
        return

    # Supported audio extensions (mainly wav as per instructions)
    extensions = ['*.wav']
    files_to_process = []
    for ext in extensions:
        files_to_process.extend(glob.glob(os.path.join(input_dir, ext)))

    if not files_to_process:
        print(f"No audio files found in {input_dir}")
        return

    print(f"Found {len(files_to_process)} files to process.")

    for file_path in files_to_process:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(output_dir, os.path.splitext(file_name)[0] + '.wav')
        
        print(f"Processing: {file_name}...")
        
        try:
            # 1. Load audio, 2. Resample to 16 kHz, 3. Convert to mono
            y, sr = librosa.load(file_path, sr=target_sr, mono=True)
            
            # 4. Trim boundary silence only (5. Internal pauses preserved by default)
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # 6. Apply conversation-level RMS normalization
            current_rms = np.sqrt(np.mean(y_trimmed**2))
            if current_rms > 0:
                y_normalized = y_trimmed * (target_rms / current_rms)
            else:
                print(f"Warning: File {file_name} is silent or has 0 RMS. Skipping normalization.")
                y_normalized = y_trimmed

            # 7. Save as 16-bit PCM WAV
            sf.write(output_path, y_normalized, target_sr, subtype='PCM_16')
            
            print(f"Successfully processed and saved to: {output_path}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

if __name__ == "__main__":
    preprocess_audio(INPUT_DIR, OUTPUT_DIR, TARGET_SR, TARGET_RMS)

import os
import glob
import librosa
import soundfile as sf
import numpy as np

def preprocess_audio(input_dir, output_dir, target_sr=16000, target_rms=0.1):
    """
    Module 0: Preprocesses raw audio files by resampling, converting to mono,
    trimming silence, and normalizing RMS.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Supported audio extensions
    extensions = ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']
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
            # librosa.load handles resampling and mono conversion automatically if specified
            y, sr = librosa.load(file_path, sr=target_sr, mono=True)
            
            # 4. Trim boundary silence only
            # top_db=20 is a common threshold for "silence"
            y_trimmed, _ = librosa.effects.trim(y, top_db=20)
            
            # 5. Apply conversation-level RMS normalization
            current_rms = np.sqrt(np.mean(y_trimmed**2))
            if current_rms > 0:
                y_normalized = y_trimmed * (target_rms / current_rms)
            else:
                print(f"Warning: File {file_name} is silent or has 0 RMS. Skipping normalization.")
                y_normalized = y_trimmed

            # 6. Save as 16-bit PCM WAV
            # soundfile.write with subtype='PCM_16' ensures 16-bit output
            sf.write(output_path, y_normalized, target_sr, subtype='PCM_16')
            
            print(f"Successfully processed and saved to: {output_path}")

        except Exception as e:
            print(f"Error processing {file_name}: {e}")
            continue

if __name__ == "__main__":
    RAW_DIR = os.path.join('data', 'raw_audio')
    PROCESSED_DIR = os.path.join('data', 'processed_audio')
    
    preprocess_audio(RAW_DIR, PROCESSED_DIR)

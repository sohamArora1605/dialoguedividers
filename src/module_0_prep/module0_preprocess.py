import os
import glob
import logging
import librosa
import numpy as np
from tqdm import tqdm
from rich.logging import RichHandler
from scipy.io import wavfile

# ---------------------------
# Logging configuration
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)]
)
log = logging.getLogger("rich")

# ---------------------------
# Configuration
# ---------------------------
INPUT_DIR = r"C:\Samsung\dialoguedividers\Track_1_SD_DevData_1\Hindi\data\wav"
OUTPUT_DIR = r"C:\Samsung\dialoguedividers\src\module_0_prep\data\processed_audio"

TARGET_SR = 16000
TARGET_RMS = 0.1

# ---------------------------
# Audio Processing Function
# ---------------------------
def process_file(file_path: str):
    """
    Module 0 Audio Preprocessing:
    1. Load audio
    2. Resample to 16 kHz
    3. Convert to mono
    5. Preserve all internal pauses
    6. Apply conversation-level RMS normalization (~0.1)
    7. Save as 16-bit PCM WAV
    """
    try:
        # 1–3. Load audio (resample + mono)
        audio, _ = librosa.load(
            file_path,
            sr=TARGET_SR,
            mono=True
        )

        # 6. RMS normalization (numerically stable)
        rms = np.sqrt(np.mean(audio ** 2) + 1e-9)

        if rms > 0:
            scale = TARGET_RMS / rms
            audio_norm = audio * scale
        else:
            audio_norm = audio

        # Prevent clipping
        audio_norm = np.clip(audio_norm, -1.0, 1.0)

        # Optional sanity check (does NOT modify audio)
        final_rms = np.sqrt(np.mean(audio_norm ** 2) + 1e-9)
        if not (0.08 <= final_rms <= 0.12):
            log.warning(
                f"{os.path.basename(file_path)} RMS off-target: {final_rms:.3f}"
            )

        # 7. Save as 16-bit PCM WAV using SciPy (Windows-safe)
        output_path = os.path.join(
            OUTPUT_DIR,
            os.path.basename(file_path)
        )

        audio_int16 = (audio_norm * 32767).astype(np.int16)

        wavfile.write(
            output_path,
            TARGET_SR,
            audio_int16
        )

    except Exception as e:
        log.error(f"Skipping file {os.path.basename(file_path)} due to error: {e}")

# ---------------------------
# Main execution
# ---------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    wav_files = glob.glob(os.path.join(INPUT_DIR, "*.wav"))

    if not wav_files:
        log.warning(f"No WAV files found in {INPUT_DIR}")
        return

    log.info(f"Found {len(wav_files)} WAV files. Processing...")

    for wav_path in tqdm(wav_files, desc="Processing Audio"):
        process_file(wav_path)

    log.info("Module 0 processing complete.")

# ---------------------------
if __name__ == "__main__":
    main()

import os
import glob
import librosa
import numpy as np

# ---------------------------
# Configuration
# ---------------------------
AUDIO_DIR = r"C:\Samsung\dialoguedividers\src\module_0_prep\data\processed_audio"
TARGET_SR = 16000

# ---------------------------
# Verification Function
# ---------------------------
def verify_audio(file_path):
    audio, sr = librosa.load(file_path, sr=None, mono=False)

    # Mono / Stereo check
    if audio.ndim == 1:
        channels = 1
    else:
        channels = audio.shape[0]

    # Duration
    duration = len(audio) / sr

    # RMS Loudness
    rms = np.sqrt(np.mean(audio ** 2) + 1e-9)

    # Clipping check
    clipped = np.any(np.abs(audio) >= 1.0)

    print(f"\nFile: {os.path.basename(file_path)}")
    print(f"Sample Rate : {sr} Hz")
    print(f"Channels    : {channels}")
    print(f"Duration    : {duration:.2f} sec")
    print(f"RMS Loudness: {rms:.3f}")
    print(f"Clipping    : {'YES ❌' if clipped else 'NO ✅'}")

# ---------------------------
# Main
# ---------------------------
def main():
    wav_files = glob.glob(os.path.join(AUDIO_DIR, "*.wav"))

    if not wav_files:
        print("No WAV files found.")
        return

    print(f"Verifying {len(wav_files)} audio files...\n")

    for wav in wav_files:
        verify_audio(wav)

    print("\nVerification complete.")

# ---------------------------
if __name__ == "__main__":
    main()

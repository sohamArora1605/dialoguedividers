import librosa
import numpy as np
import os

def verify_files(directory):
    if not os.path.exists(directory):
        print(f"Directory {directory} does not exist.")
        return

    files = [f for f in os.listdir(directory) if f.endswith('.wav')]
    if not files:
        print(f"No processed files found in {directory}.")
        return

    print(f"Verifying {len(files)} files in {directory}...")
    
    for f in files:
        path = os.path.join(directory, f)
        try:
            y, sr = librosa.load(path, sr=None)
            
            is_16k = (sr == 16000)
            is_mono = (y.ndim == 1)
            rms = np.sqrt(np.mean(y**2))
            is_approx_01 = (0.09 < rms < 0.11)
            
            status = "OK" if (is_16k and is_mono and is_approx_01) else "FAIL"
            
            print(f"{f}: {status}")
            if status == "FAIL":
                print(f"  Sample Rate: {sr} {'[OK]' if is_16k else '[FAIL (Expected 16000)]'}")
                print(f"  Channels: {'Mono' if is_mono else f'Multi ({y.ndim})'} {'[OK]' if is_mono else '[FAIL]'}")
                print(f"  RMS: {rms:.4f} {'[OK]' if is_approx_01 else '[FAIL (Expected ~0.1)]'}")
        except Exception as e:
            print(f"Error verifying {f}: {e}")
        
if __name__ == "__main__":
    verify_files(os.path.join('data', 'processed_audio'))

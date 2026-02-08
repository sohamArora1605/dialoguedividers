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

    print(f"{'File Name':<30} | {'SR':<6} | {'Ch':<2} | {'RMS':<6} | {'Duration':<8} | {'Status'}")
    print("-" * 75)
    
    all_passed = True
    for f in files:
        path = os.path.join(directory, f)
        try:
            # sr=None preserves original sampling rate of the file
            y, sr = librosa.load(path, sr=None)
            
            duration = librosa.get_duration(y=y, sr=sr)
            channels = 1 if y.ndim == 1 else y.shape[0] if y.ndim == 2 else "Error"
            rms = np.sqrt(np.mean(y**2))
            
            # Validation criteria
            is_16k = (sr == 16000)
            is_mono = (channels == 1)
            is_rms_valid = (0.08 <= rms <= 0.12)
            is_duration_valid = (duration > 0.1)  # Extremely short check
            
            file_passed = is_16k and is_mono and is_rms_valid and is_duration_valid
            if not file_passed:
                all_passed = False
                
            status_str = "PASS" if file_passed else "FAIL"
            
            print(f"{f:<30} | {sr:<6} | {channels:<2} | {rms:.4f} | {duration:>7.2f}s | {status_str}")
            
            if not file_passed:
                reasons = []
                if not is_16k: reasons.append(f"SR={sr}")
                if not is_mono: reasons.append(f"Ch={channels}")
                if not is_rms_valid: reasons.append(f"RMS={rms:.4f}")
                if not is_duration_valid: reasons.append(f"Dur={duration:.2f}s")
                print(f"  -> FAIL Reasons: {', '.join(reasons)}")
                
        except Exception as e:
            print(f"{f:<30} | ERROR: {e}")
            all_passed = False

    print("-" * 75)
    if all_passed:
        print("\nMODULE 0 STATUS: PASS")
    else:
        print("\nMODULE 0 STATUS: FAIL")
        
if __name__ == "__main__":
    verify_files(os.path.join('data', 'processed_audio'))

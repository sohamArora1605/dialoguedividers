# Module 0: Audio Standardization for Speaker Diarization

## Overview
Module 0 is the **Audio Preprocessing** stage of the pipeline. Its purpose is to standardize raw input audio into a consistent format suitable for downstream tasks (Speaker Diarization), ensuring strict sample-level precision and label preservation.

**Key Features:**
*   **Strict Standardization:** Resamples audio to 16 kHz, converts to Mono, and ensures 16-bit PCM format.
*   **No ML Inference:** Pure signal processing; no VAD, speaker embeddings, or neural networks are used.
*   **Label Preservation:** Trims *only* leading/trailing silence. Internal pauses are preserved 100% to maintain alignment with ground-truth RTTM labels.
*   **Normalization:** Applies conversation-level RMS normalization (Target RMS: 0.1) for consistent volume levels.
*   **Robustness:** Handles corrupted files and prevents clipping artifacts.

## Requirements
*   **Python:** 3.10+
*   **Dependencies:** Listed in [`requirements.txt`](./requirements.txt)
    *   `librosa` (Loading, resampling, trimming)
    *   `soundfile` / `scipy` (Writing 16-bit PCM WAV)
    *   `numpy` (RMS calculation, normalization)
    *   `tqdm` (Progress tracking)
    *   `rich` (Logging)

## Installation

1.  **Navigate to the project root.**
2.  **Install dependencies:**
    ```bash
    pip install -r src/module_0_prep/requirements.txt
    ```

## Usage

### 1. Run Preprocessing
Execute the main script to process all `.wav` files from the input directory.

```bash
python src/module_0_prep/module0_preprocess.py
```

*   **Input Directory:** `Track_1_SD_DevData_1/Hindi/data/wav`
*   **Output Directory:** `src/module_0_prep/data/processed_audio`

### 2. Verify Output
Run the verification script to check the properties of the processed audio files (Sample Rate, Channels, Bit Depth, Clipping, RMS).

```bash
python src/module_0_prep/verify_audio.py
```

## Output Specifications
All properties of the processed audio are strictly enforced:

| Property | Value |
| :--- | :--- |
| **Format** | WAV (RIFF) |
| **Codec** | PCM Signed 16-bit (`PCM_16`) |
| **Sample Rate** | 16,000 Hz |
| **Channels** | Mono (1 channel) |
| **Normalization** | RMS ~0.1 (Conversation-level) |
| **Silence Handling** | Trim start/end ONLY; Internal silence preserved |

## Directory Structure
```
src/module_0_prep/
├── data/
│   └── processed_audio/   # Generated output files (Git-ignored)
├── module0_preprocess.py  # Main processing script
├── verify_audio.py        # Verification utility
├── requirements.txt       # Module-specific dependencies
└── README.md              # This file
```

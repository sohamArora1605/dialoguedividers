# Module 1 – Speech Activity Detection (SAD)

This module implements **Speech Activity Detection (SAD)** for the Displace2026
language-agnostic speaker diarization pipeline.

The goal of this module is to detect **speech vs non-speech regions** in audio
and output precise speech segment timestamps for downstream processing.

---

## Overview

- **Model**: `pyannote/voice-activity-detection`
- **Framework**: PyTorch + PyAnnote
- **Device**: CPU or CUDA (recommended)
- **Output format**: JSON timestamps

This module is designed to be:
- Standalone (can be run independently)
- Importable by other modules
- Reproducible across machines

---

## Directory Structure

```
module_1_sad/
├── sad.py        # SAD implementation
├── __init__.py   # Module export
└── README.md     # This file
```

---

## Input

Directory containing `.wav` audio files.

Default path (relative to repo root):

```
Track_1_SD_DevData_1/Hindi/data/wav/
```

Each file must be:
- Mono or stereo WAV
- Readable by torchaudio

---

## Output

JSON files containing detected speech segments.

Default output directory:

```
outputs/sad/
```

Example output:

```json
[
  { "start": 0.32, "end": 1.84 },
  { "start": 2.10, "end": 4.67 }
]
```

Each entry represents a detected speech region in seconds.

---

## Environment Setup

### Python Version
- Python 3.12

### Install Dependencies (from repo root)

```
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

Important:
- Do NOT install torch manually
- Always use the command above to ensure CUDA compatibility

---

## Hugging Face Authentication (Required)

PyAnnote models are gated and require explicit access approval.

### Step 1: Request access to required models

Visit the following pages after logging in to Hugging Face:

1. https://huggingface.co/pyannote/voice-activity-detection
2. https://huggingface.co/pyannote/segmentation

On each page:
- Click “Request access”
- Answer the questionnaire
- Accept the license terms

---

### Step 2: Login from the command line

Run once:

```
huggingface-cli login
```

Alternatively, set the token manually:

```
export HUGGINGFACE_TOKEN=hf_xxxxx
```

Tokens must never be committed to the repository.

---

## Running the SAD Module

From the repository root:

```
python src/module_1_sad/sad.py
```

The script will:
1. Detect available device (CPU / GPU)
2. Load the pretrained SAD model
3. Process all WAV files in the input directory
4. Write JSON outputs to outputs/sad/

---

## Importing from Other Modules

This module exposes a single public function:

```
from src.module_1_sad import run_sad
```

This allows seamless integration with downstream modules such as segmentation
and speaker embedding extraction.

---

## Notes & Constraints

- Virtual environments must NOT be committed
- Hugging Face tokens must NOT be committed
- Output files are generated at runtime
- Paths are resolved relative to the repository root

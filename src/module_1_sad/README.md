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


```text
module_1_sad/
├── sad.py        # SAD implementation
├── __init__.py   # Module export
└── README.md     # This file
Input

Directory containing .wav audio files.

Default path (relative to repo root):

Track_1_SD_DevData_1/Hindi/data/wav/

Each file must be:

Mono or stereo WAV

Readable by torchaudio

Output

JSON files containing detected speech segments.

Default output directory:

outputs/sad/

Example output:

[
  { "start": 0.32, "end": 1.84 },
  { "start": 2.10, "end": 4.67 }
]

Each entry represents a detected speech region in seconds.

Environment Setup
Python Version

Python 3.12

Install Dependencies (from repo root)
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121

⚠️ Important

Do NOT install torch manually

Always use the command above to ensure CUDA compatibility

Hugging Face Authentication (Required)

PyAnnote models are gated and require explicit access approval.

Step 1: Request access to required models

Log in to Hugging Face and visit the following pages:

https://huggingface.co/pyannote/voice-activity-detection

https://huggingface.co/pyannote/segmentation

On each page:

Click “Request access”

Answer the short questionnaire

Accept the license terms

Access is usually granted quickly.

Step 2: Login from the command line

After access is granted, run once:

huggingface-cli login

Alternatively, you may set the token manually:

export HUGGINGFACE_TOKEN=hf_xxxxx

⚠️ Tokens must never be committed to the repository.

Running the SAD Module

From the repository root:

python src/module_1_sad/sad.py

The script will:

Detect available device (CPU / GPU)

Load the pretrained SAD model

Process all WAV files in the input directory

Write JSON outputs to outputs/sad/

Importing from Other Modules

This module exposes a single public function:

from src.module_1_sad import run_sad

This allows seamless integration with downstream modules such as:

Segmentation

Speaker embedding extraction

Notes & Constraints

Virtual environments must NOT be committed

Hugging Face tokens must NOT be committed

Output files are generated at runtime

Path resolution is relative to the repository root
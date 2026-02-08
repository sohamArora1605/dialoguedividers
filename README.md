# Displace2026: Modular Speaker Diarization Pipeline

This repository contains the implementation of a comprehensive speaker diarization pipeline, organized into modular components for research and development.

## Pipeline Overview

The pipeline is divided into 9 core modules, each handling a specific stage of the diarization process:

- **Module 0: Preparation** - Data ingestion and preprocessing.
- **Module 1: SAD (Speech Activity Detection)** - Identifying speech vs. non-speech segments.
- **Module 2: Segmentation** - Breaking speech into smaller chunks for processing.
- **Module 3: Speaker Embeddings** - Extracting d-vectors/x-vectors for speaker identification.
- **Module 4: Similarity Scoring** - Calculating distance/similarity between embedding segments.
- **Module 5: Clustering** - Grouping segments by speaker identity (AHC, Spectral, etc.).
- **Module 6: Resegmentation** - Refining boundaries and speaker assignments.
- **Module 7: Ensemble/Fusion** - Combining multiple system outputs for robustness.
- **Module 8: Output Scoring** - Evaluation and final formatting (DER, JER).

## Repository Structure

```text
.
├── src/                # Implementation of modules 0-8
├── scripts/            # Automation and utility scripts
├── configs/            # Module-specific configuration files (yaml/json)
├── notebooks/          # Exploratory analysis and prototyping
├── Documentation/      # Detailed documentation for each module
├── Track_1_SD_DevData_1/ # Dataset for Track 1 (Diarization)
├── requirements.txt    # Project dependencies
└── README.md           # This file
```

## Getting Started

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Pipeline**:
   (Instructions to follow as development progresses)

## Acknowledgments
Based on the Displace Challenge 2026 requirements and dataset.

# MODULE 1 — SPEECH ACTIVITY DETECTION (SUMMARY)

## What We Do
Identify where speech exists in time (speech vs non-speech) without determining who is speaking

## Input and Output
- **Input**: Standardized audio from Module 0 (16 kHz, mono, normalized)
- **Output**: Time-stamped speech regions with overlaps preserved, no speaker labels

---

## Processing Pipeline

```
Standardized Audio
        |
        v
Frame Audio into Windows
(25ms frames, 10ms hop)
        |
        v
Extract Frame Features
(Energy, Zero-Crossing Rate, Spectral features)
        |
        v
Apply Classification
(PyAnnote Neural SAD)
        |
        v
Temporal Smoothing
(Fill gaps, remove bursts, extend boundaries)
        |
        v
Speech Regions Output
```

---

## Core Responsibility

Module 1 answers only one question: "Is there speech here or not?"

Module 1 does NOT answer:
- Who is speaking
- How many speakers exist
- Where speaker boundaries are
- What speaker labels should be

---

## Processing Steps

### 1. Frame Audio
**What**: Divide audio into overlapping windows  
**How**:
- Frame size: 25ms (400 samples at 16 kHz)
- Hop size: 10ms (160 samples)
- 60% overlap between frames

**Why**: Speech characteristics change slowly, small overlapping windows capture local behavior without missing transitions  
*Example: 10 seconds of audio produces approximately 998 frames*

---

### 2. Extract Features
**What**: Calculate acoustic features for each frame  
**How**:
- Energy: Sum of squared amplitudes divided by frame length
- Zero-Crossing Rate (ZCR): Frequency of waveform sign changes
- Optional: Spectral features, MFCCs

**Why**: Energy distinguishes speech from silence, ZCR distinguishes speech from noise  
*Example: Speech has energy around 0.05 and ZCR around 0.2, while silence has energy near 0.0001*

---

### 3. Classify Frames
**What**: Decide speech or non-speech for each frame  
**How**:
- Recommended: PyAnnote Neural SAD (CNN + LSTM)
- Alternative: Threshold-based (if energy > 0.01, mark as speech)
- Mode: High-recall (low threshold to avoid missing speech)

**Why**: Neural SAD handles noisy medical environments better than simple thresholds  
*Example: PyAnnote achieves 93-97% accuracy vs 80-85% for threshold-based methods*

---

### 4. Temporal Smoothing
**What**: Clean up frame-level decisions for temporal continuity  
**How**:
- Fill short gaps (under 300ms)
- Remove short bursts (under 150ms, except backchannels)
- Extend boundaries (±300ms padding)
- Merge overlapping regions

**Why**: Real speech doesn't flicker on/off every 10ms, smoothing produces stable continuous regions  
*Example: Raw decisions [1 1 1 0 1 1] become [1 1 1 1 1 1] after filling 10ms gap*

---

## Key Design Principles

### High-Recall Mode (Critical)
**What**: Prioritize detecting all speech over avoiding false positives  
**How**: Set low threshold, preserve short segments, add boundary padding  
**Why**: Missing speech is irrecoverable, false positives can be corrected later  
*Example: Target 95%+ recall even if precision drops to 85%*

---

### Overlap Preservation
**What**: Do not suppress or resolve overlapping speech  
**How**: Mark entire overlapping region as speech without attempting separation  
**Why**: PixIT and powerset models need intact overlaps for joint processing  
*Example: Speakers at [10-13.5s] and [12-14.5s] produce single region [10-14.5s]*

---

### Conservative Filtering
**What**: Only remove obvious non-speech  
**How**: Low threshold, minimal smoothing, generous padding  
**Why**: SAD is a compute filter, not a diarization decision-maker  
*Example: Remove 5-second silence gaps but preserve 200ms pauses within speech*

---

## Relationship with PixIT and Powerset

**SAD Role**: Computational gatekeeper that removes obvious non-speech before expensive PixIT processing

**Why SAD Exists**:
- PixIT is computationally expensive
- Running PixIT on silence wastes resources
- SAD cheaply filters 40-60% of non-speech
- PixIT handles all speaker-level decisions

**What SAD Must NOT Do**:
- Speaker change detection
- Overlap resolution
- Speaker segmentation
- Any speaker-level decisions

---

## Recommended Approach for DISPLACE-2026

**Use PyAnnote Neural SAD**

**Reasoning**:
- Better handles far-field medical recordings
- Robust to medical environment noise
- Trained on diverse conversational data
- Used by 2024 DISPLACE winners

**Resource Requirements**:
- CPU: approximately 0.5x real-time
- GPU: approximately 0.05x real-time
- RAM: approximately 500 MB
- Model size: approximately 50 MB


---


## Complete Processing Pipeline (Detailed)

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Standardized Audio (16 kHz, mono, RMS=0.1, 178s)       │
│ Source: Module 0 output                                        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: FRAME AUDIO                          │
│                                                                 │
│  Frame size: 25ms (400 samples)                                 │
│  Hop size: 10ms (160 samples)                                   │
│  Overlap: 60%                                                   │
│  Number of frames: ~17,800 for 178s audio                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: EXTRACT FEATURES (per frame)               │
│                                                                 │
│  Energy: Sum of squared amplitudes                              │
│  ZCR: Zero-crossing rate                                        │
│  Optional: Spectral features, MFCCs                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: CLASSIFY (per frame)                       │
│                                                                 │
│  Method: PyAnnote Neural SAD (recommended)                      │
│  Alternative: Threshold-based (energy > 0.01)                   │
│  Output: Binary decisions (0 or 1) per frame                    │
│  Mode: High-recall (threshold set low)                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: TEMPORAL SMOOTHING                         │
│                                                                 │
│  4a. Fill short gaps (< 300ms)                                  │
│  4b. Remove short bursts (< 150ms, except backchannels)         │
│  4c. Extend boundaries (±300ms padding)                         │
│  4d. Merge overlapping regions                                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: SPEECH REGIONS                       │
│                                                                 │
│  Format: List of (start, end) tuples in seconds                 │
│  Example: [(0.3, 4.3), (4.8, 9.5), (10.2, 13.8), ...]         │
│  Properties: Continuous, overlap-preserved, no speaker info     │
│                                                                 │
│  Next: PixIT joint diarization-separation (Module 2+)           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete End-to-End Example

**Scenario**: Running SAD on the processed medical consultation from Module 0

### Initial State

```
File: clinic_visit_2026_01_15_processed.wav
Format: 16 kHz, Mono, 16-bit PCM, 178.0 seconds, RMS=0.1
Source: Module 0 output

Timeline (sample segments):
  [0.3-4.3s]:   Doctor speaking
  [4.8-4.98s]:  Patient "Um" (180ms backchannel)
  [5.2-9.5s]:   Patient speaking
  [9.8-9.95s]:  Doctor "Okay" (150ms backchannel)
  [10.2-13.8s]: Doctor speaking
  ...
  [45.0-46.6s]: Overlap (both speaking)
  ...
  [175.0-178.0s]: Patient speaking
```

---

### Step 1: Frame Audio

**Configuration**:
```
Frame size: 25ms = 400 samples at 16 kHz
Hop size: 10ms = 160 samples
Overlap: (400-160)/400 = 60%
```

**Calculate number of frames**:
```
Duration: 178.0 seconds
Total samples: 178.0 × 16,000 = 2,848,000 samples

Number of frames:
  (Total samples - Frame size) / Hop size + 1
  (2,848,000 - 400) / 160 + 1
  = 17,798 frames
```

**Frame layout**:
```
Frame 1:  samples [0:400]       → time [0.000-0.025s]
Frame 2:  samples [160:560]     → time [0.010-0.035s]
Frame 3:  samples [320:720]     → time [0.020-0.045s]
...
Frame 17,798: samples [2,847,520:2,847,920] → time [177.970-177.995s]
```

---

### Step 2: Extract Features

**For each frame, calculate Energy and ZCR**

**Example: Frame at 5.0s (Patient "Um" starts)**

Frame 500 (approximately 5.0s):
```
Samples: [0.12, 0.18, -0.15, 0.22, -0.10, 0.25, ...]
(400 samples total)

Energy calculation:
  Squared: [0.0144, 0.0324, 0.0225, 0.0484, 0.0100, 0.0625, ...]
  Sum of all 400 squared values: 18.4
  Energy = 18.4 / 400 = 0.046

ZCR calculation:
  Sign changes: 0.18→-0.15 (1), -0.15→0.22 (2), 0.22→-0.10 (3), ...
  Total crossings in frame: 87
  ZCR = 87 / 400 = 0.218
```

**Example: Frame at 4.5s (Silence between speakers)**

Frame 450 (approximately 4.5s):
```
Samples: [0.002, -0.001, 0.003, -0.002, 0.001, ...]
(400 samples total)

Energy calculation:
  Squared: [0.000004, 0.000001, 0.000009, ...]
  Sum: 0.0016
  Energy = 0.0016 / 400 = 0.000004

ZCR calculation:
  Crossings: 45
  ZCR = 45 / 400 = 0.113
```

**Feature summary for key frames**:

| Frame | Time (s) | Context | Energy | ZCR | Expected |
|-------|----------|---------|--------|-----|----------|
| 30 | 0.3 | Doctor speech | 0.052 | 0.195 | Speech |
| 450 | 4.5 | Silence | 0.000004 | 0.113 | Non-speech |
| 500 | 5.0 | Patient "Um" | 0.046 | 0.218 | Speech |
| 980 | 9.8 | Doctor "Okay" | 0.041 | 0.202 | Speech |
| 4500 | 45.0 | Overlap start | 0.068 | 0.225 | Speech |

---

### Step 3: Classify Frames

**Using PyAnnote Neural SAD**

**Process**:
```
1. Load PyAnnote model (pre-trained)
2. Extract log-mel spectrogram from audio
3. Forward pass through network:
   - CNN layers detect spectral patterns
   - LSTM layers model temporal context
   - Output layer produces probability per frame
4. Apply threshold (0.5 for high-recall)
```

**Classification results (sample frames)**:

| Frame | Time (s) | Energy | ZCR | PyAnnote Prob | Threshold | Decision |
|-------|----------|--------|-----|---------------|-----------|----------|
| 30 | 0.3 | 0.052 | 0.195 | 0.94 | 0.5 | 1 (Speech) |
| 450 | 4.5 | 0.000004 | 0.113 | 0.02 | 0.5 | 0 (Non-speech) |
| 480 | 4.8 | 0.012 | 0.180 | 0.45 | 0.5 | 0 (Non-speech) |
| 500 | 5.0 | 0.046 | 0.218 | 0.89 | 0.5 | 1 (Speech) |
| 518 | 5.18 | 0.038 | 0.205 | 0.82 | 0.5 | 1 (Speech) |
| 520 | 5.2 | 0.008 | 0.145 | 0.38 | 0.5 | 0 (Non-speech) |
| 550 | 5.5 | 0.048 | 0.210 | 0.91 | 0.5 | 1 (Speech) |
| 980 | 9.8 | 0.041 | 0.202 | 0.87 | 0.5 | 1 (Speech) |
| 4500 | 45.0 | 0.068 | 0.225 | 0.96 | 0.5 | 1 (Speech) |

**Raw decision sequence (frames 30-600, representing 0.3-6.0s)**:
```
111111111111111111111111111111111111111111111111  (frames 30-77: Doctor)
000000000000000000000000000000000000000000000000  (frames 78-125: Silence)
000000000000000000000000000000000000000000000000  (frames 126-173: Silence)
000000000000000000000000000000000000000000000000  (frames 174-221: Silence)
000000000000000000000000000000000000000000000000  (frames 222-269: Silence)
000000000000000000000000000000000000000000000000  (frames 270-317: Silence)
000000000000000000000000000000000000000000000000  (frames 318-365: Silence)
000000000000000000000000000000000000000000000000  (frames 366-413: Silence)
000000000000000000000000000000000000000000000000  (frames 414-461: Silence)
000000000000000000000000000000000000000000000000  (frames 462-479: Silence)
111111111111111111  (frames 480-497: Transition)
11111111111111111111111111111111111111111111111111  (frames 498-545: Patient "Um" + start of next)
111111111111111111111111111111111111111111111111  (frames 546-593: Patient continues)
```

---

### Step 4: Temporal Smoothing

**4a. Fill short gaps (<300ms = 30 frames)**

```
Before (frames 30-600):
  Speech: [30-77]   (Doctor)
  Gap:    [78-479]  (402 frames = 4020ms) ← Too long, don't fill
  Speech: [480-497] (Transition)
  Gap:    [498-500] (2 frames = 20ms) ← Fill this!
  Speech: [501-593] (Patient)

After filling gaps <30 frames:
  Speech: [30-77]
  Gap:    [78-479]
  Speech: [480-593] (merged: 480-497 + 498-500 + 501-593)
```

**4b. Remove short bursts (<150ms = 15 frames, except backchannels)**

```
Check for isolated short segments:
  [480-497]: 17 frames = 170ms ← Keep (above 150ms threshold)
  
No segments to remove in this example
```

**4c. Extend boundaries (±300ms = ±30 frames)**

```
Speech region [30-77]:
  Original: frames 30-77 → time [0.3-0.77s]
  Extended: frames 0-107 → time [0.0-1.07s]
  (added -30 frames before, +30 frames after)

Speech region [480-593]:
  Original: frames 480-593 → time [4.8-5.93s]
  Extended: frames 450-623 → time [4.5-6.23s]
  (added -30 frames before, +30 frames after)
```

**4d. Merge overlapping regions**

```
Extended regions:
  Region 1: [0-107]   (0.0-1.07s)
  Region 2: [450-623] (4.5-6.23s)
  
No overlap, keep as separate regions
```

**Final smoothed decisions for 0-6s**:
```
Speech: [0.0-1.07s]   (Doctor, with padding)
Non-speech: [1.07-4.5s]
Speech: [4.5-6.23s]   (Patient "Um" + next utterance, with padding)
```

---

### Complete Output

**Processing all 178 seconds**:

```
Total frames processed: 17,798
Processing time (GPU): 3.56 seconds (0.02× real-time)

Raw speech frames detected: 12,450 / 17,798 = 70%
After smoothing: 12,680 / 17,798 = 71% (some gaps filled)
```

**Final speech regions**:

```
[(0.0, 4.8),      # Doctor first utterance (with padding)
 (4.5, 9.8),      # Patient "Um" + utterance (regions merged)
 (9.5, 14.2),     # Doctor "Okay" + utterance (regions merged)
 (14.0, 18.5),    # Patient response
 (18.2, 23.6),    # Doctor question
 ...
 (44.7, 47.0),    # Overlap region (both speakers, preserved as one)
 ...
 (174.7, 178.3)]  # Final utterance (with padding)

Total regions: 87
Total speech time: 125.4 seconds (70.4% of audio)
Non-speech removed: 52.6 seconds (29.6% of audio)
```

**Quality checks**:
```
✓ Short segments preserved:
  - Patient "Um" [4.8-4.98s]: 180ms ✓
  - Doctor "Okay" [9.8-9.95s]: 150ms ✓
  - All 12 backchannels intact

✓ Overlaps preserved:
  - Overlap [45.0-46.6s] marked as continuous speech [44.7-47.0s]
  - No attempt to separate speakers

✓ Boundary padding applied:
  - All regions extended by ±300ms
  - Overlapping padded regions merged

✓ High recall achieved:
  - Estimated recall: 97.2% (missed only 2.8% of actual speech)
  - Estimated precision: 89.5% (10.5% false positives)

✓ No speaker decisions made:
  - Output contains only time regions
  - No speaker labels
  - No speaker change markers
```

**Output format**:

```python
# Python list of tuples
speech_regions = [
    (0.0, 4.8),
    (4.5, 9.8),
    (9.5, 14.2),
    # ... 87 regions total
]

# Or RTTM-like format (without speaker labels)
# SPEECH file1 1 0.0 4.8
# SPEECH file1 1 4.5 9.8
# SPEECH file1 1 9.5 14.2
# ...
```

**Ready for Module 2**: PixIT will receive these 87 speech regions and perform joint speaker diarization and overlap separation.

---

## Key Constraints

**MUST DO**:
- Preserve all potential speech (high recall)
- Keep overlaps intact
- Protect short segments (down to 150ms)
- Add boundary padding (±300ms)
- Operate in high-recall mode

**MUST NOT DO**:
- Speaker change detection
- Overlap resolution
- Speaker segmentation
- Speaker counting
- Language identification
- Any speaker-level decisions

---

## Failure Modes to Avoid

| Wrong Behavior | Consequence |
|----------------|-------------|
| Aggressive silence removal | Deletes short backchannels |
| Energy-based SAD only | Misses low-energy patient speech |
| Overlap suppression | Breaks PixIT overlap detection |
| Early speaker splits | Confuses segmentation models |
| Fixed threshold in noise | Inconsistent performance |

---

## Success Criteria
1. Recall: At least 95% of actual speech detected
2. Precision: At least 85% of detections are speech
3. Continuous regions (not fragmented)
4. Overlaps preserved (not suppressed)
5. Processing faster than real-time
6. Short segments protected (down to 150ms)
7. No speaker-level decisions made

---

## Technical Specifications

| Parameter | Value | Reason |
|-----------|-------|--------|
| Frame Size | 25ms | Captures local speech characteristics |
| Hop Size | 10ms | 60% overlap ensures smooth coverage |
| Energy Threshold | 0.005-0.01 | Low threshold for high recall |
| Min Segment | 150ms | Preserves backchannels |
| Max Gap to Fill | 300ms | Merges close speech |
| Boundary Padding | ±300ms | Protects speech edges |
| Target Recall | Greater than 95% | Never miss speech |
| Target Precision | Greater than 85% | Balance false positives |

---

## Summary
Speech Activity Detection identifies where speech exists without determining who is speaking, operating as a conservative computational filter that preserves all potential speech (including overlaps and short segments) for PixIT and powerset models to process.

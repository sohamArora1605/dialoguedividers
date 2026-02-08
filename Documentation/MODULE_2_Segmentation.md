# MODULE 2 — NEURAL SEGMENTATION (Speaker-Turn & Overlap Segmentation)

*(DISPLACE-2026 compatible, PixIT + powerset aware, NO classical SCD)*

---

## 2.1 What This Module Does

**Purpose (2-line summary)**  
Converts coarse speech regions from SAD into fine-grained time frames with speaker-activity representations (single-speaker and overlap frames), without assigning speaker identities, using neural segmentation instead of classical rule-based methods.

**Input → Output**
- **Input**: Speech regions from Module 1 (time-stamped segments with no speaker labels)
- **Output**: Frame-level speaker-activity representations (probabilities for single-speaker, overlap, silence)

---

## 2.2 Core Design Decision (Critical)

### ❌ What Was REMOVED (Classical Approach)

**Completely eliminated from DISPLACE-2026 pipeline**:
- Energy-based segmentation
- Speaker Change Detection (SCD) using BIC, KL-divergence, or GLR
- Heuristic turn splitting based on silence duration
- Hard boundary detection algorithms
- Fixed-window sliding approaches

### ✅ What REPLACED It (Neural Approach)

**Neural segmentation learned end-to-end**:
- PixIT joint segmentation (primary)
- Powerset neural segmentation (secondary/ensemble)

**Why This Change?**

*Simple*: Medical conversations are messy - people interrupt, overlap, speak softly. Rule-based methods can't handle this.

*Technical*: Classical SCD assumes:
- Abrupt speaker changes (false in medical dialogue)
- Single speaker per segment (false with overlaps)
- Energy differences between speakers (false with soft-spoken patients)
- Clear silence boundaries (false with rapid turn-taking)

Neural methods **learn** conversation patterns from data instead of relying on brittle assumptions.

---

## 2.3 Why Classical SCD Fails for DISPLACE-2026

| Classical SCD Method | Assumption | Why It Fails in Medical Audio |
|---------------------|------------|-------------------------------|
| **Energy-based** | Speaker changes = energy changes | Patients often speak softly; energy may not change significantly |
| **BIC (Bayesian Information Criterion)** | Gaussian distribution of features | Medical overlaps violate single-distribution assumption |
| **Silence-based splitting** | Speakers separated by silence | Rapid turn-taking, backchannels, interruptions have minimal silence |
| **Fixed-window SCD** | Speaker changes align with windows | Natural turns don't align with arbitrary time grids |
| **GLR (Generalized Likelihood Ratio)** | Statistical difference between segments | Overlaps create mixed statistics that confuse GLR |

**Numerical Example of Failure**:

```
Medical conversation timeline:
Doctor:  [10.0-12.5s] "How long have you been feeling—"
Patient: [12.3-12.5s] "Um" ← 200ms overlap!
Patient: [12.5-15.0s] "about two weeks"

Classical SCD analysis:
- Energy at 12.3s: Mixed (doctor + patient)
- BIC test at 12.3s: Fails to detect change (mixed distribution)
- Silence detection: No silence found
- Result: Misses speaker change, treats as single speaker ✗

Neural segmentation:
- Learns overlap pattern from training data
- Outputs: [10.0-12.3s] single, [12.3-12.5s] overlap, [12.5-15.0s] single ✓
```

---

## 2.4 Two Parallel Segmentation Paths

**Module 2 is NOT a single algorithm**. It runs **two complementary approaches** in parallel:

### Path A: PixIT Joint Segmentation (Primary)
- End-to-end learned segmentation + separation
- Implicit speaker-activity modeling
- Optimized for overlap handling

### Path B: Powerset Neural Segmentation (Secondary/Ensemble)
- Explicit multi-label classification
- Treats speaker combinations as distinct classes
- Provides diversity for ensemble methods

**Both operate on the same speech regions from Module 1.**

---

## 2.5 Path A — PixIT Joint Segmentation (Detailed)

### 2.5.1 What PixIT Does (Segmentation Perspective)

**PixIT (Permutation Invariant Training for source separation) learns**:
1. Where speech exists (refines SAD)
2. Where overlaps exist (overlap detection)
3. How many speakers are active per frame (speaker counting)
4. **Jointly with** separating overlapping speakers (source separation)

**Key Insight**: Segmentation is **not a separate step** — it's learned implicitly as part of the separation objective.

### 2.5.2 PixIT Architecture Overview

```
Input: Mixed audio waveform
    ↓
Encoder (CNN or Transformer)
    ↓
Separation Network (LSTM/Transformer)
    ↓
Decoder (Mask estimation)
    ↓
Output: Per-speaker waveforms + activity masks
```

**Segmentation comes from activity masks**:
- Each speaker gets an activity mask (0-1 per frame)
- Overlap = multiple masks > threshold simultaneously
- Single-speaker = one mask > threshold

### 2.5.3 PixIT Segmentation Characteristics

**Frame-level granularity**:
- Typical: 10-20ms frames
- Much finer than classical SCD (which uses 1-3 second windows)

**Overlap-aware by design**:
- No explicit "overlap detection" step needed
- Overlaps emerge naturally from multi-speaker separation

**No hard boundaries**:
- Soft probabilities instead of binary decisions
- Reduces fragmentation errors

**Joint optimization**:
- Segmentation loss + separation loss trained together
- Segmentation improves separation, separation improves segmentation

### 2.5.4 PixIT Training Process

**Loss Function**:

```
Total Loss = Separation Loss + Activity Loss

Separation Loss (SI-SNR):
  SI-SNR = 10 × log₁₀(||s_target||² / ||s_target - s_estimated||²)
  
  Where:
  - s_target = ground truth speaker signal
  - s_estimated = separated speaker signal
  
Activity Loss (Binary Cross-Entropy):
  BCE = -Σ[y × log(p) + (1-y) × log(1-p)]
  
  Where:
  - y = ground truth activity (0 or 1)
  - p = predicted activity probability
```

**Training Data Requirements**:
- Mixed audio with overlaps
- Ground truth per-speaker signals
- Frame-level activity annotations

**Augmentation**:
- Synthetic overlaps (mix clean utterances)
- Noise injection
- Reverberation
- Speed/pitch perturbation

### 2.5.5 PixIT Inference Process

**Step-by-step**:

```
1. Load mixed audio (from Module 1 speech regions)
2. Extract features (STFT or learned representations)
3. Forward pass through encoder
4. Separation network predicts per-speaker masks
5. Apply masks to mixed spectrogram
6. Inverse STFT → per-speaker waveforms
7. Extract activity from mask magnitudes
8. Threshold activity → binary segmentation
```

**Numerical Example**:

```
Input: Mixed audio [10.0-15.0s] (5 seconds)
Frame size: 20ms
Frames: 250

PixIT output (2 speakers):

Frame | Time (s) | Speaker A Mask | Speaker B Mask | Interpretation
------|----------|----------------|----------------|---------------
0     | 10.00    | 0.92          | 0.05          | A speaking
1     | 10.02    | 0.94          | 0.03          | A speaking
...
115   | 12.30    | 0.88          | 0.85          | Overlap!
116   | 12.32    | 0.90          | 0.87          | Overlap!
117   | 12.34    | 0.91          | 0.86          | Overlap!
...
125   | 12.50    | 0.06          | 0.91          | B speaking
126   | 12.52    | 0.04          | 0.93          | B speaking
...

Threshold: 0.5
Segmentation output:
  [10.00-12.30s]: Speaker A only
  [12.30-12.50s]: Overlap (A+B)
  [12.50-15.00s]: Speaker B only
```

### 2.5.6 Why PixIT Segmentation is Superior

**Advantages over classical SCD**:

1. **Overlap modeling**: Overlaps are explicitly modeled, not guessed
2. **No hard boundaries**: Soft probabilities reduce fragmentation
3. **Context-aware**: LSTM/Transformer layers use long-range context
4. **Data-driven**: Learns medical conversation patterns from training data
5. **Joint optimization**: Segmentation and separation reinforce each other

**Medical conversation benefits**:

```
Scenario: Doctor interrupts patient

Classical SCD:
  Patient: [10.0-12.5s] (misses interruption)
  Result: Single segment ✗

PixIT:
  Patient: [10.0-12.3s]
  Overlap: [12.3-12.5s] (doctor interrupts)
  Doctor:  [12.5-15.0s]
  Result: Three segments ✓
```

---

## 2.6 Path B — Powerset Neural Segmentation (Detailed)

### 2.6.1 What Powerset Segmentation Means

**Traditional approach** (multi-label):
```
Frame → {Speaker A: yes/no, Speaker B: yes/no}
```

**Powerset approach** (single-label with combined classes):
```
Frame → {∅, {A}, {B}, {A+B}}
```

Each **speaker combination** is treated as **one distinct class**.

### 2.6.2 Powerset Class Structure

**For 2 speakers**:

| Class ID | Label | Meaning |
|----------|-------|---------|
| 0 | ∅ | Silence (no speakers) |
| 1 | {A} | Only speaker A |
| 2 | {B} | Only speaker B |
| 3 | {A+B} | Both speakers (overlap) |

**For 3 speakers** (if needed):

| Class ID | Label | Meaning |
|----------|-------|---------|
| 0 | ∅ | Silence |
| 1 | {A} | Only A |
| 2 | {B} | Only B |
| 3 | {C} | Only C |
| 4 | {A+B} | A and B overlap |
| 5 | {A+C} | A and C overlap |
| 6 | {B+C} | B and C overlap |
| 7 | {A+B+C} | All three overlap |

**Total classes = 2^N** (where N = number of speakers)

### 2.6.3 Why Powerset Works for DISPLACE-2026

**Medical conversation characteristics**:
- Usually 2 speakers (doctor + patient)
- Overlaps are common but typically involve only 2 speakers
- Short overlaps (interruptions, backchannels)
- Predictable patterns (doctor asks, patient responds, occasional overlap)

**Powerset advantages**:

1. **Explicit overlap modeling**: {A+B} is a distinct class, not inferred
2. **Manageable complexity**: 2^2 = 4 classes for 2 speakers
3. **Direct optimization**: Cross-entropy loss on overlap class
4. **No post-processing**: Overlaps are predicted directly

**Numerical Example**:

```
Medical conversation:
Doctor:  [10.0-12.3s]
Overlap: [12.3-12.5s] (doctor + patient)
Patient: [12.5-15.0s]

Powerset prediction (per frame):

Frame | Time (s) | Class Probabilities              | Predicted Class
------|----------|----------------------------------|----------------
0     | 10.00    | [0.02, 0.94, 0.03, 0.01]        | {A} (Doctor)
...
115   | 12.30    | [0.01, 0.45, 0.05, 0.49]        | {A+B} (Overlap)
116   | 12.32    | [0.02, 0.42, 0.06, 0.50]        | {A+B} (Overlap)
...
125   | 12.50    | [0.03, 0.04, 0.91, 0.02]        | {B} (Patient)
...

Class labels: [∅, {A}, {B}, {A+B}]
```

### 2.6.4 Powerset Architecture

```
Input: Audio features (log-mel spectrogram or embeddings)
    ↓
Feature Extraction (CNN or Transformer)
    ↓
Temporal Modeling (LSTM or Transformer)
    ↓
Classification Head (Fully connected → Softmax)
    ↓
Output: Class probabilities per frame
```

**Training**:

```
Loss: Categorical Cross-Entropy

CCE = -Σ y_c × log(p_c)

Where:
- y_c = one-hot ground truth (1 for correct class, 0 otherwise)
- p_c = predicted probability for class c
```

### 2.6.5 Powerset Constraints (Important)

**Maximum simultaneous speakers**:
- DISPLACE-2026: Max 2 speakers (doctor + patient)
- If 3-way conversations exist: Max 3 speakers
- Complexity: 2^N classes

**Why limit to 2-3 speakers?**
- Medical consultations rarely have >2 speakers
- Exponential class growth (2^4 = 16 classes for 4 speakers)
- Training data becomes sparse for rare combinations

**Frame-level prediction**:
- Typical: 10-20ms frames
- Same granularity as PixIT

**Explicit overlap classes**:
- {A+B}, {A+C}, etc. are distinct classes
- No ambiguity in overlap representation

---

## 2.7 Comparison: PixIT vs Powerset

| Aspect | PixIT | Powerset |
|--------|-------|----------|
| **Approach** | Source separation + activity | Multi-class classification |
| **Overlap handling** | Implicit (from masks) | Explicit (overlap classes) |
| **Output** | Continuous masks | Discrete class labels |
| **Complexity** | High (separation network) | Medium (classification) |
| **Training data** | Needs separated signals | Needs frame labels |
| **Scalability** | Scales to many speakers | Limited by 2^N classes |
| **Medical audio** | Excellent (handles noise) | Good (if 2-3 speakers) |
| **Use case** | Primary segmentation | Ensemble diversity |

**When to use which?**

- **PixIT**: Primary method, especially if separation is needed later
- **Powerset**: Secondary method for ensemble, or if only segmentation is needed
- **Both**: Ensemble for best performance (average predictions)

---

## 2.8 What Module 2 Does NOT Do

**Critical constraints**:

❌ **No speaker embeddings**: Module 2 doesn't extract speaker representations  
❌ **No clustering**: Module 2 doesn't group similar speakers  
❌ **No speaker IDs**: Module 2 doesn't assign "Speaker 1", "Speaker 2" labels  
❌ **No speaker counting** (total): Module 2 doesn't determine how many unique speakers exist  
❌ **No re-segmentation of Module 1**: Module 2 never marks speech regions as non-speech

**What Module 2 DOES answer**:

> "How many speakers are active **at this specific time frame**?"

**What Module 2 does NOT answer**:

> "Who are they?" (answered by Module 3 + Module 4)  
> "How many unique speakers total?" (answered by Module 4)

---

## 2.9 Interaction with Module 1 (SAD)

**Division of responsibilities**:

| Module | Responsibility |
|--------|---------------|
| **Module 1 (SAD)** | Removes obvious non-speech (silence, noise) |
| **Module 2 (Segmentation)** | Refines speech structure (single vs overlap) |

**Important rules**:

1. **Module 2 never re-introduces non-speech**: If Module 1 marked a region as non-speech, Module 2 doesn't process it
2. **Module 2 assumes everything from Module 1 may be speech**: Even if energy is low, Module 2 processes it
3. **Module 2 refines structure, not presence**: Module 2 determines "how many speakers", not "speech vs silence"

**Example**:

```
Module 1 output:
  Speech: [10.0-15.0s]
  Non-speech: [15.0-20.0s]
  Speech: [20.0-25.0s]

Module 2 processing:
  Process [10.0-15.0s] → segment into single/overlap frames ✓
  Skip [15.0-20.0s] → already marked non-speech ✓
  Process [20.0-25.0s] → segment into single/overlap frames ✓

Module 2 output:
  [10.0-12.3s]: Single speaker
  [12.3-12.5s]: Overlap
  [12.5-15.0s]: Single speaker
  [15.0-20.0s]: Non-speech (preserved from Module 1)
  [20.0-25.0s]: Single speaker
```

---

## 2.10 Output Representations

### 2.10.1 PixIT Output Format

**Per-speaker activity masks**:

```
Shape: (num_speakers, num_frames)
Values: [0.0, 1.0] (continuous probabilities)

Example (2 speakers, 250 frames):
Speaker A: [0.92, 0.94, 0.91, ..., 0.88, 0.06, 0.04, ...]
Speaker B: [0.05, 0.03, 0.04, ..., 0.85, 0.91, 0.93, ...]

Interpretation:
  Frame 0: A=0.92, B=0.05 → A speaking
  Frame 115: A=0.88, B=0.85 → Overlap
  Frame 125: A=0.06, B=0.91 → B speaking
```

**Thresholding**:

```
Threshold: 0.5 (typical)

Binary activity:
  A_active = (A_mask > 0.5)
  B_active = (B_mask > 0.5)

Speaker count per frame:
  count = A_active + B_active
  
  count = 0 → Silence
  count = 1 → Single speaker
  count = 2 → Overlap
```

### 2.10.2 Powerset Output Format

**Class probabilities per frame**:

```
Shape: (num_frames, num_classes)
Values: [0.0, 1.0] (probabilities sum to 1 per frame)

Example (250 frames, 4 classes):
Frame 0:   [0.02, 0.94, 0.03, 0.01] → {A}
Frame 115: [0.01, 0.45, 0.05, 0.49] → {A+B}
Frame 125: [0.03, 0.04, 0.91, 0.02] → {B}

Classes: [∅, {A}, {B}, {A+B}]
```

**Predicted class**:

```
predicted_class = argmax(probabilities)

Frame 0: argmax([0.02, 0.94, 0.03, 0.01]) = 1 → {A}
Frame 115: argmax([0.01, 0.45, 0.05, 0.49]) = 3 → {A+B}
Frame 125: argmax([0.03, 0.04, 0.91, 0.02]) = 2 → {B}
```

### 2.10.3 Time Alignment

**Both outputs are time-aligned with original audio**:

```
Frame size: 20ms
Hop size: 10ms (50% overlap)

Frame 0: [0.000-0.020s]
Frame 1: [0.010-0.030s]
Frame 2: [0.020-0.040s]
...

Mapping frame index to time:
  time_start = frame_index × hop_size
  time_end = time_start + frame_size
```

---

## 2.11 Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Speech Regions from Module 1                            │
│ Example: [(0.0, 4.8), (4.5, 9.8), (9.5, 14.2), ...]           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PREPROCESSING: Extract Features                    │
│ - Load audio for each speech region                            │
│ - Compute STFT or log-mel spectrogram                          │
│ - Normalize features                                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│ PATH A: PixIT            │  │ PATH B: Powerset         │
│                          │  │                          │
│ 1. Encoder (CNN/Trans)   │  │ 1. Feature Extraction    │
│ 2. Separation Network    │  │ 2. Temporal Modeling     │
│ 3. Mask Estimation       │  │ 3. Classification Head   │
│ 4. Activity Extraction   │  │ 4. Softmax               │
│                          │  │                          │
│ Output: Activity masks   │  │ Output: Class probs      │
│ Shape: (speakers, frames)│  │ Shape: (frames, classes) │
└──────────────────────────┘  └──────────────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              POST-PROCESSING: Combine & Threshold               │
│ - Threshold PixIT masks (> 0.5)                                │
│ - Argmax powerset classes                                      │
│ - Optional: Ensemble (average predictions)                     │
│ - Convert to time segments                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Frame-level Speaker Activity                           │
│                                                                 │
│ Format: List of (start, end, num_speakers)                     │
│ Example:                                                        │
│   [(0.0, 12.3, 1),    # Single speaker                         │
│    (12.3, 12.5, 2),   # Overlap                                │
│    (12.5, 15.0, 1)]   # Single speaker                         │
│                                                                 │
│ Next: MODULE 3 (Speaker Embedding Extraction)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2.12 Complete End-to-End Example

**Scenario**: Segmenting a 5-second medical consultation excerpt

### Initial State

```
Input from Module 1:
  Speech region: [10.0-15.0s]

Audio content (ground truth):
  [10.0-12.3s]: Doctor speaking
  [12.3-12.5s]: Doctor + Patient overlap (patient says "mm-hmm")
  [12.5-15.0s]: Patient speaking

Frame configuration:
  Frame size: 20ms
  Hop size: 10ms
  Total frames: 500
```

---

### Path A: PixIT Processing

**Step 1: Feature Extraction**

```
Load audio [10.0-15.0s]
Compute STFT:
  FFT size: 512
  Hop: 160 samples (10ms at 16 kHz)
  Spectrogram shape: (257 freq bins, 500 frames)
```

**Step 2: PixIT Forward Pass**

```
Encoder output: (512 channels, 500 frames)
Separation network: Predicts 2 speaker masks
Mask shapes: (257 freq bins, 500 frames) each

Speaker A mask (sample values):
  Frame 0:   0.92 (high confidence)
  Frame 115: 0.88 (still high, but overlap starting)
  Frame 125: 0.06 (low, speaker B dominant)

Speaker B mask (sample values):
  Frame 0:   0.05 (low)
  Frame 115: 0.85 (high, overlap)
  Frame 125: 0.91 (high confidence)
```

**Step 3: Activity Extraction**

```
Average mask magnitude across frequency bins:

Speaker A activity:
  [0.92, 0.94, 0.91, ..., 0.88, 0.87, ..., 0.06, 0.04, ...]
  
Speaker B activity:
  [0.05, 0.03, 0.04, ..., 0.85, 0.87, ..., 0.91, 0.93, ...]
```

**Step 4: Thresholding**

```
Threshold: 0.5

Frame | A_mask | B_mask | A_active | B_active | Interpretation
------|--------|--------|----------|----------|---------------
0     | 0.92   | 0.05   | 1        | 0        | A only
115   | 0.88   | 0.85   | 1        | 1        | Overlap
125   | 0.06   | 0.91   | 0        | 1        | B only
```

**PixIT Output**:

```
Segments:
  [10.00-12.30s]: 1 speaker (A)
  [12.30-12.50s]: 2 speakers (overlap)
  [12.50-15.00s]: 1 speaker (B)
```

---

### Path B: Powerset Processing

**Step 1: Feature Extraction**

```
Same as PixIT: Log-mel spectrogram
Shape: (80 mel bins, 500 frames)
```

**Step 2: Powerset Forward Pass**

```
Feature extraction (CNN): (256 channels, 500 frames)
Temporal modeling (LSTM): (256 hidden units, 500 frames)
Classification head: (4 classes, 500 frames)

Classes: [∅, {A}, {B}, {A+B}]
```

**Step 3: Softmax Probabilities**

```
Frame | ∅    | {A}  | {B}  | {A+B} | Predicted
------|------|------|------|-------|----------
0     | 0.02 | 0.94 | 0.03 | 0.01  | {A}
115   | 0.01 | 0.45 | 0.05 | 0.49  | {A+B}
125   | 0.03 | 0.04 | 0.91 | 0.02  | {B}
```

**Powerset Output**:

```
Segments:
  [10.00-12.30s]: {A}
  [12.30-12.50s]: {A+B} (overlap)
  [12.50-15.00s]: {B}
```

---

### Ensemble (Optional)

**Combine PixIT and Powerset**:

```
Method: Average speaker activity

Frame 115 (overlap):
  PixIT: A=0.88, B=0.85
  Powerset: {A+B} probability = 0.49
  
  Ensemble decision:
    - Both methods agree on overlap
    - High confidence → Overlap confirmed
```

---

### Final Output

```
Segmentation result:
  [10.00-12.30s]: Single speaker (1 active)
  [12.30-12.50s]: Overlap (2 active)
  [12.50-15.00s]: Single speaker (1 active)

Comparison with ground truth:
  [10.0-12.3s]: Doctor ✓ (correctly identified as single)
  [12.3-12.5s]: Overlap ✓ (correctly identified)
  [12.5-15.0s]: Patient ✓ (correctly identified as single)

Accuracy: 100% (all frames correctly classified)
```

---

## 2.13 Key Terminology

| Term | Simple | Technical | Why It Matters |
|------|--------|-----------|----------------|
| **Segmentation** | Dividing audio into parts | Temporal partitioning of audio | Enables per-speaker processing |
| **SCD (Speaker Change Detection)** | Finding when speakers change | Detecting speaker boundaries | Classical method, removed in DISPLACE-2026 |
| **PixIT** | Separation + segmentation | Permutation Invariant Training | Primary method for neural segmentation |
| **Powerset** | All speaker combinations | 2^N classes for N speakers | Explicit overlap modeling |
| **Activity mask** | Speaker presence per frame | Binary or continuous indicator | Output of PixIT segmentation |
| **Overlap** | Multiple speakers simultaneously | Temporal region with >1 active speaker | Common in medical conversations |
| **Frame** | Small time window | Typically 10-20ms | Basic unit of segmentation |
| **Joint optimization** | Training multiple tasks together | Segmentation + separation loss | Improves both tasks simultaneously |

---

## 2.14 Success Criteria

Module 2 is successful if:

1. **Overlap detection**: ≥90% of overlap regions correctly identified
2. **Single-speaker accuracy**: ≥95% of single-speaker frames correct
3. **Temporal precision**: Boundaries within ±50ms of ground truth
4. **No false non-speech**: Never marks speech regions as silence
5. **Consistent with Module 1**: All Module 1 speech regions processed
6. **Frame-level output**: Fine-grained (10-20ms) segmentation
7. **No speaker IDs**: Output contains activity counts, not identities
8. **Processing speed**: <2.0× real-time (acceptable for offline processing)

---

## 2.15 What Module 2 Does NOT Do

**Does NOT perform**:
- Speaker embedding extraction (Module 3)
- Speaker clustering (Module 4)
- Speaker identification/labeling (Module 4)
- Total speaker counting (Module 4)
- Re-segmentation of non-speech regions (Module 1 decision is final)
- Speaker verification or recognition
- Language identification
- Emotion detection

**Module 2 is pure segmentation** - it determines "how many speakers when" without knowing "who they are."

---

## 2.16 Temporal Smoothing & Minimum-Duration Enforcement

### 2.16.1 Why This Is Needed

**Neural frame-level predictions can produce noise**:

1. **1-3 frame flips** (10-30 ms jitter)
2. **Very short bursts**:
   - A → overlap → A (for 30 ms)
   - A → B → A (for 40 ms)

**These are not real speaker turns, just model noise.**

**If you keep them**:
- Module 3 drops segments <250 ms (minimum duration requirement)
- Coverage reduces (fewer embeddings extracted)
- Clustering becomes weaker (insufficient data)
- DER increases (fragmentation errors)

**Solution**: Stabilize the segmentation timeline through post-processing.

### 2.16.2 Input and Output

**Input**: Frame-level activity decisions from PixIT / Powerset

```
Example (per frame, hop = 10 ms):
[1,1,1,1,2,2,1,1,1,1,1, ...]

Where:
  1 = single speaker
  2 = overlap (2 speakers active)
```

**Output**: Cleaned, stable sequence

```
[1,1,1,1,1,1,1,1,1,1,1, ...]

Then converted to segments:
[(0.0, 12.5, 1), (12.5, 15.0, 1)]
```

### 2.16.3 Step 1 — Median Filter (Frame-Level)

**Apply median filter over activity count sequence.**

**Parameters**:
- Window size: 5 frames (≈ 50 ms) or 7 frames (≈ 70 ms)
- Operation: sliding window median

**Why median, not average?**
- Preserves sharp boundaries
- Removes isolated spikes
- Does not blur overlaps

**Example**:

```
Before median filter:
Frames:      1  1  1  1  2  1  1  1
             ↑  ↑  ↑  ↑  ↑  ↑  ↑  ↑
             
After median(5):
Frames:      1  1  1  1  1  1  1  1
                        ↑
                  Fake 10 ms overlap removed ✓
```

**Numerical Example**:

```
Input sequence (20 frames):
[1, 1, 1, 1, 2, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 2, 1, 1, 1, 1]
 ↑           ↑              ↑           ↑     ↑
 Single      Spike          Real        Spike
             (noise)        overlap     (noise)

After median filter (window=5):
[1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 1, 1, 1, 1, 1, 1, 1, 1]
                ↑                          ↑
          Spike removed              Spike removed
          
Real overlap preserved ✓
```

### 2.16.4 Step 2 — Segment Extraction

**Convert frame labels → segments**:

```
Frame labels:
[1,1,1,1,1,1,2,2,2,1,1]

Segments:
[(0.00-0.06s, 1),   # Frames 0-5 (6 frames × 10ms = 60ms)
 (0.06-0.09s, 2),   # Frames 6-8 (3 frames × 10ms = 30ms)
 (0.09-0.11s, 1)]   # Frames 9-10 (2 frames × 10ms = 20ms)
```

### 2.16.5 Step 3 — Minimum Duration Enforcement

**Define minimum durations**:

```python
MIN_SINGLE = 0.25   # 250 ms (matches Module 3 requirement)
MIN_OVERLAP = 0.10  # 100 ms (overlaps can be shorter)
```

**Process each segment**:

**Case A: Segment is shorter than minimum**

```
Example:
(12.30-12.36, overlap) = 60 ms  ❌ too short (< 100ms)
```

**Rule 1 — If neighbors have same label → merge**

```
Before:
(1) 12.0-12.30  single
(2) 12.30-12.36 overlap   ❌ 60ms too short
(3) 12.36-13.0  single

After:
(1) 12.0-13.0   single    ✓ merged

Reasoning: Overlap too short, likely noise
```

**Rule 2 — Otherwise absorb into longer neighbor**

```
Example:
(1) 10.0-12.3   single (2.3s)
(2) 12.3-12.36  overlap (60ms) ❌
(3) 12.36-12.8  single (440ms)

Both neighbors are single → merge into single:
(1) 10.0-12.8   single ✓
```

**Rule 3 — Never create new classes**

```
We only:
  - Merge consecutive segments
  - Extend into neighbors
  - Absorb short segments

We NEVER:
  - Invent new labels
  - Change existing segment types arbitrarily
```

### 2.16.6 Special Rule for Overlaps

**Overlaps are important but also noisy.**

**Strategy**:
- Allow shorter minimum for overlap (100 ms vs 250 ms)
- But still remove:
  - 10-30 ms fake overlaps (neural jitter)
  - Single-frame spikes (10-20 ms)

**Why 100 ms minimum for overlaps?**
- Real backchannels ("mm-hmm", "yeah") are typically 100-200 ms
- Interruptions start with brief overlap before full turn
- Medical conversations have many short overlaps

**Example**:

```
Real backchannel:
Doctor:  [10.0-12.5s] "How long have you been feeling this way?"
Patient: [12.3-12.5s] "Mm-hmm" (200ms overlap)

Duration: 200ms > 100ms minimum ✓ Keep it

Fake overlap (neural noise):
Doctor:  [10.0-12.5s]
Patient: [12.3-12.33s] (30ms spike)

Duration: 30ms < 100ms minimum ✗ Remove it
```

### 2.16.7 Concrete Example

**Before smoothing**:

```
Frame labels (hop=10ms):
[1,1,1,1,2,1,1,1,1,2,2,2,1,1]
 ↑       ↑         ↑       ↑
 Single  Spike     Real    Single
         (noise)   overlap
```

**After median filter**:

```
[1,1,1,1,1,1,1,1,1,2,2,2,1,1]
         ↑
    Spike removed
```

**Segments**:

```
(0-90ms)   single   (9 frames × 10ms = 90ms)
(90-120ms) overlap  (3 frames × 10ms = 30ms) ❌ too short
(120-140ms) single  (2 frames × 10ms = 20ms)
```

**Apply minimum duration**:

```
Segment 2 (overlap, 30ms) < 100ms minimum ❌
Neighbors: single (left), single (right)
Action: Merge all three into single

Result:
(0-140ms) single ✓
```

**Final output**:

```
[(0.0, 0.14, 1)]  # Single segment, 140ms duration
```

### 2.16.8 Guarantees After This Step

**After temporal smoothing + minimum-duration enforcement**:

✅ **No 10-50 ms fake segments** (removed by median filter + min duration)  
✅ **No over-fragmentation** (merged short segments)  
✅ **All segments long enough for Module 3** (≥250ms for single-speaker)  
✅ **Overlaps preserved if real** (100ms minimum allows backchannels)  
✅ **Boundaries are stable** (no jitter)  
✅ **DER improves** (fewer fragmentation errors)

### 2.16.9 Where This Fits in the Pipeline

```
PixIT / Powerset outputs
        ↓
Frame-level activity
        ↓
▶▶ TEMPORAL SMOOTHING + MIN-DURATION ENFORCEMENT ◀◀
        ↓
Final segmentation structure
        ↓
Module 3 (Embedding Extraction)
```

### 2.16.10 Final Important Note

**This step is**:
- ✅ Post-processing of neural predictions to remove jitter
- ✅ Minimum-duration enforcement for downstream compatibility
- ✅ Standard practice in SOTA diarization systems

**This step is NOT**:
- ❌ Heuristic SCD (we're not detecting speaker changes)
- ❌ Rule-based segmentation (we're cleaning neural outputs)
- ❌ Replacing neural methods (we're refining them)

**All SOTA diarization systems incorporate temporal smoothing** - it's a necessary post-processing step to ensure stable, usable segmentation for downstream modules.

---

## 2.17 Summary

> **Module 2 learns "how many speakers are talking when" using neural segmentation (PixIT + powerset), replacing classical rule-based SCD with data-driven methods that handle medical conversation overlaps and rapid turn-taking.**

---

**Next Module**: MODULE 3 — Speaker Embedding Extraction

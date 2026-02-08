# MODULE 1 — SPEECH ACTIVITY DETECTION (SAD / VAD)

*(DISPLACE-2026 compatible, PixIT and powerset aware)*

---

## 1.1 What This Module Does

**Purpose (2-line summary)**  
Module 1 identifies where speech exists in time (speech vs non-speech regions) without determining who is speaking, operating as a conservative computational filter that preserves all potential speech for downstream models.

**Input and Output**
- **Input**: Standardized audio from Module 0 (16 kHz, mono, normalized)
- **Output**: Time regions marked as speech or non-speech, with overlaps preserved

---

## 1.2 Core Responsibility

Module 1 answers only one question:

> "Is there speech here or not?"

Module 1 does NOT answer:
- Who is speaking
- How many speakers exist
- Where speaker boundaries are
- What speaker labels should be

---

## 1.3 Why Module 1 Must Be Redesigned for DISPLACE-2026

### Medical Conversation Characteristics

Medical conversations present unique challenges:
- Very short speech bursts (backchannels, acknowledgments)
- Heavy speaker overlaps (interruptions, simultaneous speech)
- Low-energy patient speech (soft-spoken, hesitant)
- Noise that resembles speech (equipment beeps, ambient sounds)

### Required SAD Properties

**High Recall (Priority)**  
Never miss actual speech, even if quiet or brief

**Overlap-Aware**  
Preserve overlapping speech regions without attempting to separate them

**Non-Destructive**  
Avoid aggressive filtering that removes short or low-energy speech

**Why This Matters**  
Missing speech at this stage creates irrecoverable diarization errors later. False positives (marking noise as speech) can be corrected downstream, but false negatives (missing speech) cannot be recovered.

---

## 1.4 Relationship with PixIT and Powerset Models (Critical)

### Two Valid SAD Approaches

**Option A: SAD as Coarse Gate (Recommended)**  
SAD removes obvious non-speech regions before PixIT/powerset processing. SAD operates independently and conservatively.

**Option B: SAD Inside PixIT (Advanced)**  
PixIT performs joint speech detection and diarization. SAD is integrated into the model itself.

### DISPLACE-2026 Design Choice

We use **Option A: SAD as Coarse Gate**

**Reasoning**:
- PixIT is computationally expensive
- Running PixIT on long silence regions wastes resources
- SAD cheaply filters obvious non-speech
- SAD operates as a compute filter, not a diarization component

**Critical Rule**:  
SAD must NOT attempt fine segmentation or speaker separation. All speaker-level decisions happen in later modules.

---

## 1.5 Complete Breakdown (Technical and Non-Technical)

### 1.5.1 What is Speech Activity Detection?

**Non-Technical Explanation**  
Imagine listening to a recording of a medical consultation. Sometimes the doctor speaks, sometimes the patient speaks, sometimes there's silence while someone thinks, and sometimes you hear background noise like equipment beeping or doors closing. SAD is like a filter that marks "human voice here" or "no human voice here" for every moment in the recording.

**Technical Explanation**  
Speech Activity Detection performs binary temporal classification to distinguish regions containing human speech from regions containing silence, noise, music, or other non-speech audio. It operates at the frame level (typically 10-30ms windows) and produces time-stamped speech/non-speech decisions.

### 1.5.2 Why SAD is Critical

**Without SAD**:

```
Audio composition:
- 40% actual speech
- 30% silence
- 20% background noise
- 10% non-speech sounds (clicks, beeps, etc.)
```

**Problems**:
- Embedding extraction models process noise, creating spurious "speaker" representations
- Clustering algorithms treat noise embeddings as separate speakers
- Computational resources wasted on non-speech regions
- Higher Diarization Error Rate (DER)

**With SAD**:

```
Filtered audio:
- 100% speech regions only
```

**Benefits**:
- Embedding models process only actual speech
- No spurious speaker representations from noise
- 60% reduction in computational load
- Improved clustering quality and lower DER

---

### 1.5.3 Understanding Frames

**What is a Frame?**

**Non-Technical**  
A frame is a tiny slice of audio, like cutting a long rope into small overlapping pieces. Each piece is analyzed independently.

**Technical**  
A frame is a short-time window of audio samples, typically 10-30 milliseconds in duration. Frames overlap (usually 50-75%) to ensure smooth temporal coverage and prevent missing transitions at frame boundaries.

**Numerical Example**:

```
Audio: 10 seconds at 16 kHz = 160,000 samples

Frame Configuration:
- Frame size: 25ms = 400 samples
- Hop size: 10ms = 160 samples (60% overlap)
- Number of frames: (160,000 - 400) / 160 + 1 = 998 frames

Frame Layout:
Frame 1: samples [0:400]
Frame 2: samples [160:560]
Frame 3: samples [320:720]
...
```

**Why Overlap?**  
Speech events (phonemes, words) don't align with frame boundaries. Overlapping frames ensure we capture transitions and don't miss speech that starts mid-frame.

**Visual Representation**:

```
Audio Timeline:
|--------------------------------------------------|
|----Frame 1----|
      |----Frame 2----|
            |----Frame 3----|
                  |----Frame 4----|

Overlap ensures continuous coverage
```

---

### 1.5.4 How SAD Works (Algorithmic Pipeline)

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
(Threshold-based or Neural network)
        |
        v
Temporal Smoothing
(Merge gaps, remove short bursts, extend boundaries)
        |
        v
Speech Regions Output
```

---

### 1.5.5 Feature Extraction for SAD

SAD uses physics-based or learned features to distinguish speech from non-speech.

**Feature 1: Energy (Primary Feature)**

**What It Measures**  
The loudness or power of the audio frame.

**Formula**:

```
Energy = (1/N) × Σ(x[n]²)

Where:
- N = number of samples in frame
- x[n] = amplitude of sample n
- Σ = sum over all samples in frame
```

**Numerical Example**:

```
Frame samples: [0.1, 0.2, -0.15, 0.3, -0.1]

Step 1: Square each sample
Squared: [0.01, 0.04, 0.0225, 0.09, 0.01]

Step 2: Sum
Sum: 0.01 + 0.04 + 0.0225 + 0.09 + 0.01 = 0.1725

Step 3: Divide by number of samples
Energy: 0.1725 / 5 = 0.0345
```

**Interpretation**:

```
Silence:     Energy ≈ 0.0001  (very quiet)
Speech:      Energy ≈ 0.05    (moderate)
Loud noise:  Energy ≈ 0.08    (high)

Decision threshold typically: 0.005 - 0.02
```

**Why Energy Works**  
Speech has significantly higher energy than silence. However, energy alone cannot distinguish speech from noise (both can be loud), so additional features are needed.

**Feature 2: Zero-Crossing Rate (ZCR)**

**What It Measures**  
How frequently the waveform crosses the zero amplitude line.

**Numerical Example**:

```
Frame: [0.1, 0.2, -0.1, -0.2, 0.15, 0.3, -0.05]

Identify sign changes:
  0.2 → -0.1   (positive to negative) ← crossing
 -0.2 → 0.15   (negative to positive) ← crossing
  0.3 → -0.05  (positive to negative) ← crossing

Total crossings: 3
ZCR: 3 / 7 = 0.43
```

**Interpretation**:

```
Speech:  Smooth waveform → ZCR ≈ 0.1-0.3
Noise:   Random waveform → ZCR ≈ 0.4-0.6
Silence: Near-zero signal → ZCR ≈ 0.0-0.1
```

**Why ZCR Helps**  
Speech has periodic structure (vocal cord vibrations), resulting in lower ZCR. Random noise has high ZCR. Combined with energy, ZCR helps distinguish speech from noise.

**Combined Decision Logic**:

```
High Energy + Low ZCR   → Speech (likely)
High Energy + High ZCR  → Noise (likely)
Low Energy              → Silence (regardless of ZCR)
```

**Feature 3: Spectral Features (Optional)**

More advanced SAD systems use:
- **Log-mel energy**: Energy distribution across frequency bands
- **Spectral centroid**: Center of mass of the spectrum
- **Spectral flux**: Rate of spectral change
- **MFCCs**: Mel-frequency cepstral coefficients (simplified version of Module 3 features)

These features capture frequency-domain characteristics that distinguish speech from other sounds.

---

### 1.5.6 Classification Methods

**Method 1: Neural Network Classification (Modern)**

**How It Works**:

```
Audio → Neural Network → Frame Probabilities → Threshold → Regions

Neural network learns complex patterns from labeled data
```

**Architecture (Simplified)**:

```
Input: Audio frames or spectral features
Hidden Layers: CNN or RNN layers
Output: Probability of speech for each frame

Example: PyAnnote SAD uses CNN + LSTM architecture
```

**Advantages**:
- Learns complex speech patterns
- Adapts to diverse acoustic conditions
- Better performance in noisy environments
- State-of-the-art accuracy

**Disadvantages**:
- Requires labeled training data
- More computationally expensive
- Needs GPU for real-time processing (or slower on CPU)

---

### 1.5.7 Temporal Smoothing (Critical Post-Processing)

**The Problem**:

Raw frame-level decisions are noisy:

```
Raw decisions: 1 1 1 0 1 1 0 1 0 1 1 1
```

This is unrealistic. Speech doesn't flicker on/off every 10ms. Real speech has temporal continuity.

**Smoothing Operations**:

**Rule 1: Merge Nearby Speech (Fill Short Gaps)**

```
Before: 1 1 1 0 0 1 1 1
               ↑
           Short gap (20ms)

After:  1 1 1 1 1 1 1 1
           Gap filled

Threshold: If gap < 300ms, fill it
```

**Why**: Short gaps are often pauses within continuous speech, not true silence.

**Rule 2: Remove Short Bursts (Minimum Duration)**

```
Before: 0 0 0 1 1 0 0 0
               ↑
         Short burst (20ms)

After:  0 0 0 0 0 0 0 0
         Burst removed

Threshold: If speech < 200ms, remove it
```

**Why**: Very short bursts are likely clicks, pops, or artifacts, not actual speech.

**DISPLACE Exception**: In medical conversations, preserve segments down to 150ms to keep backchannels like "mm-hmm."

**Rule 3: Extend Boundaries (Padding)**

```
Before: 0 0 1 1 1 1 0 0

After:  0 1 1 1 1 1 1 0
        ↑           ↑
     +50ms      +50ms

Add padding before and after speech
```

**Why**: Protects speech onsets and offsets from being clipped.

**Complete Smoothing Example**:

```
Raw decisions:
0 0 1 1 1 0 1 1 0 1 0 0 0 1 1 1 1

Step 1: Fill short gaps (< 3 frames)
0 0 1 1 1 1 1 1 0 1 0 0 0 1 1 1 1

Step 2: Remove short bursts (< 2 frames)
0 0 1 1 1 1 1 1 0 0 0 0 0 1 1 1 1

Step 3: Extend boundaries (±1 frame)
0 1 1 1 1 1 1 1 1 0 0 0 1 1 1 1 1 1

Final: Two continuous speech regions
```

**Why Smoothing Matters**:
- Fragmented regions produce unstable embeddings in later modules
- Continuous regions improve clustering quality
- Reduces computational overhead (fewer segments to process)

---

### 1.5.8 Complete Numerical Example

**Scenario**: Process 100ms of audio

**Input**:

```
Duration: 100ms
Sample rate: 16 kHz
Total samples: 1,600
```

**Frame Configuration**:

```
Frame size: 25ms = 400 samples
Hop size: 10ms = 160 samples
Number of frames: 8
```

**Frame-by-Frame Analysis**:

| Frame | Time (ms) | Energy | ZCR  | Raw Decision |
|-------|-----------|--------|------|--------------|
| 1     | 0-25      | 0.001  | 0.15 | 0 (silence)  |
| 2     | 10-35     | 0.002  | 0.18 | 0 (silence)  |
| 3     | 20-45     | 0.015  | 0.22 | 1 (speech)   |
| 4     | 30-55     | 0.018  | 0.20 | 1 (speech)   |
| 5     | 40-65     | 0.020  | 0.19 | 1 (speech)   |
| 6     | 50-75     | 0.003  | 0.25 | 0 (non-speech)|
| 7     | 60-85     | 0.016  | 0.21 | 1 (speech)   |
| 8     | 70-95     | 0.014  | 0.23 | 1 (speech)   |

**Raw Decision Sequence**:

```
0 0 1 1 1 0 1 1
```

**After Smoothing**:

```
Step 1: Fill gap at frame 6 (only 10ms gap)
0 0 1 1 1 1 1 1

Step 2: Extend boundaries
0 1 1 1 1 1 1 1
```

**Final Output**:

```
Speech region: 10ms - 95ms
```

---

### 1.5.9 SAD Architecture Comparison

**Approach 1: Energy-Based SAD (Traditional)**

**Brief Overview**:  
Uses simple acoustic features (energy, zero-crossing rate) with threshold-based decisions. Fast but struggles with noisy environments and low-energy speech. Not recommended for DISPLACE-2026 medical recordings.

**Typical Performance**: 80-85% accuracy in clean conditions, drops significantly in far-field noisy environments.

---

**Approach 2: PyAnnote Neural SAD (Recommended for DISPLACE-2026)**

### Detailed PyAnnote Neural SAD Explanation

**Architecture Overview**

PyAnnote SAD uses a deep neural network architecture specifically designed for voice activity detection in challenging acoustic conditions.

**Network Components**:

```
Input Audio (16 kHz, mono)
        |
        v
Feature Extraction Layer
(Learnable filterbank or fixed spectral features)
        |
        v
Convolutional Neural Network (CNN)
(Captures local temporal-spectral patterns)
        |
        v
Recurrent Neural Network (LSTM/GRU)
(Models long-term temporal dependencies)
        |
        v
Fully Connected Layers
(Maps to speech probability)
        |
        v
Output: Frame-level probabilities [0, 1]
```

**Layer-by-Layer Breakdown**

**1. Feature Extraction**

**What It Does**:  
Converts raw audio waveform into a representation suitable for neural processing.

**Two Approaches**:

**Option A: Learnable Filterbank (SincNet)**
```
Raw waveform → Learnable filters → Feature maps

Advantages:
- Learns optimal frequency bands for SAD
- End-to-end trainable
- Adapts to domain-specific characteristics
```

**Option B: Fixed Spectral Features**
```
Raw waveform → STFT → Log-mel spectrogram → Feature maps

Configuration:
- FFT size: 512 (32ms at 16 kHz)
- Hop size: 160 (10ms at 16 kHz)
- Mel bands: 40-80
- Frequency range: 0-8000 Hz
```

**2. Convolutional Layers (CNN)**

**What They Do**:  
Detect local patterns in time-frequency representation.

**Architecture**:

```
Typical configuration:
- 3-5 convolutional layers
- Kernel sizes: 3x3 or 5x5 (time × frequency)
- Filters: 64-256 per layer
- Activation: ReLU
- Pooling: Max pooling (2x2)
- Batch normalization after each layer
```

**What CNN Learns**:
- Formant patterns (characteristic of speech)
- Harmonic structures (vocal cord vibrations)
- Temporal transitions (speech onsets/offsets)
- Noise vs speech spectral differences

**Example Pattern Detection**:

```
Low-level CNN layers detect:
- Edges in spectrogram (frequency transitions)
- Harmonic bands (speech fundamental frequency)

High-level CNN layers detect:
- Phoneme-like patterns
- Speech vs noise spectral shapes
```

**3. Recurrent Layers (LSTM/GRU)**

**What They Do**:  
Model temporal dependencies and context across frames.

**Architecture**:

```
Typical configuration:
- 2-3 LSTM/GRU layers
- Hidden units: 128-256 per layer
- Bidirectional (processes forward and backward in time)
- Dropout: 0.2-0.3 for regularization
```

**Why Recurrent Layers Matter**:

Speech has temporal structure that CNNs alone cannot fully capture:
- Speech segments last hundreds of milliseconds
- Pauses between words are brief but meaningful
- Context helps distinguish speech from transient noise

**Example**:

```
Frame sequence:
[silence] [silence] [speech] [speech] [speech] [silence]

LSTM learns:
- Isolated high-energy frame → likely noise (reject)
- Sequence of high-energy frames → likely speech (accept)
- Brief low-energy between speech → likely pause (keep as speech)
```

**Bidirectional Processing**:

```
Forward LSTM: Uses past context
Backward LSTM: Uses future context

Combined: Each frame decision uses both past and future

Example:
Frame at t=5:
- Forward LSTM sees: frames 0,1,2,3,4
- Backward LSTM sees: frames 10,9,8,7,6
- Decision uses both contexts
```

**4. Output Layer**

**What It Does**:  
Converts LSTM hidden states to speech probabilities.

**Architecture**:

```
LSTM output (256 dimensions)
        |
        v
Fully connected layer (128 units, ReLU)
        |
        v
Dropout (0.3)
        |
        v
Fully connected layer (1 unit, Sigmoid)
        |
        v
Probability [0, 1]
```

**Interpretation**:

```
Output = 0.95 → Very likely speech
Output = 0.50 → Uncertain
Output = 0.05 → Very likely non-speech

Threshold (typically 0.5):
if probability > 0.5:
    frame = speech
else:
    frame = non-speech
```

---

**Training Process**

**Training Data Requirements**:

```
Labeled audio with frame-level annotations:
- Speech regions marked
- Non-speech regions marked
- Diverse acoustic conditions:
  - Clean speech
  - Noisy speech
  - Far-field recordings
  - Multiple languages
  - Various recording devices
```

**Loss Function**:

PyAnnote SAD typically uses Binary Cross-Entropy (BCE) loss:

```
For each frame:
BCE = -[y × log(p) + (1-y) × log(1-p)]

Where:
- y = ground truth (0 or 1)
- p = predicted probability

Averaged over all frames in training batch
```

**Class Imbalance Handling**:

Speech and non-speech are often imbalanced (more non-speech). Solutions:

```
1. Weighted loss:
   weight_speech = 2.0
   weight_non_speech = 1.0
   
2. Focal loss:
   Focuses on hard-to-classify frames
   
3. Data augmentation:
   Oversample speech regions
```

**Data Augmentation**:

```
Techniques used during training:
- Speed perturbation (0.9x - 1.1x)
- Volume perturbation (±6 dB)
- Noise injection (SNR 5-20 dB)
- Reverberation simulation
- Codec simulation (MP3, Opus)
```

**Training Configuration**:

```
Optimizer: Adam
Learning rate: 0.0001 (with decay)
Batch size: 32-64
Epochs: 50-100
Early stopping: Monitor validation loss
```

---

**Inference Process**

**Step-by-Step**:

```
1. Load audio file (16 kHz, mono)

2. Extract features:
   - Compute log-mel spectrogram
   - Normalize (mean=0, std=1)

3. Forward pass through network:
   - CNN extracts spatial patterns
   - LSTM models temporal context
   - Output layer produces probabilities

4. Apply threshold:
   - Probabilities > 0.5 → speech
   - Probabilities ≤ 0.5 → non-speech

5. Post-processing:
   - Temporal smoothing (fill gaps, remove bursts)
   - Boundary extension (±300ms)
   - Convert to time regions
```

**Computational Details**:

```
Processing speed (on typical hardware):

CPU (Intel i7):
- Feature extraction: 0.3x real-time
- Neural network: 0.2x real-time
- Total: ~0.5x real-time

GPU (NVIDIA RTX 3080):
- Feature extraction: 0.02x real-time
- Neural network: 0.03x real-time
- Total: ~0.05x real-time

Memory usage:
- Model parameters: ~50 MB
- Runtime memory: ~500 MB
- Peak GPU memory: ~2 GB
```

---

**Why PyAnnote SAD**



```
1. Far-field recordings:
   - Low signal-to-noise ratio
   - Reverberation
   - Distance attenuation
   
   PyAnnote handles: Trained on diverse far-field data

2. Low-energy patient speech:
   - Soft-spoken patients
   - Hesitant speech
   - Breathy voice quality
   
   PyAnnote handles: Learns subtle speech patterns

3. Medical environment noise:
   - Equipment beeps
   - Ambient sounds
   - Multiple background voices
   
   PyAnnote handles: Distinguishes speech from complex noise

4. Overlapping speech:
   - Doctor-patient interruptions
   - Simultaneous speech
   
   PyAnnote handles: Detects overlaps without suppressing
```

```

---

**Practical Advantages**:

```
1. Pre-trained models available:
   - No need to train from scratch
   - Hugging Face model hub
   - Multiple domain-specific variants

2. Easy integration:
   - Python API
   - Compatible with PyTorch ecosystem
   - Works with PyAnnote diarization pipeline

3. Active development:
   - Regular updates
   - Community support
   - Used in research and production

4. Proven in competitions:
   - DISPLACE 2024 winners used PyAnnote
   - VoxConverse challenge top systems
   - DIHARD challenge strong baseline
```

---

**Model Variants**

PyAnnote offers several SAD model variants:

```
1. pyannote/voice-activity-detection
   - General-purpose SAD
   - Trained on diverse data
   - Recommended starting point

2. pyannote/segmentation
   - Joint SAD + speaker change detection
   - More complex but more capable
   - Use if speaker boundaries needed

3. Domain-specific fine-tuned models
   - Medical conversation SAD
   - Broadcast news SAD
   - Telephone conversation SAD
```

**Fine-tuning for Medical Domain**:

If development data is available:

```
1. Start with pre-trained PyAnnote SAD
2. Freeze early layers (feature extraction)
3. Fine-tune LSTM and output layers
4. Use medical conversation data
5. Train for 10-20 epochs

Expected improvement: 2-5% absolute F1-score
```

---

**Resource Requirements Summary**:

```
Processing Speed:
- CPU: approximately 0.5x real-time (10s audio → 5s processing)
- GPU: approximately 0.05x real-time (10s audio → 0.5s processing)

Memory:
- RAM: approximately 500 MB
- Model size: approximately 50 MB
- GPU VRAM: approximately 2 GB (if using GPU)

Disk Space:
- Model files: 50 MB
- Dependencies: 500 MB (PyTorch, etc.)

Still practical for competition constraints
```


---


### 1.5.11 Overlap Handling in Module 1

**Critical Rule**:

> SAD must NOT suppress or resolve overlapping speech

**What This Means**:

```
Timeline:
Speaker A: [10.0s ────── 13.5s]
Speaker B:         [12.0s ──── 14.5s]
                      ↑
                   Overlap

SAD Output:
Speech: [10.0s ──────────── 14.5s]

SAD marks entire region as speech without attempting to:
- Identify that two speakers are present
- Separate the overlapping portions
- Label overlap specially
```

**Why**:
- Overlap detection and separation happen in later modules (PixIT/powerset)
- Attempting overlap resolution in SAD can destroy information
- PixIT and powerset models expect continuous speech regions with overlaps intact

**Bad Approach (Avoid)**:

```
Some systems try to suppress overlaps:
Speech: [10.0s - 12.0s]  (only Speaker A)
Speech: [12.0s - 13.5s]  (overlap suppressed)
Speech: [13.5s - 14.5s]  (only Speaker B)

This breaks PixIT's ability to detect and separate overlaps
```

---

### 1.5.12 Boundary Padding Rules

**Purpose**  
Protect speech onsets and offsets from being clipped.

**Padding Configuration**:

```
For each detected speech region:
    extended_start = original_start - 300ms
    extended_end = original_end + 300ms
```

**Why 300ms?**
- Captures soft speech onsets (breathy starts)
- Protects speech offsets (trailing sounds)
- Preserves short acknowledgments near boundaries
- Matches Module 0 padding for consistency

**Example**:

```
Detected speech: [10.000s - 12.000s]

With padding:
Extended speech: [9.700s - 12.300s]
                  ↑              ↑
               -300ms         +300ms
```

**Multiple Regions**:

```
Region 1: [10.0s - 12.0s] → Extended: [9.7s - 12.3s]
Region 2: [13.0s - 15.0s] → Extended: [12.7s - 15.3s]

Regions overlap after padding → Merge:
Final: [9.7s - 15.3s]
```

---

### 1.5.13 What Module 1 MUST NOT Do

**No Speaker Change Detection**  
Do not attempt to identify where one speaker stops and another starts.

**No Segmentation into Turns**  
Do not split speech into speaker turns or utterances.

**No Chunking**  
Do not divide audio into fixed-length windows.

**No Overlap Resolution**  
Do not attempt to separate or suppress overlapping speech.

**No Speaker Count Estimation**  
Do not try to determine how many speakers are present.

**No Speaker Labeling**  
Do not assign speaker identities or labels.

**Why These Restrictions?**  
If SAD attempts to be "smart" and make speaker-level decisions, it interferes with PixIT and powerset models, which are designed to handle these tasks more accurately using joint optimization.

---


## 1.6 SAD Operating Mode 

**High-Recall Mode (Critical)**

SAD must prioritize recall over precision:

| Metric | Target | Reasoning |
|--------|--------|-----------|
| Recall (Sensitivity) | Greater than 95% | Never miss actual speech |
| Precision | Greater than 85% | False positives acceptable |
| Threshold Setting | Low | Err on side of including speech |

**Why High Recall?**

```
False Positive (marking noise as speech):
- Noise gets processed by later modules
- Clustering may create spurious speaker
- Can be corrected with better clustering

False Negative (missing speech):
- Speech is permanently lost
- Cannot be recovered in later modules
- Irrecoverable DER error

Conclusion: False positives are fixable, false negatives are not
```

**Threshold Configuration**:

```
Conservative threshold: 0.005 - 0.01 (lower than typical)
Minimum segment duration: 150ms (preserve backchannels)
Maximum gap to fill: 300ms (merge close speech)
Boundary padding: ±300ms (protect edges)
```

---

## 1.7 Failure Modes to Avoid

| Wrong Behavior | Why It's Bad | Consequence |
|----------------|--------------|-------------|
| Aggressive silence removal | Deletes short speech segments | Missed backchannels, incomplete diarization |
| Energy-based SAD only | Misses low-energy patient speech | False negatives, higher DER |
| Overlap suppression | Breaks PixIT's overlap detection | Cannot handle simultaneous speech |
| Early speaker splits | Creates artificial boundaries | Confuses segmentation models |
| Fixed threshold in noisy environment | Misses speech during noise bursts | Inconsistent performance |

---

## 1.8 Input and Output Specification

### Input

**Source**: Module 0 output

**Format**:
- Sample rate: 16 kHz
- Channels: Mono
- Bit depth: 16-bit PCM
- Normalization: RMS = 0.1
- Duration: Full conversation (no chunking)

**No Labels Used**:
- Speaker labels not used
- Language labels not used
- Overlap annotations not used

SAD operates purely on acoustic signal.

### Output

**Format**: Time-stamped speech regions

**Example**:

```
Speech: 0.50s - 4.20s
Speech: 5.10s - 12.80s
Speech: 14.00s - 18.50s
```

**Properties**:
- Regions may overlap (if original speech overlapped)
- No speaker information attached
- No language information attached
- Regions are continuous (after smoothing)

**Data Structure**:

```
List of tuples: [(start_time, end_time), ...]

Example:
[(0.5, 4.2), (5.1, 12.8), (14.0, 18.5)]

Units: seconds (float)
```

---


## 1.10 Processing Pipeline Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: STANDARDIZED AUDIO                    │
│                                                                 │
│  Format: 16 kHz, mono, 16-bit PCM, RMS=0.1                     │
│  Duration: Full conversation (e.g., 180 seconds)                │
│  Source: Module 0 output                                        │
└─────────────────────────────────────────────────────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────────┐
│                    STEP 1: FRAME AUDIO                          │
│                                                                 │
│  Frame size: 25ms (400 samples)                                 │
│  Hop size: 10ms (160 samples)                                   │
│  Overlap: 60%                                                   │
│  Number of frames: ~18,000 for 180s audio                       │
└─────────────────────────────────────────────────────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: EXTRACT FEATURES (per frame)               │
│                                                                 │
│  Energy: Sum of squared amplitudes                              │
│  ZCR: Zero-crossing rate                                        │
│  Optional: Spectral features, MFCCs                             │
└─────────────────────────────────────────────────────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: CLASSIFY (per frame)                       │
│                                                                 │
│  Method: PyAnnote Neural SAD (recommended)                      │
│  Alternative: Threshold-based (energy > 0.01)                   │
│  Output: Binary decisions (0 or 1) per frame                    │
│  Mode: High-recall (threshold set low)                          │
└─────────────────────────────────────────────────────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: TEMPORAL SMOOTHING                         │
│                                                                 │
│  4a. Fill short gaps (< 300ms)                                  │
│  4b. Remove short bursts (< 150ms, except backchannels)         │
│  4c. Extend boundaries (±300ms padding)                         │
│  4d. Merge overlapping regions                                  │
└─────────────────────────────────────────────────────────────────┘
                              |
                              v
┌─────────────────────────────────────────────────────────────────┐
│                    OUTPUT: SPEECH REGIONS                       │
│                                                                 │
│  Format: List of (start, end) tuples in seconds                 │
│  Example: [(0.5, 4.2), (5.1, 12.8), (14.0, 18.5)]              │
│  Properties: Continuous, overlap-preserved, no speaker info     │
│                                                                 │
│  Next: PixIT joint diarization-separation (Module 2+)           │
└─────────────────────────────────────────────────────────────────┘
```

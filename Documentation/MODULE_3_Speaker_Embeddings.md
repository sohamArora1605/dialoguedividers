# MODULE 3 — LANGUAGE-AGNOSTIC SPEAKER EMBEDDING EXTRACTION

*(DISPLACE-2026 compatible, structure-aware, LASPA-trained, overlap-aware)*

---

## 3.1 What This Module Does

**Purpose (2-line summary)**  
Extracts language-agnostic speaker identity embeddings from speech using speaker-activity structure from Module 2, avoiding overlap corruption and ensuring embeddings remain stable across language switches and code-mixing.

**Input → Output**
- **Input**: Audio (from Module 0) + Speaker-activity structure (from Module 2)
- **Output**: Speaker embeddings with confidence scores and temporal alignment

---

## 3.2 Why Module 3 Exists (Fundamental Concept)

**The Core Diarization Question**:

> "Which speech segments belong to the same speaker?"

This question is answered **only** by:
1. **Speaker embeddings** (representations of speaker identity)
2. **Similarity measurement** (comparing embeddings)

**If embeddings are corrupted, nothing downstream can fix it**:

| Embedding Problem | Consequence | Can Clustering Fix It? |
|-------------------|-------------|------------------------|
| Language leakage | Same speaker, different language → different clusters | ❌ No |
| Overlap corruption | Mixed speakers → ambiguous embeddings | ❌ No |
| Short-segment instability | Same speaker → inconsistent embeddings | ❌ No |

**Module 3's responsibility**: Produce **reliable, language-agnostic embeddings** that clustering can actually use.

---

## 3.3 What Module 3 Receives (Strict Input Specification)

**Module 3 does NOT guess or infer anything**. It consumes outputs from earlier modules.

### 3.3.1 Inputs (Inference Time)

**1. Full Conversation Audio** (from Module 0):
```
Format: 16 kHz, mono, 16-bit PCM, RMS=0.1
Duration: Full conversation (e.g., 178 seconds)
```

**2. Speaker-Activity Structure** (from Module 2):
```
Format: Frame-level activity counts
Example: [(0.0, 12.3, 1), (12.3, 12.5, 2), (12.5, 15.0, 1)]

Where:
- (start, end, num_active)
- num_active = 1: Single speaker
- num_active = 2: Overlap (2 speakers)
```

### 3.3.2 Additional Inputs (Training Time Only)

**3. Speaker Labels**:
```
Ground truth speaker identities for training
Example: "Speaker_A", "Speaker_B"
```

**4. Language Labels**:
```
Ground truth language per segment for training
Example: "English", "Hindi", "Code-mixed"
```

**Critical**: Module 3 does **NOT** perform:
- ❌ Speech activity detection (Module 1's job)
- ❌ Segmentation (Module 2's job)
- ❌ Language identification (labels provided in training data)

---

## 3.4 First Decision: Branching by Speaker Activity (CRITICAL)

**Module 3 immediately branches based on speaker-activity structure from Module 2.**

### Two Disjoint Processing Paths

```
Speaker-Activity Input
        ↓
    [Decision]
        ↓
   ┌────┴────┐
   ↓         ↓
Single    Overlap
Speaker   Regions
   ↓         ↓
Path 1    Path 2
```

**This separation is mandatory** because:
- Single-speaker regions: Clean, reliable, high-confidence embeddings
- Overlap regions: Require special handling to avoid corruption

---

## 3.5 Path 1 — Single-Speaker Regions (Primary Source)

### 3.5.1 What Qualifies as Single-Speaker

**Frames where Module 2 indicates exactly one active speaker**:

```
Example from Module 2 output:
  [(0.0, 12.3, 1),    ← Single speaker (Path 1)
   (12.3, 12.5, 2),   ← Overlap (Path 2)
   (12.5, 15.0, 1)]   ← Single speaker (Path 1)
```

**Characteristics**:
- Clean speech (no overlap)
- Reliable for identity learning
- Best source for high-confidence embeddings

### 3.5.2 Processing Steps (Single-Speaker)

**Step 1: Group Contiguous Frames**

```
Input: [(0.0, 12.3, 1), (12.5, 15.0, 1), (15.2, 18.5, 1), ...]

Group consecutive single-speaker frames:
  Segment 1: [0.0-12.3s] (12.3s duration)
  Segment 2: [12.5-15.0s] (2.5s duration)
  Segment 3: [15.2-18.5s] (3.3s duration)
```

**Step 2: Form Embedding Segments**

**Segment Duration Constraints**:
- **Minimum**: 250ms (preserve backchannels)
- **Preferred**: 0.5-2.0 seconds
- **Maximum**: 3.0 seconds (split longer segments)

**Why these constraints?**

*Simple*: Too short = unstable embeddings. Too long = wastes computation.

*Technical*:
- <250ms: Insufficient phonetic content for reliable speaker representation
- 0.5-2.0s: Optimal balance between stability and granularity
- >3.0s: Diminishing returns, better to split for finer temporal resolution

**Example**:

```
Segment 1: [0.0-12.3s] (12.3s) → Split into:
  [0.0-2.0s], [2.0-4.0s], [4.0-6.0s], [6.0-8.0s], [8.0-10.0s], [10.0-12.0s], [12.0-12.3s]
  
  Note: Last segment [12.0-12.3s] = 300ms (above 250ms minimum) ✓

Segment 2: [12.5-15.0s] (2.5s) → Split into:
  [12.5-14.5s], [14.5-15.0s]
  
  Note: Last segment [14.5-15.0s] = 500ms ✓

Segment 3: [15.2-18.5s] (3.3s) → Split into:
  [15.2-17.2s], [17.2-18.5s]
  
  Note: Last segment [17.2-18.5s] = 1.3s ✓
```

**Step 3: Extract Embeddings**

```
For each segment:
  1. Load audio [start:end]
  2. Forward pass through embedding network
  3. Obtain embedding vector (192-256 dimensions)
  4. L2 normalize
  5. Mark as high-confidence
```

**Output Format**:

```python
{
  "start_time": 0.0,
  "end_time": 2.0,
  "embedding_vector": [0.12, -0.34, 0.56, ...],  # 256-dim
  "confidence": "high",
  "source": "single_speaker"
}
```

---

## 3.6 Path 2 — Overlapped Regions (Handled Carefully)

### 3.6.1 The Overlap Problem

**Why overlapped speech is dangerous for embeddings**:

```
Overlapped audio: Speaker A + Speaker B mixed

Naive embedding extraction:
  embedding = extract(mixed_audio)
  
Result: Embedding represents BOTH speakers ❌
  - Not A's identity
  - Not B's identity
  - Ambiguous mixture that confuses clustering
```

**Numerical Example**:

```
Pure Speaker A embedding: [0.8, 0.2, 0.1, ...]
Pure Speaker B embedding: [0.1, 0.7, 0.3, ...]

Overlap mixture embedding: [0.45, 0.45, 0.2, ...]
  ← Corrupted! Neither A nor B

Cosine similarity:
  sim(A_pure, A_overlap) = 0.65 (should be ~0.95)
  sim(B_pure, B_overlap) = 0.58 (should be ~0.95)
  
Both speakers corrupted ✗
```

### 3.6.2 Module 3 NEVER Extracts from Raw Overlaps

**Forbidden**:

```python
# ❌ WRONG
if num_active == 2:
    embedding = extract(raw_audio)  # NEVER DO THIS
```

**This is the #1 cause of diarization errors in naive systems.**

### 3.6.3 Two Valid Strategies for Overlaps

**Strategy A: PixIT-Separated Sources (Preferred)**

If PixIT (Module 2) provides separated audio streams:

```
PixIT output:
  separated_stream_A: Audio with Speaker A isolated
  separated_stream_B: Audio with Speaker B isolated

Module 3 processing:
  embedding_A = extract(separated_stream_A)
  embedding_B = extract(separated_stream_B)
  
  Mark both as "medium" confidence
  (separation may introduce artifacts)
```

**Why medium confidence?**
- PixIT separation is good but not perfect
- Some residual cross-talk may remain
- Artifacts from separation process

**Strategy B: Overlap Exclusion (Fallback)**

If separation is unreliable or unavailable:

```python
if num_active == 2:
    # Do NOT extract embeddings
    # Skip this region
    # Overlap will be resolved later during assignment
```

**This is exactly how DISPLACE-2024 winning systems avoided embedding corruption.**

**Overlap resolution happens in Module 4/5**:
- Use embeddings from single-speaker regions
- Assign overlap frames based on similarity to known speakers
- Or use VBx to handle overlaps probabilistically

---

## 3.7 What is Explicitly FORBIDDEN in Module 3

**Module 3 must NOT perform**:

❌ **Embeddings from mixed overlap**: Never extract from raw overlapped audio  
❌ **Embeddings before segmentation**: Module 2 must provide structure first  
❌ **Energy-based slicing**: No heuristic segmentation  
❌ **Speaker change heuristics**: No BIC, GLR, or manual boundary detection  
❌ **Any clustering logic**: Module 3 represents speakers, does not group them  
❌ **Similarity scoring**: That's Module 4's job  
❌ **Speaker counting**: That's Module 4's job  
❌ **RTTM generation**: That's Module 5's job

**Module 3 is pure representation learning** - it creates speaker embeddings, nothing more.

---

## 3.8 Embedding Architecture (ECAPA-TDNN)

### 3.8.1 Why ECAPA-TDNN as Primary Architecture

**ECAPA-TDNN (Emphasized Channel Attention, Propagation and Aggregation in TDNN)** advantages:

1. **State-of-the-art speaker recognition**: Outperforms x-vectors and i-vectors
2. **Efficient**: Lighter than WavLM, faster inference
3. **Proven for diarization**: Used in winning DISPLACE systems
4. **Channel attention**: Focuses on speaker-discriminative features
5. **Multi-scale temporal context**: Captures both short and long-range patterns

**Performance Comparison**:

```
Method          | EER (%) | DER (%)
----------------|---------|--------
i-vector        | 12.5    | 18.3
x-vector        | 8.2     | 12.1
ECAPA-TDNN      | 6.1     | 9.4
ResNet-based    | 7.3     | 10.8

Metrics: Equal Error Rate (EER), Diarization Error Rate (DER)
Lower is better
```

**ECAPA-TDNN is optimal** for DISPLACE-2026 medical conversations.

**But**: ECAPA-TDNN alone is **NOT language-agnostic** ⚠️

### 3.8.2 ECAPA-TDNN Architecture

**Full pipeline**:

```
Input: Audio waveform (16 kHz)
    ↓
Feature Extraction (80-dim log-mel filterbank)
    ↓
Frame-level Encoder (Conv1D layers)
    ↓
SE-Res2Block 1 (Multi-scale temporal context)
    ↓
SE-Res2Block 2 (Multi-scale temporal context)
    ↓
SE-Res2Block 3 (Multi-scale temporal context)
    ↓
Channel Attention (Squeeze-and-Excitation)
    ↓
Attentive Statistics Pooling (mean + std, weighted)
    ↓
Bottleneck Layer (1536 → 192)
    ↓
Final Embedding Layer (192 → 256)
    ↓
L2 Normalization
    ↓
Output: Speaker Embedding (256-dim, unit norm)
```

### 3.8.3 Component Details

**Feature Extraction**:
```
Input: Raw audio (16 kHz)
Window: 25ms
Hop: 10ms
Mel filterbanks: 80
Output: (T, 80) where T = number of frames
```

**Frame-level Encoder**:
```
Conv1D (80 → 512 channels)
ReLU activation
Layer normalization
```

**SE-Res2Block** (Squeeze-and-Excitation Res2Net Block):
```
Multi-scale convolutions:
  - Scale 1: 3×1 convolution
  - Scale 2: 3×1 convolution (on scale 1 output)
  - Scale 3: 3×1 convolution (on scale 2 output)
  - Scale 4: 3×1 convolution (on scale 3 output)

Concatenate all scales
Apply Squeeze-and-Excitation:
  - Global average pooling
  - FC layer (channel reduction)
  - ReLU
  - FC layer (channel expansion)
  - Sigmoid
  - Channel-wise multiplication

Residual connection
```

**Why SE-Res2Block?**
- **Multi-scale**: Captures both short (phonemes) and long (prosody) patterns
- **Channel attention**: Emphasizes speaker-discriminative channels
- **Residual**: Enables deep networks without degradation

**Attentive Statistics Pooling**:
```
For each channel c:
  # Compute attention weights
  attention = softmax(FC(frame_features))
  
  # Weighted mean
  mean_c = Σ(attention[t] × frame[t, c])
  
  # Weighted std
  std_c = √(Σ(attention[t] × (frame[t, c] - mean_c)²))

Concatenate: [mean, std] across all channels
Output: 1536-dim (768 mean + 768 std)
```

**Why attentive pooling?**
- Not all frames equally important for speaker identity
- Attention focuses on speaker-discriminative regions
- Better than simple mean/std pooling

**Bottleneck Layer**:
```
Input: 1536-dim
Linear: 1536 → 192
Batch normalization
ReLU activation
```

**Final Embedding Layer**:
```
Input: 192-dim
Linear: 192 → 256
Output: 256-dim embedding
```

**L2 Normalization**:
```
embedding_normalized = embedding / ||embedding||₂

Result: Unit-norm vector (cosine similarity ready)
```

### 3.8.4 Embedding Dimension

**Choice: 256 dimensions**

**Why not higher?**
- 512-dim: Overfitting risk on limited medical data
- 1024-dim: Excessive for 2-10 speakers

**Why not lower?**
- 128-dim: Insufficient capacity for language-agnostic representation
- 64-dim: Too compressed, loses speaker nuances

**256-dim is optimal** for DISPLACE-2026 medical conversations.

---

## 3.9 Core of Module 3 — LASPA (Language-Agnostic Speaker Prefix Adaptation)

### 3.9.1 Why LASPA is REQUIRED for DISPLACE-2026

**Medical Conversation Challenges**:

```
Doctor (same person):
  [0-10s]: "How long have you been feeling this way?" (English)
  [15-25s]: "Aapko bukhar kitne din se hai?" (Hindi)
  [30-40s]: "Any chest pain or difficulty breathing?" (English)

Without language-agnostic training:
  Embedding[0-10s] ≠ Embedding[15-25s] ❌
  
  Same speaker, different clusters!
```

**Code-Mixing Example**:

```
Patient: "Doctor, mujhe headache hai and also fever"
         (Hindi)  (English)    (Hindi)  (English)

Phonetic shifts mid-sentence → embedding instability
```

**The Problem**:

> Standard speaker encoders learn **speaker + language** jointly.  
> Same speaker in different language → different embedding.

### 3.9.2 What LASPA Is (Precise Definition)

**LASPA = Language-Agnostic Speaker Prefix Adaptation**

**It is a training-time mechanism that**:
1. **Contains** language information in a separate pathway
2. **Prevents** language from leaking into speaker embeddings
3. **Stabilizes** embeddings across language switches
4. **Avoids** adversarial instability (unlike GRL)

**Key Insight**: Instead of forcing the encoder to "forget" language (adversarial), we give language information a **dedicated pathway** so it doesn't contaminate speaker embeddings.

### 3.9.3 How LASPA Works (Step-by-Step)

**Step 1: Language Labels Already Exist**

```
Training data includes:
  - Audio
  - Speaker labels
  - Language labels (per segment)

Example:
  Segment 1: Speaker=A, Language=English
  Segment 2: Speaker=A, Language=Hindi
  Segment 3: Speaker=B, Language=English
```

**No language identification inference needed** - labels are provided.

**Step 2: Language-Conditioned Feature Augmentation**

**Architecture**:

```
Language Label (e.g., "Hindi")
    ↓
Language Embedding Lookup (learnable)
    ↓
Language Vector (32-64 dim)
    ↓
    ├─→ Concatenated with mel-filterbank features
    │
Audio → Mel-filterbank (80-dim) + Language Vector (64-dim) → ECAPA-TDNN → Speaker Embedding
         Combined: 144-dim input features
```

**How Language Augmentation Works**:

```
Standard ECAPA-TDNN input:
  [mel_feature_1, mel_feature_2, ..., mel_feature_80]  (per frame)

LASPA ECAPA-TDNN input:
  [mel_feature_1, ..., mel_feature_80, lang_emb_1, ..., lang_emb_64]  (per frame)
  
Language embedding (64-dim) concatenated to each frame's mel features
```

**Effect**:
- Language variation is **absorbed by the language embedding**
- Speaker embedding pathway remains **language-clean**

**Step 3: Training Objectives**

**Multi-task loss**:

```
Total Loss = α × L_speaker + β × L_contrastive + γ × L_LASPA

Where:
  L_speaker: Speaker classification loss (cross-entropy)
  L_contrastive: Contrastive speaker loss (same/different speaker pairs)
  L_LASPA: LASPA regularization (language disentanglement)
  
  α, β, γ: Loss weights (e.g., 1.0, 0.5, 0.3)
```

**Speaker Classification Loss**:

```
L_speaker = -Σ y_s × log(p_s)

Where:
  y_s = one-hot speaker label
  p_s = predicted speaker probability
```

**Contrastive Speaker Loss**:

```
For each embedding pair (e_i, e_j):

  If same speaker:
    L_pos = -log(sim(e_i, e_j))
    
  If different speaker:
    L_neg = -log(1 - sim(e_i, e_j))
    
  L_contrastive = L_pos + L_neg
  
Where:
  sim(e_i, e_j) = cosine_similarity(e_i, e_j)
```

**LASPA Regularization**:

```
L_LASPA = ||E_speaker · E_language^T||_F

Where:
  E_speaker = speaker embedding matrix
  E_language = language embedding matrix
  ||·||_F = Frobenius norm
  
Goal: Minimize correlation between speaker and language representations
```

---

## 3.10 Contrastive Constraints (Critical for Language-Agnostic Learning)

**Explicitly enforced during training**:

### Constraint 1: Same Speaker, Different Language

```
Speaker A, English: embedding_A_en
Speaker A, Hindi: embedding_A_hi

Enforce: cosine_similarity(embedding_A_en, embedding_A_hi) > 0.9

Training pairs:
  (A_en, A_hi) → positive pair (high similarity)
```

### Constraint 2: Different Speaker, Same Language

```
Speaker A, English: embedding_A_en
Speaker B, English: embedding_B_en

Enforce: cosine_similarity(embedding_A_en, embedding_B_en) < 0.3

Training pairs:
  (A_en, B_en) → negative pair (low similarity)
```

### Constraint 3: Same Speaker, Code-Mixed

```
Speaker A, Code-mixed segment 1: embedding_A_cm1
Speaker A, Code-mixed segment 2: embedding_A_cm2

Enforce: cosine_similarity(embedding_A_cm1, embedding_A_cm2) > 0.9

Training pairs:
  (A_cm1, A_cm2) → positive pair (tight cluster)
```

**Numerical Example**:

```
Training batch:
  Sample 1: Speaker=A, Language=English
  Sample 2: Speaker=A, Language=Hindi
  Sample 3: Speaker=B, Language=English
  Sample 4: Speaker=B, Language=Hindi

Similarity matrix (target):
       A_en  A_hi  B_en  B_hi
A_en   1.00  0.95  0.20  0.18
A_hi   0.95  1.00  0.22  0.19
B_en   0.20  0.22  1.00  0.94
B_hi   0.18  0.19  0.94  1.00

Diagonal blocks (same speaker): High similarity (>0.9)
Off-diagonal blocks (different speaker): Low similarity (<0.3)
```

**This is what makes embeddings stable across language switches.**

---

## 3.11 Inference Behavior (Important)

**At inference time (during actual diarization)**:

❌ **No language prefix**  
❌ **No language head**  
❌ **No LASPA computation**  
❌ **No language labels needed**

**Just**:

```
Audio → Mel-filterbank → ECAPA-TDNN → Speaker Embedding
```

**Why this works**:

LASPA has already **shaped the embedding space** during training:
- Same speaker, any language → similar embeddings
- Different speakers → dissimilar embeddings

**The model has learned to ignore language** - no runtime language information needed.

**Inference Example**:

```
Input: Unknown speaker, unknown language
Process: Extract embedding (256-dim vector)
Output: Language-agnostic speaker representation

No language identification required ✓
```

---

## 3.12 Output of Module 3 (Final Specification)

**Module 3 outputs embedding objects, not just vectors.**

### Output Format

```python
[
  {
    "start_time": 0.0,
    "end_time": 2.0,
    "embedding_vector": [0.12, -0.34, 0.56, ..., 0.23],  # 256-dim, L2-normalized
    "confidence": "high",
    "source": "single_speaker",
    "duration": 2.0
  },
  {
    "start_time": 2.0,
    "end_time": 4.0,
    "embedding_vector": [0.15, -0.28, 0.61, ..., 0.19],
    "confidence": "high",
    "source": "single_speaker",
    "duration": 2.0
  },
  {
    "start_time": 12.3,
    "end_time": 12.5,
    "embedding_vector": [0.18, -0.31, 0.58, ..., 0.21],
    "confidence": "medium",
    "source": "pixit_separated",
    "duration": 0.2
  },
  ...
]
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `start_time` | float | Segment start (seconds) |
| `end_time` | float | Segment end (seconds) |
| `embedding_vector` | list[float] | 256-dim L2-normalized embedding |
| `confidence` | string | "high" (single-speaker) or "medium" (PixIT-separated) |
| `source` | string | "single_speaker" or "pixit_separated" |
| `duration` | float | Segment duration (seconds) |

**No speaker ID yet** - that's determined by clustering (Module 4).

---

## 3.13 What Module 3 Does NOT Do

**Module 3 is pure representation learning** - it does NOT perform:

❌ **Similarity scoring**: Comparing embeddings (Module 4)  
❌ **Clustering**: Grouping embeddings into speakers (Module 4)  
❌ **VBx**: Variational Bayes diarization (Module 4)  
❌ **Speaker counting**: Determining number of unique speakers (Module 4)  
❌ **Speaker labeling**: Assigning "Speaker 1", "Speaker 2" (Module 4)  
❌ **RTTM generation**: Creating final output format (Module 5)  
❌ **Language identification**: Not needed (LASPA handles it)  
❌ **Overlap resolution**: Assigning overlap frames to speakers (Module 4/5)

**Module 3 answers**: "What is this speaker's identity representation?"  
**Module 3 does NOT answer**: "Which speaker is this?" or "How many speakers are there?"

---

## 3.14 Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Audio + Speaker-Activity Structure                      │
│ Audio: 16 kHz, mono, 178s                                      │
│ Structure: [(0.0, 12.3, 1), (12.3, 12.5, 2), (12.5, 15.0, 1)]│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: BRANCH BY SPEAKER ACTIVITY                 │
│ Separate single-speaker regions from overlaps                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│ PATH 1: Single-Speaker   │  │ PATH 2: Overlap Regions  │
│                          │  │                          │
│ 1. Group contiguous      │  │ 1. Check PixIT available │
│ 2. Form segments         │  │ 2a. If yes: Use separated│
│    (0.5-2.0s preferred)  │  │     streams              │
│ 3. Extract embeddings    │  │ 2b. If no: Skip region   │
│ 4. Mark high-confidence  │  │ 3. Mark medium-confidence│
└──────────────────────────┘  └──────────────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: EMBEDDING EXTRACTION                       │
│ Mel-filterbank → ECAPA-TDNN → L2 Normalize                     │
│ (LASPA-trained, language-agnostic)                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Embedding Objects                                      │
│ [{start, end, embedding, confidence, source}, ...]             │
│ Next: MODULE 4 (Similarity Scoring & Clustering)               │
└─────────────────────────────────────────────────────────────────┘
```

---

## 3.15 Complete End-to-End Example

**Scenario**: Extracting embeddings from 15-second medical excerpt

### Initial State

```
Input from Module 2:
  Speaker-activity structure:
    [(0.0, 12.3, 1),    # Single speaker
     (12.3, 12.5, 2),   # Overlap
     (12.5, 15.0, 1)]   # Single speaker

Audio: 16 kHz, mono, 15 seconds
```

---

### Step 1: Branch by Speaker Activity

**Single-speaker regions**:
```
Region 1: [0.0-12.3s] (12.3s duration)
Region 2: [12.5-15.0s] (2.5s duration)
```

**Overlap regions**:
```
Region 3: [12.3-12.5s] (0.2s duration)
```

---

### Step 2: Process Single-Speaker Regions (Path 1)

**Region 1: [0.0-12.3s]**

**Form segments** (2.0s preferred):
```
Segment 1: [0.0-2.0s]
Segment 2: [2.0-4.0s]
Segment 3: [4.0-6.0s]
Segment 4: [6.0-8.0s]
Segment 5: [8.0-10.0s]
Segment 6: [10.0-12.0s]
Segment 7: [12.0-12.3s] (300ms, above 250ms minimum ✓)
```

**Extract embeddings**:

```
For Segment 1 [0.0-2.0s]:
  1. Load audio samples [0:32,000] (2.0s × 16,000 Hz)
  2. Forward pass:
     Audio → Mel-filterbank → ECAPA-TDNN
  3. Output: 256-dim vector
     [0.123, -0.342, 0.567, ..., 0.234]
  4. L2 normalize:
     norm = √(Σ(x_i²)) = 1.0
  5. Mark: confidence="high", source="single_speaker"

For Segment 2 [2.0-4.0s]:
  ... (same process)
  Output: [0.118, -0.338, 0.571, ..., 0.229]
  
... (repeat for all 7 segments)
```

**Region 2: [12.5-15.0s]**

**Form segments**:
```
Segment 8: [12.5-14.5s] (2.0s)
Segment 9: [14.5-15.0s] (0.5s, above 250ms minimum ✓)
```

**Extract embeddings** (same process as Region 1).

---

### Step 3: Process Overlap Region (Path 2)

**Region 3: [12.3-12.5s] (overlap)**

**Check PixIT availability**:

**Scenario A: PixIT separation available**

```
PixIT output:
  separated_stream_A: Audio with Speaker A isolated
  separated_stream_B: Audio with Speaker B isolated

Extract embeddings:
  embedding_A = extract(separated_stream_A)
  embedding_B = extract(separated_stream_B)
  
Output:
  {
    "start_time": 12.3,
    "end_time": 12.5,
    "embedding_vector": [0.125, -0.340, 0.565, ...],
    "confidence": "medium",
    "source": "pixit_separated"
  },
  {
    "start_time": 12.3,
    "end_time": 12.5,
    "embedding_vector": [0.089, -0.412, 0.523, ...],
    "confidence": "medium",
    "source": "pixit_separated"
  }
```

**Scenario B: PixIT separation unavailable/unreliable**

```
Skip this region (no embeddings extracted)

Overlap will be resolved in Module 4/5 using:
  - Embeddings from single-speaker regions
  - Similarity-based assignment
```

---

### Final Output

**Total embeddings extracted**: 9 (7 from Region 1, 2 from Region 2)

```python
embeddings = [
  # Region 1: [0.0-12.3s]
  {"start_time": 0.0, "end_time": 2.0, "embedding_vector": [0.123, ...], "confidence": "high", "source": "single_speaker"},
  {"start_time": 2.0, "end_time": 4.0, "embedding_vector": [0.118, ...], "confidence": "high", "source": "single_speaker"},
  {"start_time": 4.0, "end_time": 6.0, "embedding_vector": [0.121, ...], "confidence": "high", "source": "single_speaker"},
  {"start_time": 6.0, "end_time": 8.0, "embedding_vector": [0.119, ...], "confidence": "high", "source": "single_speaker"},
  {"start_time": 8.0, "end_time": 10.0, "embedding_vector": [0.124, ...], "confidence": "high", "source": "single_speaker"},
  {"start_time": 10.0, "end_time": 12.0, "embedding_vector": [0.122, ...], "confidence": "high", "source": "single_speaker"},
  {"start_time": 12.0, "end_time": 12.3, "embedding_vector": [0.120, ...], "confidence": "high", "source": "single_speaker"},
  
  # Region 2: [12.5-15.0s]
  {"start_time": 12.5, "end_time": 14.5, "embedding_vector": [0.089, ...], "confidence": "high", "source": "single_speaker"},
  {"start_time": 14.5, "end_time": 15.0, "embedding_vector": [0.091, ...], "confidence": "high", "source": "single_speaker"},
  
  # Region 3: [12.3-12.5s] (overlap) - skipped in this example
]
```

**Quality checks**:
```
✓ All single-speaker regions processed
✓ All segments ≥250ms
✓ Embeddings L2-normalized (unit norm)
✓ Confidence levels assigned
✓ No embeddings from raw overlap
✓ Ready for clustering (Module 4)
```

---

## 3.16 Key Terminology

| Term | Simple | Technical | Why It Matters |
|------|--------|-----------|----------------|
| **Embedding** | Speaker fingerprint | High-dimensional vector representing speaker identity | Core representation for clustering |
| **LASPA** | Language-agnostic training | Language-Agnostic Speaker Prefix Adaptation | Prevents language leakage into embeddings |
| **ECAPA-TDNN** | Speaker embedding model | Emphasized Channel Attention, Propagation and Aggregation in TDNN | State-of-the-art speaker encoder |
| **Overlap corruption** | Mixed speaker problem | Embeddings from overlapped speech are ambiguous | Must avoid or use separated sources |
| **Confidence** | Reliability score | High (single-speaker) or medium (separated) | Guides clustering weight |
| **L2 normalization** | Unit vector | Scale to length 1 | Enables cosine similarity |
| **Contrastive loss** | Same/different pairs | Pulls same speaker together, pushes different apart | Improves embedding quality |
| **Statistics pooling** | Temporal aggregation | Mean + std over time | Converts variable-length to fixed-size |

---

## 3.17 Success Criteria

Module 3 is successful if:

1. **Language-agnostic**: Same speaker, different language → similarity >0.9
2. **Speaker-discriminative**: Different speakers → similarity <0.3
3. **Overlap-safe**: No embeddings from raw overlaps
4. **Segment coverage**: ≥90% of single-speaker regions have embeddings
5. **Minimum duration**: All segments ≥250ms
6. **Normalized**: All embeddings are unit-norm (L2=1.0)
7. **Confidence-tagged**: All embeddings have confidence scores
8. **No clustering**: Module 3 does NOT group embeddings

---

## 3.18 Summary

> **Module 3 extracts language-agnostic speaker embeddings using LASPA-trained ECAPA-TDNN encoder, processing single-speaker regions with high confidence and handling overlaps via PixIT separation or exclusion, producing reliable representations for downstream clustering without performing any grouping decisions.**

---

**Next Module**: MODULE 4 — Similarity Scoring & Clustering (LASPA-aware, overlap-aware)

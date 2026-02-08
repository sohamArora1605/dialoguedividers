# MODULE 2 — NEURAL SEGMENTATION (SUMMARY)

## What We Do
Convert coarse speech regions into fine-grained frame-level speaker-activity representations (single-speaker and overlap frames) using neural methods, without assigning speaker identities.

## Input and Output
- **Input**: Speech regions from Module 1 (time-stamped segments, no speaker labels)
- **Output**: Frame-level speaker activity (single vs overlap, 10-20ms granularity)

---

## Core Design Change

### ❌ Removed (Classical SCD)
- Energy-based segmentation
- BIC/GLR speaker change detection
- Heuristic turn splitting
- Fixed-window approaches

### ✅ Replaced With (Neural)
- PixIT joint segmentation (primary)
- Powerset neural segmentation (secondary)

**Why**: Medical conversations have rapid turns, interruptions, overlaps. Classical SCD fails because it assumes abrupt changes, single speakers, and clear silence boundaries.

---

## Two Parallel Paths

### Path A: PixIT Joint Segmentation (Primary)
**What**: End-to-end learned segmentation + separation  
**How**: Neural network predicts per-speaker activity masks  
**Output**: Continuous probabilities (0-1) per speaker per frame

### Path B: Powerset Segmentation (Secondary)
**What**: Multi-class classification of speaker combinations  
**How**: Treats {A}, {B}, {A+B} as distinct classes  
**Output**: Class probabilities per frame

---

## Processing Steps

### 1. Feature Extraction
**What**: Extract audio features for neural processing  
**How**:
- STFT or log-mel spectrogram
- Frame size: 20ms, Hop: 10ms
- Normalize features

**Why**: Neural networks need consistent input representations  
*Example: 5-second audio → 500 frames of 80-dimensional mel features*

---

### 2. PixIT Processing
**What**: Joint segmentation and separation  
**How**:
- Encoder: CNN or Transformer
- Separation network: LSTM/Transformer
- Decoder: Per-speaker mask estimation
- Activity extraction: Threshold masks

**Why**: Learns overlap patterns from data instead of rules  
*Example: Overlap [12.3-12.5s] detected with 0.88 and 0.85 mask values for two speakers*

---

### 3. Powerset Processing
**What**: Classify each frame into speaker combination  
**How**:
- Feature extraction: CNN
- Temporal modeling: LSTM
- Classification: Softmax over classes
- Classes: {∅, {A}, {B}, {A+B}} for 2 speakers

**Why**: Explicit overlap modeling as distinct class  
*Example: Frame classified as {A+B} with 0.49 probability*

---

### 4. Post-Processing
**What**: Convert predictions to time segments  
**How**:
- Threshold PixIT masks (>0.5)
- Argmax powerset classes
- Optional: Ensemble both methods
- Merge consecutive frames

**Why**: Produces clean segmentation boundaries  
*Example: Frames 0-230 (single) + 230-250 (overlap) + 250-500 (single)*

---

## Complete Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Speech Regions from Module 1                            │
│ [(0.0, 4.8), (4.5, 9.8), (9.5, 14.2), ...]                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              PREPROCESSING: Extract Features                    │
│ STFT or log-mel spectrogram, normalize                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│ PATH A: PixIT            │  │ PATH B: Powerset         │
│ Encoder → Separation     │  │ CNN → LSTM → Softmax     │
│ → Masks → Activity       │  │ → Class probabilities    │
└──────────────────────────┘  └──────────────────────────┘
                    │                   │
                    └─────────┬─────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              POST-PROCESSING: Threshold & Segment               │
│ Threshold, argmax, ensemble, convert to time segments          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Frame-level Speaker Activity                           │
│ [(0.0, 12.3, 1), (12.3, 12.5, 2), (12.5, 15.0, 1)]           │
│ Next: MODULE 3 (Speaker Embedding Extraction)                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete End-to-End Example

**Scenario**: Segmenting 5-second medical excerpt

### Initial State
```
Input: Speech region [10.0-15.0s]
Ground truth:
  [10.0-12.3s]: Doctor
  [12.3-12.5s]: Overlap (doctor + patient)
  [12.5-15.0s]: Patient

Frames: 500 (20ms frames, 10ms hop)
```

---

### PixIT Processing

**Features**: STFT spectrogram (257 freq bins, 500 frames)

**Masks** (sample frames):
```
Frame | Time (s) | Speaker A | Speaker B | Interpretation
------|----------|-----------|-----------|---------------
0     | 10.00    | 0.92      | 0.05      | A only
115   | 12.30    | 0.88      | 0.85      | Overlap
125   | 12.50    | 0.06      | 0.91      | B only
```

**Thresholded** (>0.5):
```
[10.00-12.30s]: 1 speaker
[12.30-12.50s]: 2 speakers (overlap)
[12.50-15.00s]: 1 speaker
```

---

### Powerset Processing

**Classes**: {∅, {A}, {B}, {A+B}}

**Probabilities** (sample frames):
```
Frame | Time (s) | ∅    | {A}  | {B}  | {A+B} | Predicted
------|----------|------|------|------|-------|----------
0     | 10.00    | 0.02 | 0.94 | 0.03 | 0.01  | {A}
115   | 12.30    | 0.01 | 0.45 | 0.05 | 0.49  | {A+B}
125   | 12.50    | 0.03 | 0.04 | 0.91 | 0.02  | {B}
```

**Segments**:
```
[10.00-12.30s]: {A}
[12.30-12.50s]: {A+B} (overlap)
[12.50-15.00s]: {B}
```

---

### Final Output

```
Segmentation:
  [10.00-12.30s]: Single speaker (1 active)
  [12.30-12.50s]: Overlap (2 active)
  [12.50-15.00s]: Single speaker (1 active)

Accuracy: 100% (all frames correct)

Quality checks:
✓ Overlap detected correctly
✓ Boundaries within ±20ms of ground truth
✓ No false non-speech regions
✓ Frame-level granularity (20ms)
```

---

## Key Design Principles

### Neural vs Classical

| Classical SCD | Why It Fails | Neural Segmentation | Why It Works |
|--------------|--------------|---------------------|--------------|
| Energy-based | Soft-spoken patients | Learned features | Adapts to patterns |
| Hard boundaries | Gradual transitions | Soft probabilities | Smooth transitions |
| Assumes single speaker | Overlaps common | Explicit overlap modeling | Handles overlaps |
| Fixed thresholds | Noisy medical audio | Context-aware | Uses temporal context |

### PixIT Advantages
- Joint segmentation + separation
- Implicit overlap detection
- Long-range temporal context (LSTM)
- Trained on medical conversation data

### Powerset Advantages
- Explicit overlap classes
- Manageable complexity (2^N classes)
- Direct optimization on overlaps
- Good for 2-3 speaker scenarios

---

## Interaction with Module 1

| Module | Responsibility |
|--------|---------------|
| **Module 1 (SAD)** | Removes obvious non-speech |
| **Module 2 (Segmentation)** | Refines speech structure |

**Rules**:
1. Module 2 never marks speech as non-speech
2. Module 2 processes all Module 1 speech regions
3. Module 2 refines structure, not presence

---

## What Module 2 Does NOT Do

❌ Speaker embeddings (Module 3)  
❌ Speaker clustering (Module 4)  
❌ Speaker IDs/labels (Module 4)  
❌ Total speaker counting (Module 4)  
❌ Re-segmentation of non-speech  
❌ Speaker verification  
❌ Language identification

**Module 2 answers**: "How many speakers are active at this time?"  
**Module 2 does NOT answer**: "Who are they?"

---

## Key Constraints

**MUST DO**:
- Detect overlaps accurately (≥90%)
- Frame-level granularity (10-20ms)
- Preserve all Module 1 speech regions
- Output activity counts, not identities
- Use neural methods (no classical SCD)

**MUST NOT DO**:
- Assign speaker IDs
- Re-introduce non-speech
- Use energy-based SCD
- Make hard boundary assumptions

---

## Success Criteria

1. Overlap detection: ≥90% accuracy
2. Single-speaker accuracy: ≥95%
3. Temporal precision: ±50ms boundaries
4. No false non-speech regions
5. Frame-level output (10-20ms)
6. Processing: <2.0× real-time
7. No speaker IDs in output

---

## Technical Specifications

| Parameter | Value | Reason |
|-----------|-------|--------|
| Frame Size | 20ms | Balances resolution and context |
| Hop Size | 10ms | 50% overlap for smooth coverage |
| PixIT Threshold | 0.5 | Balanced sensitivity |
| Powerset Classes | 2^N (N=2 typically) | Manageable for 2 speakers |
| Overlap Target Accuracy | ≥90% | Critical for medical conversations |
| Boundary Precision | ±50ms | Acceptable for downstream processing |

---

## Summary

Neural segmentation (PixIT + powerset) replaces classical SCD to handle medical conversation overlaps and rapid turn-taking, producing frame-level speaker-activity representations without assigning identities.

---

**Next Module**: MODULE 3 — Speaker Embedding Extraction

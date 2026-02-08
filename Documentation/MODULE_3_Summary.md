# MODULE 3 — LANGUAGE-AGNOSTIC SPEAKER EMBEDDINGS (SUMMARY)

## What We Do
Extract language-agnostic speaker identity embeddings from speech using speaker-activity structure from Module 2, avoiding overlap corruption and ensuring stability across language switches.

## Input and Output
- **Input**: Audio (Module 0) + Speaker-activity structure (Module 2)
- **Output**: Speaker embeddings with confidence scores and temporal alignment

---

## Core Concept

**The Diarization Question**: "Which speech segments belong to the same speaker?"

**Answer**: Speaker embeddings + similarity measurement

**If embeddings are corrupted, nothing downstream can fix it**:
- Language leakage → same speaker, different clusters ❌
- Overlap corruption → ambiguous embeddings ❌
- Short-segment instability → inconsistent embeddings ❌

---

## Two Processing Paths

### Path 1: Single-Speaker Regions (Primary)
**What**: Clean speech with one active speaker  
**How**: Group frames, form 0.5-2.0s segments, extract embeddings  
**Output**: High-confidence embeddings

### Path 2: Overlap Regions (Careful Handling)
**What**: Multiple speakers active simultaneously  
**How**: Use PixIT-separated streams OR skip region  
**Output**: Medium-confidence embeddings OR no embeddings

**Critical**: NEVER extract embeddings from raw overlapped audio ❌

---

## Processing Steps

### 1. Branch by Speaker Activity
**What**: Separate single-speaker from overlap regions  
**How**:
- Parse Module 2 output: [(start, end, num_active), ...]
- num_active=1 → Path 1 (single-speaker)
- num_active=2 → Path 2 (overlap)

**Why**: Different regions need different handling  
*Example: [(0.0, 12.3, 1), (12.3, 12.5, 2), (12.5, 15.0, 1)] → 2 single-speaker regions, 1 overlap*

---

### 2. Process Single-Speaker Regions
**What**: Extract high-confidence embeddings  
**How**:
- Group contiguous single-speaker frames
- Form segments (250ms min, 0.5-2.0s preferred, 3.0s max)
- Extract embeddings via WavLM + Speaker Encoder
- L2 normalize
- Mark as high-confidence

**Why**: Clean speech = reliable embeddings  
*Example: 12.3s region → 7 segments (6×2.0s + 1×0.3s)*

---

### 3. Process Overlap Regions
**What**: Handle overlaps without corruption  
**How**:

**Strategy A (Preferred)**: PixIT-separated sources
- Use separated audio streams from Module 2
- Extract embeddings per stream
- Mark as medium-confidence

**Strategy B (Fallback)**: Overlap exclusion
- Skip region (no embeddings)
- Resolve overlap later in Module 4/5

**Why**: Raw overlap embeddings are ambiguous  
*Example: Mixed audio → corrupted embedding representing both speakers ❌*

---

### 4. Embedding Extraction
**What**: Convert audio to speaker representation  
**How**:
- Mel-filterbank features (80-dim)
- ECAPA-TDNN with SE-Res2Blocks
- Statistics pooling (mean + std)
- Projection (→ 256-dim)
- L2 normalization (unit norm)

**Why**: WavLM captures speaker identity robustly  
*Example: 2.0s audio → 256-dim unit-norm vector*

---

## LASPA: Language-Agnostic Training

### The Problem
```
Same doctor:
  English: "How are you feeling?" → embedding_1
  Hindi: "Aapko kya taklif hai?" → embedding_2
  
Without LASPA: embedding_1 ≠ embedding_2 ❌
```

### The Solution: LASPA

**LASPA = Language-Agnostic Speaker Prefix Adaptation**

**How it works**:
```
Language Label → Language Vector (64-dim)
                      ↓
Audio → Mel-filterbank (80-dim) + Language Vector → ECAPA-TDNN → Embedding

Language variation absorbed by language embedding
Speaker embedding stays language-clean
```

**Training objectives**:
- Speaker classification loss
- Contrastive loss (same speaker, different language → high similarity)
- LASPA regularization (minimize speaker-language correlation)

**Inference** (no language info needed):
```
Audio → Mel-filterbank → ECAPA-TDNN → Embedding
(LASPA already shaped the embedding space)
```

---

## Complete Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Audio + Speaker-Activity Structure                      │
│ [(0.0, 12.3, 1), (12.3, 12.5, 2), (12.5, 15.0, 1)]           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: BRANCH BY SPEAKER ACTIVITY                 │
│ Single-speaker vs Overlap                                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────┴─────────┐
                    ↓                   ↓
┌──────────────────────────┐  ┌──────────────────────────┐
│ PATH 1: Single-Speaker   │  │ PATH 2: Overlap          │
│ Group → Segment → Extract│  │ PixIT-separated OR Skip  │
│ High-confidence          │  │ Medium-confidence        │
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
│ Next: MODULE 4 (Similarity & Clustering)                       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete End-to-End Example

**Scenario**: Extracting embeddings from 15-second excerpt

### Initial State
```
Input from Module 2:
  [(0.0, 12.3, 1),    # Single speaker
   (12.3, 12.5, 2),   # Overlap
   (12.5, 15.0, 1)]   # Single speaker
```

---

### Path 1: Single-Speaker Processing

**Region 1: [0.0-12.3s] (12.3s)**

**Form segments** (2.0s preferred):
```
[0.0-2.0s], [2.0-4.0s], [4.0-6.0s], [6.0-8.0s], 
[8.0-10.0s], [10.0-12.0s], [12.0-12.3s]

Total: 7 segments
```

**Extract embeddings**:
```
Segment 1 [0.0-2.0s]:
  Audio → Mel-filterbank → ECAPA-TDNN
  Output: [0.123, -0.342, 0.567, ..., 0.234] (256-dim)
  L2 norm: 1.0 ✓
  Confidence: high
  
... (repeat for all 7 segments)
```

**Region 2: [12.5-15.0s] (2.5s)**

**Form segments**:
```
[12.5-14.5s], [14.5-15.0s]

Total: 2 segments
```

**Extract embeddings** (same process).

---

### Path 2: Overlap Processing

**Region 3: [12.3-12.5s] (overlap)**

**Strategy A**: PixIT-separated
```
separated_stream_A → embedding_A (medium confidence)
separated_stream_B → embedding_B (medium confidence)
```

**Strategy B**: Skip
```
No embeddings extracted
Resolve in Module 4/5
```

---

### Final Output

```python
embeddings = [
  # Region 1: 7 embeddings
  {"start_time": 0.0, "end_time": 2.0, "embedding_vector": [0.123, ...], 
   "confidence": "high", "source": "single_speaker"},
  {"start_time": 2.0, "end_time": 4.0, "embedding_vector": [0.118, ...], 
   "confidence": "high", "source": "single_speaker"},
  ... (5 more)
  
  # Region 2: 2 embeddings
  {"start_time": 12.5, "end_time": 14.5, "embedding_vector": [0.089, ...], 
   "confidence": "high", "source": "single_speaker"},
  {"start_time": 14.5, "end_time": 15.0, "embedding_vector": [0.091, ...], 
   "confidence": "high", "source": "single_speaker"},
]

Total: 9 embeddings
Quality: All L2-normalized, no raw overlap corruption ✓
```

---

## Key Design Principles

### Language-Agnostic via LASPA

**Problem**: Same speaker, different language → different embeddings

**Solution**: LASPA training
- Language prefix absorbs language variation
- Speaker embedding stays clean
- Contrastive constraints enforce similarity

**Result**: Same speaker, any language → similar embeddings (>0.9)

### Overlap Safety

**Problem**: Mixed audio → corrupted embeddings

**Solutions**:
1. PixIT-separated sources (medium confidence)
2. Skip overlap (resolve later)

**Never**: Extract from raw overlap ❌

### Structure-Aware Processing

**Module 2 provides structure** → Module 3 uses it

No heuristic segmentation, no energy-based slicing

---

## Architecture Details

### WavLM + Speaker Encoder

```
Audio (16 kHz)
    ↓
Mel-filterbank (80-dim)
    ↓
SE-Res2Block 1 (Multi-scale temporal context)
    ↓
SE-Res2Block 2
    ↓
Channel Attention
    ↓
Statistics Pooling (mean + std)
    ↓
Projection (→ 256-dim)
    ↓
L2 Normalization
    ↓
Speaker Embedding (256-dim, unit norm)
```

**Why ECAPA-TDNN**: State-of-the-art speaker recognition, efficient, proven  
**Why 256-dim**: Optimal for 2-10 speakers, avoids overfitting

---

## What Module 3 Does NOT Do

❌ Similarity scoring (Module 4)  
❌ Clustering (Module 4)  
❌ Speaker counting (Module 4)  
❌ Speaker labeling (Module 4)  
❌ RTTM generation (Module 5)  
❌ Language identification (not needed, LASPA handles it)  
❌ Overlap resolution (Module 4/5)

**Module 3 answers**: "What is this speaker's identity representation?"  
**Module 3 does NOT answer**: "Which speaker is this?"

---

## Key Constraints

**MUST DO**:
- Extract from single-speaker regions (high-confidence)
- Use PixIT-separated OR skip overlaps (never raw overlap)
- Segment duration: 250ms min, 0.5-2.0s preferred
- L2 normalize all embeddings
- Tag confidence levels
- Use LASPA-trained model (language-agnostic)

**MUST NOT DO**:
- Extract from raw overlaps
- Perform clustering
- Assign speaker IDs
- Use energy-based segmentation
- Ignore Module 2 structure

---

## Success Criteria

1. Language-agnostic: Same speaker, different language → similarity >0.9
2. Speaker-discriminative: Different speakers → similarity <0.3
3. Overlap-safe: No embeddings from raw overlaps
4. Segment coverage: ≥90% of single-speaker regions
5. Minimum duration: All segments ≥250ms
6. Normalized: All embeddings unit-norm (L2=1.0)
7. Confidence-tagged: All embeddings have scores
8. No clustering: Module 3 does NOT group

---

## Technical Specifications

| Parameter | Value | Reason |
|-----------|-------|--------|
| Embedding Dimension | 256 | Optimal for 2-10 speakers |
| Min Segment Duration | 250ms | Preserve backchannels |
| Preferred Duration | 0.5-2.0s | Balance stability and granularity |
| Max Segment Duration | 3.0s | Avoid diminishing returns |
| WavLM Backbone | Pre-trained | Strong speaker features |
| LASPA Prefix Size | 32-64 dim | Sufficient for language variation |
| L2 Normalization | Required | Enable cosine similarity |
| Overlap Strategy | PixIT-separated OR skip | Avoid corruption |

---

## Summary

Language-agnostic speaker embedding extraction using LASPA-trained ECAPA-TDNN encoder, processing single-speaker regions with high confidence and handling overlaps via PixIT separation or exclusion, producing reliable representations for downstream clustering.

---

**Next Module**: MODULE 4 — Similarity Scoring & Clustering

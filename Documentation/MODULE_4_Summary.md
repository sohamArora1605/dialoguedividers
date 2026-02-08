# MODULE 4 — SIMILARITY SCORING & AFFINITY MATRIX (SUMMARY)

## What We Do
Convert speaker embeddings into pairwise similarity scores and construct a confidence-weighted affinity matrix that accurately reflects "same speaker vs different speaker" relationships across language switches.

## Input and Output
- **Input**: Speaker embeddings with metadata (from Module 3)
- **Output**: Affinity matrix (speaker similarity graph) ready for clustering

---

## Core Concept

**Similarity scoring is NOT trivial** because:
- LASPA-trained embeddings (different geometry than x-vectors)
- Variable confidence (some embeddings more reliable)
- PixIT-separated sources (may have artifacts)
- Short medical turns (backchannels, interruptions)

**If scoring is naive, clustering will fail.**

---

## Cosine vs PLDA

### Traditional (x-vectors)
```
x-vectors → PLDA scoring (mandatory)
Assumes: Gaussian distribution, substantial training data
```

### Modern (LASPA + WavLM-Base)
```
LASPA-trained embeddings → Cosine similarity (primary)
Why: Angular separation enforced by LASPA training
```

**👉 Cosine similarity is PRIMARY for DISPLACE-2026**

**PLDA**: Optional, for calibration only (not core modeling)

---

## Processing Steps

### 1. Pairwise Cosine Similarity
**What**: Compute similarity for all embedding pairs  
**How**:
```
cos(e_i, e_j) = e_i · e_j  (both L2-normalized)

Range: [-1, +1]
  +1: Same speaker
   0: Unrelated
  -1: Very different
```

**Why**: LASPA training enforces angular geometry  
*Example: Doctor (English) vs Doctor (Hindi) → cos=0.93*

---

### 2. Confidence-Aware Weighting
**What**: Weight similarities by embedding reliability  
**How**:
```
Confidence weights:
  high (single-speaker): 1.0
  medium (PixIT-separated): 0.7

Weighted similarity:
  sim(i,j) = w_i × w_j × cos(e_i, e_j)
```

**Why**: Prevent noisy embeddings from dominating  
*Example: high×medium × 0.85 = 1.0×0.7×0.85 = 0.595*

---

### 3. Temporal Constraints
**What**: Apply soft temporal bias  
**How**:
```
Temporal weight:
  temp_weight(t_i, t_j) = exp(-|t_i - t_j| / τ)
  
  τ = 30s (decay constant)

Temporal-aware similarity:
  sim × (0.7 + 0.3 × temp_weight)
```

**Why**: Preserve conversation flow  
*Example: 2s apart → boost, 45s apart → down-weight*

---

### 4. Overlap-Aware Scoring
**What**: Weight based on embedding source  
**How**:
```
Source weights:
  single-single: 1.0
  single-separated: 0.8
  separated-separated: 0.5
```

**Why**: PixIT separation may have artifacts  
*Example: Both separated → heavily down-weighted*

---

### 5. Affinity Matrix Construction
**What**: Build N×N similarity matrix  
**How**:
```
For N embeddings:
  A[i,j] = final_similarity(i, j)
  
Properties:
  - Diagonal: 1.0 (self-similarity)
  - Symmetric: A[i,j] = A[j,i]
  - Values: [0, 1]
```

**Why**: Input to clustering (Module 5)  
*Example: 4 embeddings → 4×4 matrix*

---

### 6. Sparsification (Optional)
**What**: Remove weak connections  
**How**:
```
Top-K neighbors:
  Keep only K strongest connections per segment
  
Threshold:
  Zero out similarities < threshold
```

**Why**: Improve clustering stability  
*Example: K=10 → keep 10 strongest neighbors*

---

## Complete Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Speaker Embeddings from Module 3                        │
│ [{start, end, embedding, confidence, source}, ...]             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: PAIRWISE COSINE SIMILARITY                 │
│ cos(e_i, e_j) for all pairs                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: CONFIDENCE WEIGHTING                       │
│ high=1.0, medium=0.7                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: TEMPORAL CONSTRAINTS                       │
│ exp(-Δt/τ), boost adjacent, down-weight distant               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: OVERLAP-AWARE SCORING                      │
│ single-single=1.0, single-separated=0.8, separated-separated=0.5│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: AFFINITY MATRIX CONSTRUCTION               │
│ Build N×N symmetric matrix                                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 6: SPARSIFICATION (OPTIONAL)                  │
│ Top-K neighbors or threshold                                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Affinity Matrix                                        │
│ N×N matrix ready for clustering (Module 5)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete End-to-End Example

**Scenario**: Building affinity matrix for 4 embeddings (2 Doctor, 2 Patient)

### Initial State
```
Embeddings:
  0: Doctor, English, [0-2s], high-confidence
  1: Doctor, Hindi, [2-4s], high-confidence
  2: Patient, English, [5-7s], high-confidence
  3: Patient, Hindi, [7-9s], high-confidence
```

---

### Step 1: Cosine Similarity

```
cos(0, 1) = 0.93  # Same speaker (Doctor)
cos(0, 2) = 0.21  # Different speakers
cos(0, 3) = 0.19  # Different speakers
cos(1, 2) = 0.18  # Different speakers
cos(1, 3) = 0.20  # Different speakers
cos(2, 3) = 0.94  # Same speaker (Patient)
```

---

### Step 2: Confidence Weighting

```
All high-confidence (weight=1.0)
No change in this example
```

---

### Step 3: Temporal Constraints

```
Temporal weights (τ=30s):
  temp_weight(0, 1) = exp(-2/30) = 0.935  # 2s apart
  temp_weight(0, 2) = exp(-5/30) = 0.846  # 5s apart
  temp_weight(2, 3) = exp(-2/30) = 0.935  # 2s apart

Apply bias:
  sim(0, 1) = 0.93 × (0.7 + 0.3×0.935) = 0.91
  sim(0, 2) = 0.21 × (0.7 + 0.3×0.846) = 0.20
  sim(2, 3) = 0.94 × (0.7 + 0.3×0.935) = 0.92
```

---

### Step 4: Overlap-Aware Scoring

```
All single-speaker sources (weight=1.0)
No change in this example
```

---

### Step 5: Affinity Matrix

```
A = [
  [1.00, 0.91, 0.20, 0.18],  # Embedding 0
  [0.91, 1.00, 0.18, 0.19],  # Embedding 1
  [0.20, 0.18, 1.00, 0.92],  # Embedding 2
  [0.18, 0.19, 0.92, 1.00]   # Embedding 3
]

Block structure:
  - (0,1): Doctor (high similarity >0.9)
  - (2,3): Patient (high similarity >0.9)
  - Off-diagonal: Different speakers (low <0.2)
```

---

### Step 6: Sparsification (K=2)

```
A_sparse = [
  [1.00, 0.91, 0.20, 0.00],
  [0.91, 1.00, 0.00, 0.19],
  [0.20, 0.00, 1.00, 0.92],
  [0.00, 0.19, 0.92, 1.00]
]

Kept only top-2 neighbors per segment
```

---

### Final Output

```
Affinity matrix ready for clustering:
✓ Clear block structure (Doctor: 0-1, Patient: 2-3)
✓ High intra-speaker similarity (>0.9)
✓ Low inter-speaker similarity (<0.2)
✓ Sparsified for stability

Next: Module 5 (Clustering)
```

---

## Key Design Principles

### Cosine Similarity Primary

**Why**: LASPA training enforces angular separation
- Same speaker, different language → high similarity
- Different speakers → low similarity
- No Gaussian assumptions needed

### Confidence-Aware Weighting

**Why**: Not all embeddings equally reliable
- High-confidence (single-speaker): Full weight
- Medium-confidence (PixIT-separated): Reduced weight
- Prevents noisy embeddings from dominating

### Temporal Constraints

**Why**: Conversations have natural structure
- Adjacent segments → likely same speaker
- Distant segments → less likely
- Soft bias, not hard cutoff

### Overlap-Aware Scoring

**Why**: PixIT separation may have artifacts
- Single-single: Full weight
- Single-separated: Reduced weight
- Separated-separated: Heavily reduced

---

## What Module 4 Does NOT Do

❌ Clustering (Module 5)  
❌ Speaker counting (Module 5)  
❌ Thresholding into speakers (Module 5)  
❌ RTTM generation (Module 5)  
❌ Overlap resolution (Module 5)  
❌ VBx (Module 5)

**Module 4 answers**: "How similar are these two segments?"  
**Module 4 does NOT answer**: "Which speaker is this?"

---

## Failure Modes Avoided

| Old Mistake | New Fix | Benefit |
|-------------|---------|---------|
| PLDA overfitting | Cosine primary | Robust to limited data |
| Language drift | LASPA-trained space | Language-agnostic |
| Overlap corruption | Confidence weighting | Down-weight artifacts |
| Short utterance errors | Temporal bias | Preserve conversation flow |
| Dense matrix noise | Sparsification | Cleaner graph |

---

## Key Constraints

**MUST DO**:
- Use cosine similarity as primary metric
- Apply confidence weighting (high=1.0, medium=0.7)
- Apply temporal bias (soft, not hard)
- Apply overlap-aware weighting
- Build symmetric affinity matrix
- Optional: Sparsify (top-K or threshold)

**MUST NOT DO**:
- Perform clustering
- Assign speaker IDs
- Count speakers
- Generate RTTM
- Use PLDA as primary (optional only)

---

## Success Criteria

1. Language-agnostic: Same speaker, different language → high similarity (>0.8)
2. Speaker-discriminative: Different speakers → low similarity (<0.3)
3. Confidence-aware: Low-confidence embeddings down-weighted
4. Temporally-coherent: Adjacent segments boosted
5. Overlap-safe: Separated-separated pairs heavily down-weighted
6. Block structure: Clear speaker blocks in matrix
7. Sparse: Top-K or threshold applied
8. Symmetric: A[i,j] = A[j,i]

---

## Technical Specifications

| Parameter | Value | Reason |
|-----------|-------|--------|
| Primary Metric | Cosine similarity | LASPA angular geometry |
| High Confidence Weight | 1.0 | Single-speaker, clean |
| Medium Confidence Weight | 0.7 | PixIT-separated, artifacts |
| Temporal Decay (τ) | 30s | Balance locality and flexibility |
| Single-Single Weight | 1.0 | Both clean |
| Single-Separated Weight | 0.8 | One clean, one separated |
| Separated-Separated Weight | 0.5 | Both may have artifacts |
| Top-K Neighbors | 5-20 | Depends on conversation size |
| Sparsity Threshold | 0.2-0.4 | Depends on embedding quality |

---

## Summary

Confidence-weighted, temporally-aware, overlap-safe affinity matrix construction using cosine similarity as primary metric for LASPA-trained embeddings, producing a clean similarity graph ready for clustering.

---

**Next Module**: MODULE 5 — Clustering (Spectral + VBx refinement)

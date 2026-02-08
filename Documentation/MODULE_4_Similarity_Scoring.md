# MODULE 4 — SIMILARITY SCORING & AFFINITY MATRIX CONSTRUCTION

*(DISPLACE-2026 compatible, LASPA-aware, confidence-weighted, structure-aware)*

---

## 4.1 What This Module Does

**Purpose (2-line summary)**  
Converts speaker embeddings into pairwise similarity scores and constructs a confidence-weighted affinity matrix that accurately reflects "same speaker vs different speaker" relationships, even across language switches and short medical utterances.

**Input → Output**
- **Input**: Speaker embeddings with metadata (from Module 3)
- **Output**: Affinity matrix (speaker similarity graph) ready for clustering

---

## 4.2 Why Module 4 is Its Own Module (Fundamental Concept)

**Similarity scoring is NOT trivial distance computation** because:

| Challenge | Why It Matters | Module 4's Solution |
|-----------|----------------|---------------------|
| **LASPA-trained embeddings** | Different geometry than x-vectors | Use cosine similarity (not PLDA) |
| **Variable confidence** | Some embeddings more reliable | Confidence-aware weighting |
| **PixIT-separated sources** | May contain artifacts | Overlap-aware scoring |
| **Short medical turns** | Backchannels, interruptions | Temporal constraints |
| **Language switches** | Same speaker, different language | LASPA ensures stability |

**If scoring is naive, clustering will fail** - no matter how good the embeddings are.

---

## 4.3 What Module 4 Receives (Strict Input Specification)

**From Module 3**: List of embedding objects

### Input Format

```python
embeddings = [
  {
    "start_time": 0.0,
    "end_time": 2.0,
    "embedding_vector": [0.123, -0.342, 0.567, ..., 0.234],  # 256-dim
    "confidence": "high",
    "source": "single_speaker",
    "duration": 2.0
  },
  {
    "start_time": 2.0,
    "end_time": 4.0,
    "embedding_vector": [0.118, -0.338, 0.571, ..., 0.229],
    "confidence": "high",
    "source": "single_speaker",
    "duration": 2.0
  },
  {
    "start_time": 12.3,
    "end_time": 12.5,
    "embedding_vector": [0.089, -0.412, 0.523, ..., 0.201],
    "confidence": "medium",
    "source": "pixit_separated",
    "duration": 0.2
  },
  ...
]
```

**No speaker labels, no speaker IDs** - Module 4 doesn't know ground truth.

---

## 4.4 First Design Decision: Cosine vs PLDA

### 4.4.1 Traditional Pipeline (x-vectors)

**Old approach**:
```
x-vectors → PLDA scoring (mandatory)

PLDA assumptions:
- Speaker embeddings follow Gaussian distribution
- Within-speaker and between-speaker covariance matrices
- Requires substantial training data
```

### 4.4.2 New Pipeline (LASPA + ECAPA-TDNN)

**Modern approach**:
```
LASPA-trained embeddings → Cosine similarity (primary)

Why:
- LASPA training enforces angular separation
- ECAPA-TDNN embeddings optimized for cosine distance
- Cosine directly reflects speaker similarity
```

**👉 Cosine similarity is the PRIMARY scorer for DISPLACE-2026**

---

## 4.5 Why Cosine Similarity is Preferred

### 4.5.1 LASPA Training Enforces Angular Geometry

**During LASPA training** (Module 3), contrastive loss enforces:

```
Same speaker, different language:
  cos(e_A_english, e_A_hindi) > 0.9

Different speakers, same language:
  cos(e_A_english, e_B_english) < 0.3
```

**Result**: Speaker similarity is **directly encoded in angular distance**.

### 4.5.2 Cosine Similarity Definition

```
For embeddings e_i, e_j (both L2-normalized):

cos(e_i, e_j) = e_i · e_j = Σ(e_i[k] × e_j[k])

Range: [-1, +1]
  +1: Identical direction (same speaker)
   0: Orthogonal (unrelated)
  -1: Opposite direction (very different)
```

**Numerical Example**:

```
Embedding A1 (Doctor, English): [0.8, 0.2, 0.1, 0.5, ...]
Embedding A2 (Doctor, Hindi):   [0.75, 0.25, 0.15, 0.48, ...]
Embedding B1 (Patient, English): [0.1, 0.7, 0.6, 0.2, ...]

Cosine similarities:
  cos(A1, A2) = 0.93 → Same speaker ✓
  cos(A1, B1) = 0.21 → Different speakers ✓
  cos(A2, B1) = 0.19 → Different speakers ✓
```

### 4.5.3 Why PLDA Assumptions Fail

**PLDA assumes**:
1. Gaussian distribution of speaker embeddings
2. Sufficient training data to estimate covariance matrices
3. Embeddings from generative model (e.g., i-vectors, x-vectors)

**LASPA + ECAPA-TDNN embeddings**:
1. ❌ Not Gaussian (learned via contrastive loss)
2. ❌ Limited medical conversation data
3. ❌ Discriminative model, not generative

**PLDA on non-Gaussian embeddings** → overfitting, poor generalization

---

## 4.6 Does PLDA Disappear Completely?

**No** - but it becomes **secondary/optional**.

### Two Valid Setups

**Setup A: Pure Cosine (Default, Recommended)**

```
Use when:
  - Embeddings are LASPA-trained ✓
  - Segment lengths are short (medical backchannels) ✓
  - Dataset is small (DISPLACE) ✓
  - No dev data for PLDA training

Scoring:
  similarity(i, j) = cosine(e_i, e_j)
```

**Setup B: Light PLDA Calibration (Optional)**

```
Use when:
  - You have sufficient dev data (>10 hours labeled)
  - Want score normalization
  - Need probabilistic interpretation

Scoring:
  similarity(i, j) = PLDA_score(e_i, e_j)
  
  But PLDA is for calibration, not core modeling
```

**For DISPLACE-2026**: **Setup A (Pure Cosine) is recommended**.

---

## 4.7 How Similarity is Computed (Step-by-Step)

### 4.7.1 Step 1: Verify Normalization

**All embeddings from Module 3 are already L2-normalized**:

```python
for embedding_obj in embeddings:
    e = embedding_obj["embedding_vector"]
    norm = sqrt(sum(x**2 for x in e))
    assert abs(norm - 1.0) < 1e-6  # Verify unit norm
```

**Why this matters**: Cosine similarity requires unit-norm vectors.

### 4.7.2 Step 2: Pairwise Cosine Similarity

**For all pairs (i, j)**:

```python
def cosine_similarity(e_i, e_j):
    """
    Compute cosine similarity between two L2-normalized embeddings.
    
    Args:
        e_i: Embedding vector (256-dim, unit norm)
        e_j: Embedding vector (256-dim, unit norm)
    
    Returns:
        Similarity score in [-1, +1]
    """
    return sum(e_i[k] * e_j[k] for k in range(len(e_i)))
```

**Computational complexity**:
```
N embeddings → N×(N-1)/2 pairs
Example: 100 embeddings → 4,950 pairs
```

**Numerical Example**:

```
Embedding 1: [0.123, -0.342, 0.567, ..., 0.234]
Embedding 2: [0.118, -0.338, 0.571, ..., 0.229]

Cosine similarity:
  = (0.123×0.118) + (-0.342×-0.338) + (0.567×0.571) + ... + (0.234×0.229)
  = 0.014 + 0.116 + 0.324 + ... + 0.054
  = 0.947
  
Interpretation: Very high similarity (likely same speaker)
```

### 4.7.3 Step 3: Confidence-Aware Weighting (NEW, CRITICAL)

**Not all embeddings are equally reliable.**

**Define confidence weights**:

```python
def get_confidence_weight(embedding_obj):
    """
    Assign weight based on embedding confidence.
    
    Args:
        embedding_obj: Embedding with metadata
    
    Returns:
        Weight in (0, 1]
    """
    if embedding_obj["confidence"] == "high":
        return 1.0  # Single-speaker, clean
    elif embedding_obj["confidence"] == "medium":
        return 0.7  # PixIT-separated, may have artifacts
    else:
        return 0.5  # Low confidence (if any)
```

**Weighted similarity**:

```python
def weighted_similarity(i, j, embeddings):
    """
    Compute confidence-weighted similarity.
    
    Args:
        i, j: Embedding indices
        embeddings: List of embedding objects
    
    Returns:
        Weighted similarity score
    """
    e_i = embeddings[i]["embedding_vector"]
    e_j = embeddings[j]["embedding_vector"]
    w_i = get_confidence_weight(embeddings[i])
    w_j = get_confidence_weight(embeddings[j])
    
    cos_sim = cosine_similarity(e_i, e_j)
    
    return w_i * w_j * cos_sim
```

**Why this matters**:

```
Example:
  Embedding A: high-confidence (w=1.0)
  Embedding B: medium-confidence (w=0.7)
  Cosine similarity: 0.85
  
  Weighted similarity: 1.0 × 0.7 × 0.85 = 0.595
  
  Effect: Down-weighted to reflect uncertainty in B
```

**Prevents noisy PixIT-separated segments from dominating clustering.**

---

## 4.8 Temporal Constraints (VERY IMPORTANT)

### 4.8.1 Why Temporal Structure Matters

**Medical conversations have natural structure**:

```
Doctor:  [0-5s]   "How are you feeling?"
Patient: [5-10s]  "Not well, doctor"
Doctor:  [10-15s] "Tell me more"
Patient: [15-20s] "I have a headache"

Temporal proximity → likely speaker continuity
```

**But naive clustering ignores time** → can merge distant segments incorrectly.

### 4.8.2 Soft Temporal Gating

**Apply temporal bias to similarity scores**:

```python
def temporal_weight(t_i, t_j, tau=30.0):
    """
    Compute temporal proximity weight.
    
    Args:
        t_i, t_j: Segment start times (seconds)
        tau: Temporal decay constant (default 30s)
    
    Returns:
        Weight in (0, 1]
    """
    delta_t = abs(t_i - t_j)
    return exp(-delta_t / tau)
```

**Temporal-aware similarity**:

```python
def temporal_aware_similarity(i, j, embeddings, tau=30.0):
    """
    Compute similarity with temporal bias.
    
    Args:
        i, j: Embedding indices
        embeddings: List of embedding objects
        tau: Temporal decay constant
    
    Returns:
        Temporally-weighted similarity
    """
    # Base weighted similarity
    base_sim = weighted_similarity(i, j, embeddings)
    
    # Temporal weight
    t_i = embeddings[i]["start_time"]
    t_j = embeddings[j]["start_time"]
    temp_weight = temporal_weight(t_i, t_j, tau)
    
    # Combined (soft bias, not hard cutoff)
    return base_sim * (0.7 + 0.3 * temp_weight)
```

**Numerical Example**:

```
Embedding A: start_time = 5.0s
Embedding B: start_time = 7.0s (2s apart)
Embedding C: start_time = 50.0s (45s apart)

Cosine similarity (A, B) = 0.85
Cosine similarity (A, C) = 0.85 (same!)

Temporal weights (tau=30s):
  temp_weight(A, B) = exp(-2/30) = 0.935
  temp_weight(A, C) = exp(-45/30) = 0.223

Temporal-aware similarity:
  sim(A, B) = 0.85 × (0.7 + 0.3×0.935) = 0.85 × 0.981 = 0.834
  sim(A, C) = 0.85 × (0.7 + 0.3×0.223) = 0.85 × 0.767 = 0.652
  
Effect: Adjacent segments (A, B) boosted, distant (A, C) down-weighted
```

**This helps**:
- Short backchannels attach correctly to speaker
- Avoids accidental long-range merges
- Preserves natural conversation flow

**⚠️ This is soft bias, not a hard rule** - strong similarity can still override temporal distance.

---

## 4.9 Overlap-Aware Scoring (Critical)

### 4.9.1 Why Overlap Source Matters

**Module 4 knows which embeddings came from**:
- Single-speaker regions (clean, reliable)
- PixIT-separated overlap regions (may have artifacts)

**Different source combinations have different reliability.**

### 4.9.2 Source-Based Weighting

```python
def overlap_aware_weight(source_i, source_j):
    """
    Compute weight based on embedding sources.
    
    Args:
        source_i, source_j: "single_speaker" or "pixit_separated"
    
    Returns:
        Weight in (0, 1]
    """
    if source_i == "single_speaker" and source_j == "single_speaker":
        return 1.0  # Both clean
    elif source_i == "single_speaker" or source_j == "single_speaker":
        return 0.8  # One clean, one separated
    else:  # Both "pixit_separated"
        return 0.5  # Both may have artifacts
```

**Why heavily down-weight separated-separated pairs?**

```
PixIT separation artifacts:
  - Residual cross-talk
  - Spectral leakage
  - Phase distortion

Separated embedding A: May contain traces of Speaker B
Separated embedding B: May contain traces of Speaker A

Similarity may be artificially high ✗
```

**Final similarity computation**:

```python
def final_similarity(i, j, embeddings, tau=30.0):
    """
    Compute final similarity with all weights.
    
    Args:
        i, j: Embedding indices
        embeddings: List of embedding objects
        tau: Temporal decay constant
    
    Returns:
        Final weighted similarity score
    """
    # Base cosine similarity
    e_i = embeddings[i]["embedding_vector"]
    e_j = embeddings[j]["embedding_vector"]
    cos_sim = cosine_similarity(e_i, e_j)
    
    # Confidence weights
    w_i = get_confidence_weight(embeddings[i])
    w_j = get_confidence_weight(embeddings[j])
    
    # Temporal weight
    t_i = embeddings[i]["start_time"]
    t_j = embeddings[j]["start_time"]
    temp_weight = temporal_weight(t_i, t_j, tau)
    
    # Overlap-aware weight
    source_i = embeddings[i]["source"]
    source_j = embeddings[j]["source"]
    overlap_weight = overlap_aware_weight(source_i, source_j)
    
    # Combine all factors
    final_sim = cos_sim * w_i * w_j * (0.7 + 0.3 * temp_weight) * overlap_weight
    
    return final_sim
```

---

## 4.10 Affinity Matrix Construction (Final Product)

### 4.10.1 What is an Affinity Matrix?

**Definition**: Symmetric matrix where entry (i, j) represents similarity between segments i and j.

```
For N embeddings:

A = [
  [1.00, 0.85, 0.23, 0.12, ...],  # Segment 0 vs all
  [0.85, 1.00, 0.19, 0.08, ...],  # Segment 1 vs all
  [0.23, 0.19, 1.00, 0.91, ...],  # Segment 2 vs all
  [0.12, 0.08, 0.91, 1.00, ...],  # Segment 3 vs all
  ...
]

Properties:
  - Diagonal: 1.0 (self-similarity)
  - Symmetric: A[i,j] = A[j,i]
  - Values: [0, 1] (after weighting)
```

### 4.10.2 Construction Algorithm

```python
import numpy as np

def build_affinity_matrix(embeddings, tau=30.0):
    """
    Build affinity matrix from embeddings.
    
    Args:
        embeddings: List of embedding objects
        tau: Temporal decay constant
    
    Returns:
        Affinity matrix (N×N numpy array)
    """
    N = len(embeddings)
    A = np.zeros((N, N))
    
    # Compute all pairwise similarities
    for i in range(N):
        for j in range(i, N):  # Upper triangle + diagonal
            if i == j:
                A[i, j] = 1.0  # Self-similarity
            else:
                sim = final_similarity(i, j, embeddings, tau)
                A[i, j] = sim
                A[j, i] = sim  # Symmetric
    
    return A
```

### 4.10.3 Numerical Example

**Scenario**: 4 embeddings (2 from Doctor, 2 from Patient)

```
Embeddings:
  0: Doctor, English, [0-2s], high-confidence
  1: Doctor, Hindi, [2-4s], high-confidence
  2: Patient, English, [5-7s], high-confidence
  3: Patient, Hindi, [7-9s], high-confidence

Cosine similarities (before weighting):
  cos(0, 1) = 0.93  # Same speaker (Doctor)
  cos(0, 2) = 0.21  # Different speakers
  cos(0, 3) = 0.19  # Different speakers
  cos(1, 2) = 0.18  # Different speakers
  cos(1, 3) = 0.20  # Different speakers
  cos(2, 3) = 0.94  # Same speaker (Patient)

After temporal + confidence weighting:
  A[0, 1] = 0.93 × 1.0 × 1.0 × 0.98 × 1.0 = 0.91
  A[0, 2] = 0.21 × 1.0 × 1.0 × 0.75 × 1.0 = 0.16
  A[0, 3] = 0.19 × 1.0 × 1.0 × 0.73 × 1.0 = 0.14
  A[1, 2] = 0.18 × 1.0 × 1.0 × 0.77 × 1.0 = 0.14
  A[1, 3] = 0.20 × 1.0 × 1.0 × 0.75 × 1.0 = 0.15
  A[2, 3] = 0.94 × 1.0 × 1.0 × 0.98 × 1.0 = 0.92

Affinity matrix:
  A = [
    [1.00, 0.91, 0.16, 0.14],
    [0.91, 1.00, 0.14, 0.15],
    [0.16, 0.14, 1.00, 0.92],
    [0.14, 0.15, 0.92, 1.00]
  ]

Interpretation:
  - Block (0,1): High similarity (Doctor)
  - Block (2,3): High similarity (Patient)
  - Off-diagonal blocks: Low similarity (different speakers)
  
Perfect for clustering! ✓
```

---

## 4.11 Sparsification (Important for Stability)

### 4.11.1 Why Sparsify?

**Dense affinity matrix problems**:
- Noise accumulates (low similarities pollute clustering)
- Spectral clustering sensitive to weak edges
- VBx convergence slows down

**Solution**: Keep only strong connections.

### 4.11.2 Top-K Neighbors

**For each segment, keep only K most similar neighbors**:

```python
def sparsify_affinity_matrix(A, K=10):
    """
    Sparsify affinity matrix by keeping top-K neighbors per segment.
    
    Args:
        A: Affinity matrix (N×N)
        K: Number of neighbors to keep per segment
    
    Returns:
        Sparsified affinity matrix
    """
    N = A.shape[0]
    A_sparse = np.zeros_like(A)
    
    for i in range(N):
        # Get top-K neighbors (excluding self)
        similarities = A[i, :].copy()
        similarities[i] = -1  # Exclude self
        top_k_indices = np.argsort(similarities)[-K:]
        
        # Keep only top-K connections
        for j in top_k_indices:
            A_sparse[i, j] = A[i, j]
            A_sparse[j, i] = A[i, j]  # Maintain symmetry
        
        # Keep self-similarity
        A_sparse[i, i] = 1.0
    
    return A_sparse
```

**Typical K values**:
- Small conversations (2-3 speakers): K=5-10
- Larger conversations (4-6 speakers): K=10-20

### 4.11.3 Threshold-Based Sparsification

**Alternative: Zero out similarities below threshold**:

```python
def threshold_sparsify(A, threshold=0.3):
    """
    Zero out similarities below threshold.
    
    Args:
        A: Affinity matrix (N×N)
        threshold: Minimum similarity to keep
    
    Returns:
        Sparsified affinity matrix
    """
    A_sparse = A.copy()
    A_sparse[A_sparse < threshold] = 0.0
    np.fill_diagonal(A_sparse, 1.0)  # Preserve self-similarity
    return A_sparse
```

**Typical thresholds**: 0.2-0.4 (depends on embedding quality)

---

## 4.12 What Module 4 Does NOT Do

**Module 4 prepares the similarity graph** - it does NOT perform:

❌ **Clustering**: Grouping segments into speakers (Module 5)  
❌ **Speaker counting**: Determining number of unique speakers (Module 5)  
❌ **Thresholding into final speakers**: Assigning speaker labels (Module 5)  
❌ **RTTM generation**: Creating final output format (Module 5)  
❌ **Overlap resolution**: Assigning overlap frames to speakers (Module 5)  
❌ **VBx**: Variational Bayes diarization (Module 5)

**Module 4 answers**: "How similar are these two segments?"  
**Module 4 does NOT answer**: "Which speaker is this?" or "How many speakers are there?"

---

## 4.13 Failure Modes Avoided by This Design

| Old Mistake | Consequence | New Fix | Benefit |
|-------------|-------------|---------|---------|
| **PLDA overfitting** | Poor generalization on small data | Cosine primary | Robust to limited data |
| **Language drift** | Same speaker → different clusters | LASPA-trained space | Language-agnostic |
| **Overlap corruption** | Noisy embeddings dominate | Confidence weighting | Down-weight artifacts |
| **Short utterance errors** | Backchannels misassigned | Temporal bias | Preserve conversation flow |
| **Dense matrix noise** | Clustering instability | Sparsification | Cleaner graph |
| **Uniform weighting** | All embeddings treated equally | Multi-factor weighting | Reliability-aware |

---

## 4.14 Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Speaker Embeddings from Module 3                        │
│ [{start, end, embedding, confidence, source}, ...]             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: PAIRWISE COSINE SIMILARITY                 │
│ Compute cos(e_i, e_j) for all pairs                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: CONFIDENCE WEIGHTING                       │
│ Apply weights based on embedding confidence                    │
│ high=1.0, medium=0.7                                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: TEMPORAL CONSTRAINTS                       │
│ Apply soft temporal bias (exp(-Δt/τ))                         │
│ Boost adjacent, down-weight distant                            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: OVERLAP-AWARE SCORING                      │
│ Weight based on source:                                        │
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
│ Keep top-K neighbors or threshold                              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Affinity Matrix                                        │
│ N×N matrix ready for clustering (Module 5)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4.15 Complete End-to-End Example

**Scenario**: Building affinity matrix for 4 embeddings

### Initial State

```python
embeddings = [
  {
    "start_time": 0.0,
    "end_time": 2.0,
    "embedding_vector": [0.8, 0.2, 0.1, 0.5, ...],  # 256-dim
    "confidence": "high",
    "source": "single_speaker"
  },
  {
    "start_time": 2.0,
    "end_time": 4.0,
    "embedding_vector": [0.75, 0.25, 0.15, 0.48, ...],
    "confidence": "high",
    "source": "single_speaker"
  },
  {
    "start_time": 5.0,
    "end_time": 7.0,
    "embedding_vector": [0.1, 0.7, 0.6, 0.2, ...],
    "confidence": "high",
    "source": "single_speaker"
  },
  {
    "start_time": 7.0,
    "end_time": 9.0,
    "embedding_vector": [0.15, 0.68, 0.58, 0.22, ...],
    "confidence": "high",
    "source": "single_speaker"
  }
]

Ground truth (for validation):
  Embeddings 0, 1: Doctor
  Embeddings 2, 3: Patient
```

---

### Step 1: Pairwise Cosine Similarity

```python
# Compute all pairs
cos(0, 1) = 0.93  # Doctor, English vs Doctor, Hindi
cos(0, 2) = 0.21  # Doctor vs Patient
cos(0, 3) = 0.19  # Doctor vs Patient
cos(1, 2) = 0.18  # Doctor vs Patient
cos(1, 3) = 0.20  # Doctor vs Patient
cos(2, 3) = 0.94  # Patient, English vs Patient, Hindi
```

---

### Step 2: Confidence Weighting

```python
# All high-confidence (weight=1.0)
weighted_sim(0, 1) = 1.0 × 1.0 × 0.93 = 0.93
weighted_sim(0, 2) = 1.0 × 1.0 × 0.21 = 0.21
... (no change, all high-confidence)
```

---

### Step 3: Temporal Constraints

```python
# Temporal weights (tau=30s)
temp_weight(0, 1) = exp(-2/30) = 0.935  # 2s apart
temp_weight(0, 2) = exp(-5/30) = 0.846  # 5s apart
temp_weight(0, 3) = exp(-7/30) = 0.791  # 7s apart
temp_weight(1, 2) = exp(-3/30) = 0.905  # 3s apart
temp_weight(1, 3) = exp(-5/30) = 0.846  # 5s apart
temp_weight(2, 3) = exp(-2/30) = 0.935  # 2s apart

# Apply temporal bias
temporal_sim(0, 1) = 0.93 × (0.7 + 0.3×0.935) = 0.91
temporal_sim(0, 2) = 0.21 × (0.7 + 0.3×0.846) = 0.20
temporal_sim(0, 3) = 0.19 × (0.7 + 0.3×0.791) = 0.18
temporal_sim(1, 2) = 0.18 × (0.7 + 0.3×0.905) = 0.18
temporal_sim(1, 3) = 0.20 × (0.7 + 0.3×0.846) = 0.19
temporal_sim(2, 3) = 0.94 × (0.7 + 0.3×0.935) = 0.92
```

---

### Step 4: Overlap-Aware Scoring

```python
# All single-speaker sources (weight=1.0)
final_sim(0, 1) = 0.91 × 1.0 = 0.91
final_sim(0, 2) = 0.20 × 1.0 = 0.20
final_sim(0, 3) = 0.18 × 1.0 = 0.18
final_sim(1, 2) = 0.18 × 1.0 = 0.18
final_sim(1, 3) = 0.19 × 1.0 = 0.19
final_sim(2, 3) = 0.92 × 1.0 = 0.92
```

---

### Step 5: Affinity Matrix Construction

```python
A = [
  [1.00, 0.91, 0.20, 0.18],  # Embedding 0 vs all
  [0.91, 1.00, 0.18, 0.19],  # Embedding 1 vs all
  [0.20, 0.18, 1.00, 0.92],  # Embedding 2 vs all
  [0.18, 0.19, 0.92, 1.00]   # Embedding 3 vs all
]
```

---

### Step 6: Sparsification (Optional)

```python
# Top-K neighbors (K=2)
Embedding 0: Keep neighbors 1 (0.91), 2 (0.20)
Embedding 1: Keep neighbors 0 (0.91), 3 (0.19)
Embedding 2: Keep neighbors 3 (0.92), 0 (0.20)
Embedding 3: Keep neighbors 2 (0.92), 1 (0.19)

A_sparse = [
  [1.00, 0.91, 0.20, 0.00],
  [0.91, 1.00, 0.00, 0.19],
  [0.20, 0.00, 1.00, 0.92],
  [0.00, 0.19, 0.92, 1.00]
]
```

---

### Final Output

```python
Affinity matrix ready for clustering:
  - Clear block structure (Doctor: 0-1, Patient: 2-3)
  - High intra-speaker similarity (>0.9)
  - Low inter-speaker similarity (<0.2)
  - Sparsified for stability

Next: Module 5 (Clustering) will use this matrix
```

---

## 4.16 Key Terminology

| Term | Simple | Technical | Why It Matters |
|------|--------|-----------|----------------|
| **Affinity matrix** | Similarity table | N×N matrix of pairwise similarities | Input to clustering |
| **Cosine similarity** | Angular distance | Dot product of unit vectors | Primary metric for LASPA embeddings |
| **PLDA** | Probabilistic scorer | Probabilistic Linear Discriminant Analysis | Secondary/optional for DISPLACE |
| **Confidence weighting** | Reliability scaling | Weight based on embedding source | Down-weight noisy embeddings |
| **Temporal bias** | Time-aware scoring | Boost adjacent, down-weight distant | Preserve conversation flow |
| **Sparsification** | Noise removal | Keep only strong connections | Improve clustering stability |
| **Overlap-aware** | Source-based weighting | Different weights for separated vs clean | Avoid artifact corruption |

---

## 4.17 Success Criteria

Module 4 is successful if:

1. **Language-agnostic**: Same speaker, different language → high similarity (>0.8)
2. **Speaker-discriminative**: Different speakers → low similarity (<0.3)
3. **Confidence-aware**: Low-confidence embeddings down-weighted
4. **Temporally-coherent**: Adjacent segments boosted appropriately
5. **Overlap-safe**: Separated-separated pairs heavily down-weighted
6. **Block structure**: Affinity matrix shows clear speaker blocks
7. **Sparse**: Top-K or threshold sparsification applied
8. **Symmetric**: A[i,j] = A[j,i] for all i, j

---

## 4.18 Summary

> **Module 4 converts LASPA-trained speaker embeddings into a confidence-weighted, temporally-aware, overlap-safe affinity matrix using cosine similarity as the primary metric, producing a clean similarity graph ready for clustering without performing any grouping decisions.**

---

**Next Module**: MODULE 5 — Clustering (Spectral + VBx refinement, overlap-aware)

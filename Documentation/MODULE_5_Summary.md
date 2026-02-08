# MODULE 5 — CLUSTERING (Summary)

*(ECAPA-TDNN + LASPA embeddings, spectral clustering + VBx)*

---

## Purpose

Group speaker embeddings into speaker clusters where each cluster corresponds to one real speaker, using spectral clustering followed by VBx temporal refinement.

**Input**: Affinity matrix + segment metadata (Module 4)  
**Output**: Segment-level speaker IDs

---

## Why ECAPA-TDNN + LASPA Changes Clustering

| Property | Effect on Clustering |
|----------|---------------------|
| Tight intra-speaker clusters | Easier grouping, cleaner boundaries |
| Angular separation | Spectral methods work well |
| Language invariance | No language-based splits |
| Short-segment stability | Backchannels cluster correctly |
| Non-linear manifold | Spectral > simple k-means |

**Result**: Clean cluster structure, no complicated heuristics needed.

---

## Two-Stage Clustering Approach

```
Stage 1: Spectral Clustering (coarse grouping)
         ↓
Stage 2: VBx Refinement (temporal smoothing)
```

---

## Stage 1: Spectral Clustering

### Why Spectral Clustering?

1. **ECAPA embeddings form non-linear clusters** - Graph-based methods handle this
2. **Cosine affinity graph is strong** - LASPA creates clear structure
3. **No Gaussian assumption** - Works with any similarity metric
4. **Handles variable cluster sizes** - Medical conversations often unbalanced

### How It Works

```
1. Treat affinity matrix as graph (nodes=segments, edges=similarities)
2. Compute normalized graph Laplacian: L = I - D^(-1/2) × A × D^(-1/2)
3. Eigen decomposition → find K smallest eigenvectors
4. K-means in spectral space → cluster labels
```

### Speaker Count Estimation

**Eigen-gap heuristic**:

```
Eigenvalues: [0.0, 0.08, 1.42, 1.50, ...]
                   ↑      ↑
Gaps:        [0.08, 1.34, 0.08, ...]
                   ↑
            Largest gap → K=2 speakers
```

**Bounds**: min=2, max=6-8 (medical conversations)

---

## Confidence-Aware Clustering

**Down-weight edges involving**:
- Medium-confidence segments (×0.85)
- Very short segments <300ms (×0.7)
- PixIT-separated pairs (×0.5-0.8)

**Why**: Clean single-speaker segments should anchor clusters, not noisy separated segments.

---

## Stage 2: VBx Refinement

### What VBx Does

**VBx = Variational Bayesian HMM over speakers**

Models:
- Speaker identity as hidden states
- Embeddings as observations
- **Temporal continuity explicitly**

### Why VBx is Mandatory

Even perfect embeddings:
- Don't know time order
- Treat segments independently

VBx enforces: **"Speakers don't switch every 200ms"**

### How VBx Works

```
HMM structure:
  States: {Speaker 0, Speaker 1, ...}
  
  Transition probabilities:
    P(stay) = 0.95  (conservative)
    P(switch) = 0.05
  
  Emission probabilities:
    P(embedding | speaker) ∝ exp(cosine(emb, speaker_mean))

Variational Bayes inference:
  1. Initialize from spectral clustering
  2. E-step: Compute posteriors
  3. M-step: Update speaker means
  4. Repeat until convergence
```

### VBx Fixes

- Merges fragmented clusters
- Smooths timelines
- Fixes short-segment errors (backchannels)
- Enforces conversation structure

---

## Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Affinity Matrix + Metadata                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Confidence Weighting                                   │
│ Down-weight medium-confidence, short segments                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Speaker Count Estimation                               │
│ Eigen-gap heuristic (bounds: 2-8)                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Spectral Clustering                                    │
│ Graph Laplacian → eigenvectors → k-means                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: VBx Initialization                                     │
│ Speaker means from spectral clusters                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: VBx Refinement                                         │
│ Variational Bayes HMM (E-step, M-step, iterate)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Clustered Segments with speaker_id                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Example

**Input**: 4 segments, affinity matrix

```
A = [
  [1.00, 0.91, 0.16, 0.14],  # Segment 0 (Doctor)
  [0.91, 1.00, 0.14, 0.15],  # Segment 1 (Doctor)
  [0.16, 0.14, 1.00, 0.92],  # Segment 2 (Patient)
  [0.14, 0.15, 0.92, 1.00]   # Segment 3 (Patient)
]
```

**Spectral clustering**:

```
Eigenvalues: [0.0, 0.08, 1.42, 1.50]
Eigen-gap → K=2

Spectral labels: [0, 0, 1, 1] ✓
```

**VBx refinement**:

```
Initialize speaker means
Run Variational Bayes
Converged labels: [0, 0, 1, 1] ✓
```

**Output**:

```python
[
  {"segment_id": 0, "start_time": 0.0, "end_time": 2.0, "speaker_id": 0},
  {"segment_id": 1, "start_time": 2.0, "end_time": 4.0, "speaker_id": 0},
  {"segment_id": 2, "start_time": 5.0, "end_time": 7.0, "speaker_id": 1},
  {"segment_id": 3, "start_time": 7.0, "end_time": 9.0, "speaker_id": 1}
]
```

---

## What Module 5 Does NOT Do

❌ Overlap resolution (Module 6)  
❌ RTTM generation (Module 6)  
❌ Resegmentation (Module 6)  
❌ Timeline reconstruction (Module 6)

**Module 5 only**: Assigns speaker IDs to segments.

---

## Key Advantages for DISPLACE-2026

✅ **ECAPA-TDNN optimized** - Spectral clustering handles non-linear manifolds  
✅ **LASPA respected** - Language-agnostic clustering  
✅ **Medical conversation aware** - Conservative VBx priors for turn structure  
✅ **Confidence-aware** - Down-weights noisy segments  
✅ **SOTA approach** - Spectral + VBx used in modern systems

---

## Summary

Spectral clustering groups ECAPA-TDNN + LASPA embeddings into coarse speaker clusters using graph structure, then VBx refines these clusters by enforcing temporal continuity and conversation structure, producing consistent speaker IDs for each segment.

---

**Next**: MODULE 6 — Resegmentation, Overlap Assignment & Timeline Reconstruction

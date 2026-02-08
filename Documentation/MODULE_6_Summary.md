# MODULE 6 — RESEGMENTATION & TIMELINE RECONSTRUCTION (Summary)

*(Boundary refinement, overlap assignment, temporal smoothing)*

---

## Purpose

Convert clustered segments into a clean, continuous speaker timeline with accurate boundaries, correctly assigned overlaps, and merged short fragments.

**Input**: Clustered segments (Module 5) + frame activity (Module 2)  
**Output**: Time-aligned speaker turns with overlap handling

---

## Why Module 6 is Needed

**After Module 5**:
✅ Correct speaker groups  
❌ Rough boundaries  
❌ Overlap ambiguity  
❌ Fragmented short segments  
❌ Temporal inconsistencies

**Module 6 is where DER drops sharply** (from ~15-20% to ~8-12%).

---

## Five-Step Refinement Process

```
Step 1: Boundary Realignment
Step 2: Short-Segment Cleanup
Step 3: Overlap Assignment
Step 4: Temporal Consistency
Step 5: Timeline Construction
```

---

## Step 1: Boundary Realignment

**Problem**: Clustering operates on segments → coarse boundaries

**Solution**: Frame-level resegmentation

```
For each boundary:
  1. Search ±100ms window
  2. Find speech transition using SAD probabilities
  3. Snap boundary to largest change
  
Example:
  Original: 2.0s
  SAD change at 1.95s (speech → silence)
  Refined: 1.95s ✓
```

**DER improvement**: ~2-3% absolute

---

## Step 2: Short-Segment Cleanup

**Problem**: Medical dialogue has tiny fragments ("mm-hmm", "okay")

**Solution**: Attach to temporal neighbors

```
If segment < 250ms:
  1. Check neighbors (previous, next)
  2. If both same speaker → reassign
  3. Else choose longer neighbor
  
Example:
  [0-5s]: Speaker 0 (Doctor)
  [5.0-5.15s]: Speaker 1 (150ms) ❌
  [5.2-8s]: Speaker 1 (Patient)
  
  Previous ≠ Next, but Next same as current
  → Keep as Speaker 1 ✓
```

**DER improvement**: ~2-3% absolute

---

## Step 3: Overlap Assignment

**Problem**: Overlap frames not assigned to specific speakers

### Strategy A: PixIT-Separated Streams

```
1. Extract embeddings from separated streams
2. Compare to speaker cluster means
3. Assign each stream to best-matching speaker

Example:
  stream_A → cos(emb_A, Speaker 0) = 0.93 ✓
  stream_B → cos(emb_B, Speaker 1) = 0.94 ✓
  
  Overlap: {Speaker 0, Speaker 1}
```

### Strategy B: Temporal Context

```
1. Find segments adjacent to overlap
2. Identify top-2 most likely speakers
3. Assign overlap to those speakers

Example:
  Before overlap: Speaker 0
  After overlap: Speaker 1
  
  Overlap: {Speaker 0, Speaker 1}
```

**DER improvement**: ~3-5% absolute

---

## Step 4: Temporal Consistency

**Problem**: Rapid speaker switching artifacts

**Solution**: Minimum turn duration (200ms)

```
If turn < 200ms:
  Check neighbors
  If same speaker before and after → merge
  Else keep (real interruption)
  
Example:
  [10.0-10.2s]: Speaker 0
  [10.2-10.35s]: Speaker 1 (150ms) ❌
  [10.35-12.0s]: Speaker 0
  
  Same speaker before and after
  → Merge: [10.0-12.0s]: Speaker 0 ✓
```

**Preserves real interruptions** when neighbors differ.

---

## Step 5: Timeline Construction

**Merge adjacent same-speaker segments**:

```
Before:
  [0-2s]: Speaker 0
  [2-4s]: Speaker 0
  [4-6s]: Speaker 0
  
After:
  [0-6s]: Speaker 0 ✓
```

**Handle overlaps** (multiple entries per time range):

```
Timeline:
  [7.0-7.2s]: Speaker 0 (overlap)
  [7.0-7.2s]: Speaker 1 (overlap)
  
Representation: Two entries, same time range ✓
```

---

## Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Clustered Segments + Frame Activity                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Boundary Realignment (±100ms, SAD-based)              │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Short-Segment Cleanup (<250ms → neighbors)            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Overlap Assignment (PixIT or temporal context)        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Temporal Consistency (min 200ms turns)                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Timeline Construction (merge + overlap handling)      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Final Speaker Timeline (overlap-aware, refined)       │
└─────────────────────────────────────────────────────────────────┘
```

---

## Example

**Input** (from Module 5):

```
[0.0-2.0s]: Speaker 0
[2.0-5.0s]: Speaker 0
[5.0-5.15s]: Speaker 1 (150ms, short)
[5.2-7.0s]: Speaker 1
[7.0-7.2s]: Overlap (both speakers)
[7.2-10.0s]: Speaker 0
```

**After Module 6**:

```
[0.0-5.0s]: Speaker 0 (merged)
[5.0-5.15s]: Speaker 1 (kept, justified)
[5.2-7.0s]: Speaker 1
[7.0-7.2s]: Speaker 0 (overlap)
[7.0-7.2s]: Speaker 1 (overlap)
[7.2-10.0s]: Speaker 0
```

**DER**: ~5-8% (excellent)

---

## DER Improvement Breakdown

| Step | DER Reduction |
|------|---------------|
| Boundary realignment | ~2-3% |
| Short-segment cleanup | ~2-3% |
| Overlap assignment | ~3-5% |
| Temporal consistency | ~1-2% |
| **Total** | **~8-13%** |

**Module 6 is critical** for competitive DISPLACE-2026 performance.

---

## What Module 6 Does NOT Do

❌ Clustering (Module 5)  
❌ Embedding extraction (Module 3)  
❌ Ensemble fusion (Module 7)  
❌ RTTM formatting (Module 8)

**Module 6 only**: Refines timeline from clusters.

---

## Medical Conversation Handling

| Challenge | Module 6 Solution |
|-----------|-------------------|
| Rapid turn-taking | Temporal consistency |
| Frequent overlaps | PixIT-aware assignment |
| Soft patient speech | SAD-based realignment |
| Short backchannels | Short-segment cleanup |

---

## Summary

Module 6 refines clustered segments into a smooth speaker timeline by realigning boundaries to speech transitions, attaching short segments to neighbors, assigning overlaps using PixIT separation or temporal context, enforcing minimum turn durations, and merging adjacent same-speaker segments, reducing DER by ~8-13% absolute.

---

**Next**: MODULE 7 — Ensemble & Fusion or MODULE 8 — Output Formatting & Scoring

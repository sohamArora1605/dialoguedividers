# MODULE 7 — ENSEMBLE & FUSION (Summary)

*(DOVER-Lap voting, multi-system combination)*

---

## Purpose

Combine outputs from multiple diarization systems into a single, more accurate result by exploiting complementary strengths through voting-based fusion.

**Input**: Multiple speaker timelines from different systems  
**Output**: One fused speaker timeline (overlap-aware)

---

## Why Ensemble is Required

**Single systems fail differently**:

| System | Strengths | Weaknesses |
|--------|-----------|------------|
| ECAPA + Spectral + VBx | Clean single-speaker regions | May miss short overlaps |
| PixIT-based | Excellent overlap detection | Weaker boundaries |
| Powerset-based | Strong on short interruptions | Limited scalability |

**Ensemble advantage**: ~2-4% absolute DER improvement

**DISPLACE-2024 evidence**:
- 1st place: 5-system ensemble, DER = 8.2%
- 3rd place: Single system, DER = 12.4%

---

## DOVER-Lap Algorithm

**DOVER-Lap = Diarization Output Voting Error Reduction with Overlap**

### Five Steps

```
1. Speaker Alignment (Hungarian algorithm)
2. Frame-Level Voting Grid (10-20ms frames)
3. Weighted Voting (system weights)
4. Speaker Selection (threshold)
5. Timeline Conversion
```

---

## Step 1: Speaker Alignment

**Problem**: Different systems use different speaker IDs

**Solution**: Hungarian algorithm on temporal overlap

```
System 1: [Speaker 0, Speaker 1]
System 2: [Speaker A, Speaker B]

Compute overlap matrix:
           Spk 0  Spk 1
Spk A       4.8s   0s
Spk B       0s     4.8s

Hungarian matching:
  A → 0, B → 1 ✓
```

---

## Step 2: Voting Grid

**Create frame-level votes** (10ms frames):

```
Frame 0 (0-10ms):
  System 1: Speaker 0 = 1
  System 2: Speaker 0 = 1
  System 3: Speaker 0 = 1
  
  Votes: [3, 0] (unanimous)

Frame 5 (50-60ms, overlap):
  System 1: [1, 1] (overlap)
  System 2: [1, 1] (overlap)
  System 3: [1, 0] (no overlap)
  
  Votes: [3, 2] (2/3 detect overlap)
```

---

## Step 3: Weighted Voting

**Apply system weights** (tuned on dev set):

```
Example weights:
  System 1 (ECAPA): 0.42
  System 2 (PixIT): 0.33
  System 3 (Powerset): 0.25

Weighted votes = votes × weights

Frame 5:
  Speaker 0: 3 × [0.42, 0.33, 0.25] = 2.5
  Speaker 1: 2 × [0.42, 0.33, 0.25] = 1.58
```

---

## Step 4: Speaker Selection

**Threshold voting scores**:

```
Normalized votes:
  Speaker 0: 2.5 / 4.08 = 0.61
  Speaker 1: 1.58 / 4.08 = 0.39

Threshold: 0.5

Active:
  Speaker 0: 0.61 >= 0.5 ✓
  Speaker 1: 0.39 < 0.5 ✗

Result: Speaker 0 only (no overlap)
```

**Lower threshold (0.3-0.4) for overlap detection**

---

## Step 5: Timeline Conversion

**Convert frames to timeline**:

```
Merge consecutive frames with same speakers
Handle overlaps (multiple entries per time range)

Output:
  [0-5s]: Speaker 0
  [5-7s]: Speaker 1
  [7-7.2s]: Speaker 0 (overlap)
  [7-7.2s]: Speaker 1 (overlap)
  [7.2-10s]: Speaker 0
```

---

## Weighting Strategy

**Not all systems equally reliable**:

```
Uniform weights (1/3 each):
  Fused DER = 9.2%

Optimized weights (0.5, 0.3, 0.2):
  Fused DER = 8.1% ✓ (1.1% improvement)
```

**Adaptive weighting** (optional):
- Boost PixIT in overlap frames (×1.5)
- Boost main system in single-speaker frames (×1.2)

---

## What Module 7 Does NOT Do

❌ Embedding extraction  
❌ Clustering  
❌ Resegmentation  
❌ Threshold tuning

**Module 7 only**: Fuses system outputs via voting.

---

## Summary

DOVER-Lap fuses multiple diarization systems by aligning speakers via Hungarian algorithm, creating frame-level voting grids, applying tuned system weights, selecting speakers based on vote thresholds, and converting to a final overlap-aware timeline, achieving 2-4% DER improvement over single systems.

---

**Next**: MODULE 8 — Output Formatting & Scoring

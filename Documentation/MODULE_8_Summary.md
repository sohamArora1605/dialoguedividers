# MODULE 8 — OUTPUT FORMATTING & SCORING (Summary)

*(RTTM generation, DER computation, parameter tuning)*

---

## Purpose

Convert final diarization timeline into challenge-compliant RTTM output, compute evaluation metrics, and tune system parameters to minimize DER.

**Input**: Final speaker timeline (Module 7 or 6)  
**Output**: RTTM files + DER scores + optimized parameters

---

## RTTM Format

**RTTM = Rich Transcription Time Marked**

```
SPEAKER <file_id> 1 <start> <duration> <NA> <NA> <speaker_id> <NA> <NA>

Example:
SPEAKER file_001 1 0.00 5.00 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER file_001 1 7.00 0.20 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER file_001 1 7.00 0.20 <NA> <NA> speaker_1 <NA> <NA>
```

**Rules**:
- Times in seconds (2 decimals)
- Multiple lines for overlaps (same time, different speakers)
- No silence segments

---

## Post-Processing Before RTTM

**Three cleanup steps**:

### 1. Remove Micro-Segments

```
Remove segments <50ms (jitter)

Before: [5.0-5.05s] (50ms)
After: Removed ✓
```

### 2. Merge Adjacent Segments

```
Merge same-speaker segments with gap <100ms

Before:
  [0-5s]: Speaker 0
  [5.1-7s]: Speaker 0

After:
  [0-7s]: Speaker 0 ✓
```

### 3. Snap Boundaries

```
Round to 10ms grid

Before: 2.037s
After: 2.04s ✓
```

**Effect**: ~0.5-1% DER improvement

---

## DER Computation

**DER = Diarization Error Rate**

```
DER = (Missed Speech + False Alarm + Speaker Confusion) / Total Speech

Components:
  - Missed Speech: Ground truth speech not detected
  - False Alarm: System speech not in ground truth
  - Speaker Confusion: Wrong speaker label
```

**DISPLACE-2026 rules**:
- **Collar**: 0 ms (strict boundaries)
- **Overlaps**: Included (fully scored)
- **Tool**: pyannote.metrics

---

## DER Breakdown Example

```
DER: 10.5%
  - Missed speech: 3.2%
  - False alarm: 2.1%
  - Speaker confusion: 5.2%

Interpretation:
  - High confusion → clustering issue (Module 5)
  - High miss → SAD too conservative (Module 1)
  - High FA → SAD too aggressive (Module 1)
```

---

## Parameter Tuning

**Key parameters to optimize on dev set**:

| Module | Parameter | Range |
|--------|-----------|-------|
| Module 1 | SAD threshold | 0.3 - 0.7 |
| Module 2 | Min segment duration | 0.2 - 0.5s |
| Module 5 | VBx transition prior | 0.90 - 0.98 |
| Module 6 | Min turn duration | 0.15 - 0.3s |
| Module 7 | System weights | 0.0 - 1.0 |

**Tuning strategy**:

```
Grid search on dev set:
  1. Generate parameter combinations
  2. Run pipeline with each combination
  3. Compute DER
  4. Select best parameters

Example:
  Tested: 243 combinations
  Best DER: 8.9%
  Best params: {sad: 0.5, vbx: 0.95, ...}
```

---

## Tuning Priority

**Minimize speaker confusion first**:

```
Scenario A (high confusion):
  DER: 12.5% (Confusion: 9.0%) ❌

Scenario B (low confusion):
  DER: 9.5% (Confusion: 3.5%) ✓

Scenario B preferred:
  - Speaker identity more important
  - Confusion harder to fix post-hoc
```

**Priority order**:
1. Clustering (Module 5) → reduce confusion
2. Embeddings (Module 3) → improve discrimination
3. SAD/Segmentation (Modules 1-2) → balance miss/FA
4. Fusion (Module 7) → final optimization

---

## Error Analysis

**Identify failure patterns**:

| Error Pattern | Root Cause | Fix |
|---------------|------------|-----|
| High confusion in short segments | Embedding quality | Module 3: increase min duration |
| Missed overlaps | Segmentation | Module 2: tune PixIT |
| Boundary errors (±100ms) | Coarse boundaries | Module 6: improve realignment |
| Speaker switching artifacts | Insufficient smoothing | Module 5: increase VBx prior |

**Iterative improvement**:

```
1. Run on dev set
2. Analyze errors
3. Identify dominant pattern
4. Adjust module parameters
5. Validate improvement
6. Repeat
```

---

## Final Outputs

### 1. RTTM Files

```
output/
  file_001.rttm
  file_002.rttm
  ...
```

### 2. DER Report

```
Overall DER: 9.2%
  - Missed: 2.5%
  - FA: 1.8%
  - Confusion: 4.9%

Per-file:
  file_001: 8.5%
  file_002: 10.1%
  ...
```

### 3. Optimized Parameters

```python
{
  "sad_threshold": 0.5,
  "min_segment_duration": 0.3,
  "vbx_transition_prior": 0.95,
  "min_turn_duration": 0.25,
  "dover_weights": [0.5, 0.3, 0.2]
}
```

---

## What Module 8 Does NOT Do

❌ Learning/training  
❌ Clustering  
❌ Fusion logic  
❌ Embedding extraction

**Module 8 only**: Evaluation, formatting, parameter optimization.

---

## Summary

Module 8 generates challenge-compliant RTTM output by post-processing the timeline (removing micro-segments, merging adjacent segments, snapping boundaries), computing DER and error components using pyannote.metrics with DISPLACE-2026 rules (no collar, overlaps included), analyzing errors to identify failure patterns, and tuning system parameters on the dev set via grid search to minimize DER with priority on reducing speaker confusion.

---

**PIPELINE COMPLETE**: All modules (0-8) documented for DISPLACE-2026

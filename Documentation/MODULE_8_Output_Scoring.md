# MODULE 8 — OUTPUT FORMATTING, SCORING & DER OPTIMIZATION

*(DISPLACE-2026 compliant, RTTM generation, metric computation, parameter tuning)*

---

## 8.1 What This Module Does

**Purpose (2-line summary)**  
Converts the final diarization timeline into challenge-compliant RTTM output, computes evaluation metrics (DER, missed speech, false alarm, speaker confusion), and tunes system parameters on the dev set to minimize DER for optimal submission performance.

**Input → Output**
- **Input**: Final speaker timeline (from Module 7 or Module 6)
- **Output**: RTTM files + DER scores + error analysis

---

## 8.2 RTTM Output Formatting (Challenge-Compliant)

### 8.2.1 What is RTTM?

**RTTM = Rich Transcription Time Marked**

Standard format for diarization output in NIST evaluations, DIHARD, DISPLACE, VoxConverse.

**Format specification**:

```
SPEAKER <file_id> 1 <start_time> <duration> <NA> <NA> <speaker_id> <NA> <NA>
```

**Field meanings**:

| Field | Value | Description |
|-------|-------|-------------|
| Type | `SPEAKER` | Always "SPEAKER" for diarization |
| File ID | `file_001` | Audio file identifier |
| Channel | `1` | Always "1" (mono channel) |
| Start time | `0.00` | Segment start (seconds, 2 decimals) |
| Duration | `2.50` | Segment duration (seconds, 2 decimals) |
| Orthography | `<NA>` | Not used in diarization |
| Speaker type | `<NA>` | Not used in diarization |
| Speaker ID | `speaker_0` | Speaker label |
| Confidence | `<NA>` | Not used in diarization |
| Signal lookahead | `<NA>` | Not used in diarization |

### 8.2.2 RTTM Generation from Timeline

**Algorithm**:

```python
def timeline_to_rttm(timeline, file_id, output_path):
    """
    Convert speaker timeline to RTTM file.
    
    Args:
        timeline: [(start, end, speaker_id), ...]
        file_id: Audio file identifier
        output_path: Path to output RTTM file
    """
    with open(output_path, 'w') as f:
        for start, end, speaker in timeline:
            duration = end - start
            
            # Format: SPEAKER file_id 1 start duration <NA> <NA> speaker_id <NA> <NA>
            line = f"SPEAKER {file_id} 1 {start:.2f} {duration:.2f} <NA> <NA> speaker_{speaker} <NA> <NA>\n"
            f.write(line)
```

**Example**:

```python
# Input timeline
timeline = [
  (0.0, 5.0, 0),
  (5.0, 7.0, 1),
  (7.0, 7.2, 0),  # Overlap
  (7.0, 7.2, 1),  # Overlap
  (7.2, 10.0, 0)
]

# Output RTTM
"""
SPEAKER file_001 1 0.00 5.00 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER file_001 1 5.00 2.00 <NA> <NA> speaker_1 <NA> <NA>
SPEAKER file_001 1 7.00 0.20 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER file_001 1 7.00 0.20 <NA> <NA> speaker_1 <NA> <NA>
SPEAKER file_001 1 7.2 2.80 <NA> <NA> speaker_0 <NA> <NA>
"""
```

**Key rules**:
- ✅ Multiple RTTM lines allowed for overlaps (same time range, different speakers)
- ✅ Times in seconds with 2 decimal precision
- ❌ No silence segments written (only speech)
- ❌ No gaps required between segments

---

## 8.3 Post-Processing Before RTTM (Important)

### 8.3.1 Why Post-Processing?

**Raw timeline may have artifacts**:

```
Issues:
  - Micro-segments (<50ms) from jitter
  - Tiny gaps (<100ms) between same-speaker segments
  - Boundary quantization errors

These inflate DER without representing real errors
```

### 8.3.2 Final Cleanup Steps

**Step 1: Remove micro-segments**

```python
def remove_micro_segments(timeline, min_duration=0.05):
    """
    Remove very short segments (likely jitter).
    
    Args:
        timeline: [(start, end, speaker), ...]
        min_duration: Minimum segment duration (seconds)
    
    Returns:
        Cleaned timeline
    """
    cleaned = []
    
    for start, end, speaker in timeline:
        duration = end - start
        if duration >= min_duration:
            cleaned.append((start, end, speaker))
        # else: discard micro-segment
    
    return cleaned
```

**Step 2: Merge adjacent same-speaker segments**

```python
def merge_adjacent_segments(timeline, max_gap=0.1):
    """
    Merge same-speaker segments with small gaps.
    
    Args:
        timeline: [(start, end, speaker), ...]
        max_gap: Maximum gap to merge (seconds)
    
    Returns:
        Merged timeline
    """
    if not timeline:
        return []
    
    # Sort by start time
    timeline = sorted(timeline, key=lambda x: x[0])
    
    merged = [timeline[0]]
    
    for start, end, speaker in timeline[1:]:
        prev_start, prev_end, prev_speaker = merged[-1]
        
        gap = start - prev_end
        
        if speaker == prev_speaker and gap <= max_gap:
            # Merge: extend previous segment
            merged[-1] = (prev_start, end, speaker)
        else:
            # Different speaker or gap too large
            merged.append((start, end, speaker))
    
    return merged
```

**Step 3: Snap boundaries to grid**

```python
def snap_boundaries(timeline, grid_size=0.01):
    """
    Snap boundaries to time grid (reduces quantization noise).
    
    Args:
        timeline: [(start, end, speaker), ...]
        grid_size: Grid size (seconds)
    
    Returns:
        Snapped timeline
    """
    snapped = []
    
    for start, end, speaker in timeline:
        # Round to nearest grid point
        start_snapped = round(start / grid_size) * grid_size
        end_snapped = round(end / grid_size) * grid_size
        
        # Ensure duration > 0
        if end_snapped > start_snapped:
            snapped.append((start_snapped, end_snapped, speaker))
    
    return snapped
```

**Combined cleanup**:

```python
def final_cleanup(timeline):
    """
    Apply all cleanup steps before RTTM generation.
    
    Args:
        timeline: Raw speaker timeline
    
    Returns:
        Cleaned timeline ready for RTTM
    """
    # Step 1: Remove micro-segments
    timeline = remove_micro_segments(timeline, min_duration=0.05)
    
    # Step 2: Merge adjacent same-speaker segments
    timeline = merge_adjacent_segments(timeline, max_gap=0.1)
    
    # Step 3: Snap boundaries
    timeline = snap_boundaries(timeline, grid_size=0.01)
    
    return timeline
```

**Effect on DER**:

```
Before cleanup:
  - 50 micro-segments (<50ms)
  - 30 tiny gaps (<100ms)
  - DER: 10.5%

After cleanup:
  - Micro-segments removed
  - Gaps merged
  - DER: 9.8% ✓ (0.7% improvement)
```

---

## 8.4 DER Computation (DISPLACE-Compliant)

### 8.4.1 What is DER?

**DER = Diarization Error Rate**

**Definition**:

```
DER = (Missed Speech + False Alarm + Speaker Confusion) / Total Speech Time

Where:
  - Missed Speech: Ground truth speech not detected
  - False Alarm: System speech not in ground truth
  - Speaker Confusion: Speech detected but wrong speaker
  - Total Speech Time: Total duration of speech in ground truth
```

### 8.4.2 DISPLACE-2026 Scoring Rules

**Critical rules**:

| Rule | Value | Impact |
|------|-------|--------|
| **Collar** | 0 ms | No forgiveness for boundary errors |
| **Overlap handling** | Included | Overlap errors fully counted |
| **Minimum segment** | None | All segments scored |
| **UEM file** | Required | Specifies scored regions |

**No collar** means boundary errors are heavily penalized.

### 8.4.3 DER Computation with pyannote.metrics

```python
from pyannote.metrics.diarization import DiarizationErrorRate

def compute_der(hypothesis_rttm, reference_rttm, uem_file=None):
    """
    Compute DER using pyannote.metrics.
    
    Args:
        hypothesis_rttm: System output RTTM file
        reference_rttm: Ground truth RTTM file
        uem_file: Optional UEM file (scored regions)
    
    Returns:
        DER components: {der, miss, fa, confusion}
    """
    # Load RTTM files
    from pyannote.database.util import load_rttm
    
    hypothesis = load_rttm(hypothesis_rttm)
    reference = load_rttm(reference_rttm)
    
    # Initialize metric
    metric = DiarizationErrorRate(collar=0.0, skip_overlap=False)
    
    # Compute DER
    for uri in reference:
        ref = reference[uri]
        hyp = hypothesis.get(uri, None)
        
        if hyp is None:
            # No hypothesis for this file
            continue
        
        # Compute error
        metric(ref, hyp, uem=uem_file)
    
    # Get results
    der = metric.report()
    
    return {
        "der": der["diarization error rate"],
        "miss": der["missed detection"],
        "fa": der["false alarm"],
        "confusion": der["speaker confusion"]
    }
```

### 8.4.4 Error Breakdown

**Understanding DER components**:

```
Example DER report:
  DER: 10.5%
    - Missed speech: 3.2%
    - False alarm: 2.1%
    - Speaker confusion: 5.2%

Interpretation:
  - Missed speech: SAD too conservative
  - False alarm: SAD too aggressive
  - Speaker confusion: Clustering errors (main issue)
```

**Prioritization**:

```
1. Speaker confusion (highest priority)
   → Improve clustering (Module 5)
   → Better embeddings (Module 3)

2. Missed speech
   → Improve SAD sensitivity (Module 1)
   → Better overlap detection (Module 2)

3. False alarm
   → Reduce SAD false positives (Module 1)
   → Better silence detection
```

---

## 8.5 Threshold & Parameter Tuning

### 8.5.1 What to Tune

**Key parameters to optimize on dev set**:

| Module | Parameter | Typical Range | Impact |
|--------|-----------|---------------|--------|
| Module 1 (SAD) | Activation threshold | 0.3 - 0.7 | Miss vs FA trade-off |
| Module 2 (Seg) | Minimum segment duration | 0.2 - 0.5s | Fragmentation |
| Module 3 (Emb) | Minimum embedding duration | 0.25 - 0.5s | Embedding quality |
| Module 5 (Clust) | Spectral clustering threshold | Auto (eigen-gap) | Speaker count |
| Module 5 (VBx) | Speaker change prior | 0.90 - 0.98 | Temporal smoothing |
| Module 6 (Reseg) | Minimum turn duration | 0.15 - 0.3s | Switching artifacts |
| Module 7 (Fusion) | System weights | 0.0 - 1.0 | Ensemble balance |

### 8.5.2 Tuning Strategy

**Grid search on dev set**:

```python
def tune_parameters(dev_set, param_grid):
    """
    Tune parameters to minimize DER on dev set.
    
    Args:
        dev_set: Development set with ground truth
        param_grid: Dictionary of parameter ranges
    
    Returns:
        Best parameters
    """
    best_params = None
    best_der = float('inf')
    
    # Generate all parameter combinations
    from itertools import product
    
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for values in product(*param_values):
        params = dict(zip(param_names, values))
        
        # Run pipeline with these parameters
        outputs = run_pipeline(dev_set, params)
        
        # Compute DER
        der = compute_der(outputs, dev_set.ground_truth)
        
        if der["der"] < best_der:
            best_der = der["der"]
            best_params = params
    
    return best_params, best_der
```

**Example parameter grid**:

```python
param_grid = {
    "sad_threshold": [0.4, 0.5, 0.6],
    "min_segment_duration": [0.25, 0.3, 0.35],
    "vbx_transition_prior": [0.92, 0.95, 0.97],
    "min_turn_duration": [0.2, 0.25, 0.3],
    "dover_weights": [
        [0.5, 0.3, 0.2],
        [0.6, 0.25, 0.15],
        [0.4, 0.35, 0.25]
    ]
}

# Total combinations: 3 × 3 × 3 × 3 × 3 = 243
```

### 8.5.3 Prioritization: Speaker Confusion First

**Goal**: Minimize speaker confusion before optimizing miss/FA

**Why**:

```
Scenario A (high confusion):
  DER: 12.5%
    - Miss: 2.0%
    - FA: 1.5%
    - Confusion: 9.0% ❌ (main problem)

Scenario B (low confusion):
  DER: 9.5%
    - Miss: 3.5%
    - FA: 2.5%
    - Confusion: 3.5% ✓ (acceptable)

Scenario B is better for DISPLACE:
  - Speaker identity more important than exact boundaries
  - Confusion errors are harder to fix post-hoc
```

**Tuning priority**:

```
1. Optimize clustering parameters (Module 5)
   → Minimize speaker confusion

2. Optimize embedding parameters (Module 3)
   → Improve speaker discrimination

3. Optimize SAD/segmentation (Modules 1-2)
   → Balance miss vs FA

4. Optimize fusion weights (Module 7)
   → Final DER reduction
```

---

## 8.6 Error Analysis Loop (Critical)

### 8.6.1 Why Error Analysis?

**Systematic improvement requires understanding failure modes**:

```
Without error analysis:
  - Tune parameters blindly
  - May improve one metric, hurt another
  - No insight into root causes

With error analysis:
  - Identify specific failure patterns
  - Target improvements to modules
  - Validate fixes on dev set
```

### 8.6.2 Error Analysis Workflow

```python
def analyze_errors(hypothesis_rttm, reference_rttm):
    """
    Analyze diarization errors to identify failure modes.
    
    Args:
        hypothesis_rttm: System output
        reference_rttm: Ground truth
    
    Returns:
        Error analysis report
    """
    from pyannote.core import Annotation, Timeline
    
    # Load annotations
    hyp = load_rttm(hypothesis_rttm)
    ref = load_rttm(reference_rttm)
    
    errors = {
        "speaker_confusion": [],
        "missed_overlaps": [],
        "false_overlaps": [],
        "short_segment_errors": [],
        "boundary_errors": []
    }
    
    for uri in ref:
        ref_ann = ref[uri]
        hyp_ann = hyp[uri]
        
        # Analyze speaker confusion
        for segment, track in ref_ann.itertracks():
            ref_speaker = track
            
            # Find overlapping hypothesis segments
            hyp_segments = hyp_ann.crop(segment)
            
            if not hyp_segments:
                # Missed speech
                continue
            
            # Check if speaker matches
            hyp_speakers = set(hyp_segments.labels())
            
            if ref_speaker not in hyp_speakers:
                # Speaker confusion
                errors["speaker_confusion"].append({
                    "time": segment,
                    "ref_speaker": ref_speaker,
                    "hyp_speakers": list(hyp_speakers)
                })
        
        # Analyze overlap errors
        ref_overlaps = find_overlaps(ref_ann)
        hyp_overlaps = find_overlaps(hyp_ann)
        
        for overlap in ref_overlaps:
            if not any(overlap.overlaps(h) for h in hyp_overlaps):
                # Missed overlap
                errors["missed_overlaps"].append(overlap)
        
        # ... (more analysis)
    
    return errors
```

### 8.6.3 Diagnostic Feedback to Modules

**Error patterns guide module improvements**:

| Error Pattern | Root Cause | Module to Fix |
|---------------|------------|---------------|
| High speaker confusion in short segments | Embedding quality | Module 3 (increase min duration) |
| Missed overlaps | Segmentation | Module 2 (tune PixIT sensitivity) |
| False overlaps | Over-aggressive overlap detection | Module 2 (increase overlap threshold) |
| Boundary errors (±100-200ms) | Coarse boundaries | Module 6 (improve realignment) |
| Speaker switching artifacts | Insufficient smoothing | Module 5 (increase VBx transition prior) |
| Fragmented speakers | Under-clustering | Module 5 (adjust spectral threshold) |

**Iterative improvement**:

```
1. Run system on dev set
2. Analyze errors
3. Identify dominant error pattern
4. Adjust relevant module parameters
5. Re-run and validate improvement
6. Repeat until DER converges
```

---

## 8.7 What Module 8 Does NOT Do

**Module 8 performs evaluation and formatting only** - it does NOT:

❌ **Learning**: Training models or learning parameters (all modules trained separately)  
❌ **Clustering**: Grouping segments by speaker (Module 5)  
❌ **Fusion logic**: Combining systems (Module 7)  
❌ **Embedding extraction**: Computing speaker representations (Module 3)  
❌ **Segmentation**: Detecting speaker changes (Module 2)

**Module 8 answers**: "How good is the system?" and "What parameters minimize DER?"  
**Module 8 does NOT answer**: "How do we cluster speakers?" or "How do we fuse systems?"

---

## 8.8 Final Outputs

### 8.8.1 RTTM Files (Submission-Ready)

```
output/
  file_001.rttm
  file_002.rttm
  file_003.rttm
  ...
```

**Each RTTM file**:
- Challenge-compliant format
- Overlap-aware (multiple lines per time range)
- Post-processed (cleaned, merged)

### 8.8.2 DER Report

```
Overall DER: 9.2%
  - Missed speech: 2.5%
  - False alarm: 1.8%
  - Speaker confusion: 4.9%

Per-file breakdown:
  file_001: 8.5%
  file_002: 10.1%
  file_003: 9.0%
  ...

Error analysis:
  - Speaker confusion in short segments: 35% of errors
  - Missed overlaps: 25% of errors
  - Boundary errors: 20% of errors
  - Other: 20% of errors
```

### 8.8.3 Optimized Parameters

```python
optimized_params = {
    "sad_threshold": 0.5,
    "min_segment_duration": 0.3,
    "min_embedding_duration": 0.3,
    "vbx_transition_prior": 0.95,
    "min_turn_duration": 0.25,
    "dover_weights": [0.5, 0.3, 0.2]
}
```

---

## 8.9 Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Final Speaker Timeline (Module 7 or 6)                  │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: POST-PROCESSING                            │
│ Remove micro-segments, merge adjacent, snap boundaries         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: RTTM GENERATION                            │
│ Convert timeline to challenge-compliant RTTM format            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: DER COMPUTATION                            │
│ Compute DER, miss, FA, confusion using pyannote.metrics        │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: ERROR ANALYSIS                             │
│ Identify failure patterns, diagnose root causes                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: PARAMETER TUNING                           │
│ Grid search on dev set to minimize DER                         │
│ Prioritize speaker confusion reduction                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: RTTM Files + DER Report + Optimized Parameters        │
│ Submission-ready, fully evaluated                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 8.10 Complete End-to-End Example

**Scenario**: Generating RTTM and computing DER for test file

### Input Timeline

```python
timeline = [
  (0.0, 5.0, 0),
  (5.0, 5.05, 0),  # Micro-segment (50ms)
  (5.1, 7.0, 1),
  (7.0, 7.2, 0),   # Overlap
  (7.0, 7.2, 1),   # Overlap
  (7.2, 10.0, 0)
]
```

---

### Step 1: Post-Processing

```python
# Remove micro-segments (<50ms)
timeline = remove_micro_segments(timeline, min_duration=0.05)
# Result: [5.0-5.05s] removed

# Merge adjacent same-speaker (<100ms gap)
timeline = merge_adjacent_segments(timeline, max_gap=0.1)
# Result: [0-5.0s] and [5.1-7.0s] not merged (different speakers)

# Snap boundaries
timeline = snap_boundaries(timeline, grid_size=0.01)
# Result: boundaries rounded to 10ms grid

Cleaned timeline:
  [0.0-5.0s]: Speaker 0
  [5.1-7.0s]: Speaker 1
  [7.0-7.2s]: Speaker 0 (overlap)
  [7.0-7.2s]: Speaker 1 (overlap)
  [7.2-10.0s]: Speaker 0
```

---

### Step 2: RTTM Generation

```
SPEAKER file_001 1 0.00 5.00 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER file_001 1 5.10 1.90 <NA> <NA> speaker_1 <NA> <NA>
SPEAKER file_001 1 7.00 0.20 <NA> <NA> speaker_0 <NA> <NA>
SPEAKER file_001 1 7.00 0.20 <NA> <NA> speaker_1 <NA> <NA>
SPEAKER file_001 1 7.20 2.80 <NA> <NA> speaker_0 <NA> <NA>
```

---

### Step 3: DER Computation

```python
# Ground truth RTTM
"""
SPEAKER file_001 1 0.00 5.00 <NA> <NA> doctor <NA> <NA>
SPEAKER file_001 1 5.00 7.00 <NA> <NA> patient <NA> <NA>
SPEAKER file_001 1 7.00 0.20 <NA> <NA> doctor <NA> <NA>
SPEAKER file_001 1 7.00 0.20 <NA> <NA> patient <NA> <NA>
SPEAKER file_001 1 7.20 2.80 <NA> <NA> doctor <NA> <NA>
"""

# Compute DER
der_results = compute_der("file_001.rttm", "ground_truth.rttm")

# Results
{
  "der": 0.095,  # 9.5%
  "miss": 0.010,  # 1.0% (gap at 5.0-5.1s)
  "fa": 0.000,    # 0.0%
  "confusion": 0.085  # 8.5% (speaker label mismatch)
}
```

**Note**: Speaker confusion high because speaker IDs don't match (speaker_0 vs doctor). This is expected - DER computation handles speaker permutation.

---

### Step 4: Error Analysis

```python
errors = analyze_errors("file_001.rttm", "ground_truth.rttm")

# Findings
{
  "missed_speech": [
    {"time": (5.0, 5.1), "duration": 0.1}  # Gap between segments
  ],
  "speaker_confusion": [],  # None (after permutation)
  "missed_overlaps": [],    # None
  "boundary_errors": []     # None
}

# Conclusion: Minimal errors, DER mostly from 100ms gap
```

---

### Step 5: Parameter Tuning (Dev Set)

```python
# Tune on full dev set
param_grid = {
    "min_segment_duration": [0.25, 0.3, 0.35],
    "min_turn_duration": [0.2, 0.25, 0.3]
}

best_params, best_der = tune_parameters(dev_set, param_grid)

# Results
{
  "best_params": {
    "min_segment_duration": 0.3,
    "min_turn_duration": 0.25
  },
  "best_der": 0.089  # 8.9% on dev set
}
```

---

## 8.11 Key Terminology

| Term | Simple | Technical | Why It Matters |
|------|--------|-----------|----------------|
| **RTTM** | Output format | Rich Transcription Time Marked | Standard diarization output format |
| **DER** | Error rate | Diarization Error Rate | Primary evaluation metric |
| **Collar** | Boundary forgiveness | Time window around boundaries | DISPLACE uses 0ms (strict) |
| **UEM** | Scored regions | Un-partitioned Evaluation Map | Specifies which regions to score |
| **Speaker confusion** | Wrong speaker label | Speech detected but wrong speaker | Most important DER component |

---

## 8.12 Success Criteria

Module 8 is successful if:

1. **RTTM compliance**: All output files pass format validation
2. **DER computation**: Matches official DISPLACE scorer (±0.1%)
3. **Parameter optimization**: Tuned parameters reduce DER by ≥1% vs defaults
4. **Error analysis**: Identifies top-3 failure modes correctly
5. **Reproducibility**: Same parameters produce same DER (±0.1%)
6. **Processing speed**: RTTM generation + scoring <0.1× real-time

---

## 8.13 Summary

> **Module 8 converts the final speaker timeline into challenge-compliant RTTM output by applying post-processing cleanup (removing micro-segments, merging adjacent segments, snapping boundaries), generating RTTM files with proper formatting, computing DER and error components using pyannote.metrics with DISPLACE-2026 rules (no collar, overlaps included), analyzing errors to identify failure patterns, and tuning system parameters on the dev set to minimize DER with priority on reducing speaker confusion.**

---

**PIPELINE COMPLETE**: All 9 modules (0-8) documented for DISPLACE-2026

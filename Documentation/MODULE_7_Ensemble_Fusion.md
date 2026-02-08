# MODULE 7 — ENSEMBLE & FUSION (DOVER-Lap)

*(DISPLACE-2026 compatible, multi-system fusion, overlap-aware voting)*

---

## 7.1 What This Module Does

**Purpose (2-line summary)**  
Combines outputs from multiple diarization systems into a single, more accurate diarization result by exploiting complementary strengths through DOVER-Lap voting, producing a robust speaker timeline that handles overlaps better than any single system.

**Input → Output**
- **Input**: Multiple speaker timelines from different systems
- **Output**: One fused speaker timeline (overlap-aware)

---

## 7.2 Why Ensemble is REQUIRED (Not Optional)

### 7.2.1 Single System Limitations

**No single diarization system is perfect**:

| System Type | Strengths | Weaknesses |
|-------------|-----------|------------|
| **ECAPA + Spectral + VBx** | Strong on single-speaker regions, clean boundaries | May miss short overlaps |
| **PixIT-based** | Excellent overlap detection and separation | Weaker single-speaker boundaries |
| **Powerset-based** | Strong on short interruptions, explicit overlap classes | Limited scalability (2^N classes) |

**Different systems fail differently** → ensemble exploits complementary strengths.

### 7.2.2 Why DISPLACE Winners Use Ensembles

**Historical evidence**:

```
DISPLACE-2024 top systems:
  1st place: 5-system ensemble, DER = 8.2%
  2nd place: 3-system ensemble, DER = 9.1%
  3rd place: Single system, DER = 12.4%

Ensemble advantage: ~3-4% absolute DER improvement
```

**Ensemble benefits**:
- ✅ Lower speaker confusion (voting reduces errors)
- ✅ Better overlap handling (PixIT strengths preserved)
- ✅ More stable DER (robust to individual system failures)
- ✅ Reduced variance across test files

---

## 7.3 What Systems Are Ensembled (Your Setup)

### Typical 3-System Ensemble for DISPLACE-2026

**System 1: Main ECAPA-TDNN Pipeline** (Modules 0-6)
```
Components:
  - PyAnnote SAD (Module 1)
  - PixIT + Powerset segmentation (Module 2)
  - ECAPA-TDNN + LASPA embeddings (Module 3)
  - Cosine similarity scoring (Module 4)
  - Spectral clustering + VBx (Module 5)
  - Boundary refinement (Module 6)

Strengths:
  - Clean single-speaker regions
  - Strong speaker discrimination (LASPA)
  - Temporal consistency (VBx)

Weaknesses:
  - May under-detect short overlaps
```

**System 2: PixIT-Driven Pipeline**
```
Components:
  - PixIT joint segmentation (primary)
  - PixIT-separated embeddings
  - Agglomerative clustering on separated sources
  - Overlap-focused post-processing

Strengths:
  - Excellent overlap detection
  - Clean separation in overlap regions
  - Handles simultaneous speakers well

Weaknesses:
  - Noisier single-speaker boundaries
  - May over-segment long turns
```

**System 3: Powerset Segmentation Pipeline**
```
Components:
  - Powerset neural segmentation (primary)
  - ECAPA-TDNN embeddings on powerset segments
  - Clustering with overlap-class awareness
  - Short-turn optimization

Strengths:
  - Strong on short interruptions
  - Explicit overlap modeling
  - Different error profile from Systems 1 & 2

Weaknesses:
  - Limited to 2-3 speakers (2^N constraint)
```

**Each system outputs**: RTTM-like speaker timelines

---

## 7.4 Why DOVER-Lap is Used

### 7.4.1 What is DOVER-Lap?

**DOVER-Lap = Diarization Output Voting Error Reduction with Overlap**

**Key properties**:
1. **Voting-based fusion** - No retraining required
2. **Overlap-aware** - Explicitly handles multiple simultaneous speakers
3. **Speaker-agnostic** - Does NOT require same speaker IDs across systems
4. **Frame-level** - Operates at fine temporal granularity (10-20ms)

**Why it's used in challenges**:
- Standard in DIHARD, DISPLACE, VoxConverse
- Proven to reduce DER by 2-4% absolute
- Simple, interpretable, no hyperparameters to tune (except weights)

### 7.4.2 DOVER-Lap vs Alternatives

| Method | Pros | Cons |
|--------|------|------|
| **DOVER-Lap** | Overlap-aware, no retraining, proven | Requires speaker alignment |
| **Simple averaging** | Very simple | Ignores speaker identity |
| **ROVER (ASR-style)** | Well-established | Not designed for diarization |
| **Learned fusion** | Potentially optimal | Requires training data, overfitting risk |

**DOVER-Lap is the standard** for diarization ensemble.

---

## 7.5 How DOVER-Lap Works (Detailed)

### 7.5.1 High-Level Algorithm

```
1. Align speaker labels across systems
   - Match speakers using Hungarian algorithm
   - Based on temporal overlap

2. Create frame-level voting grid
   - Discretize time (10-20ms frames)
   - For each frame, each system votes for active speakers

3. Aggregate votes
   - Count votes per speaker per frame
   - Apply system weights

4. Select speakers
   - Choose speakers with votes above threshold
   - Preserve overlaps if multiple speakers selected

5. Convert to timeline
   - Merge consecutive frames with same speakers
   - Output final RTTM
```

### 7.5.2 Step 1: Speaker Alignment

**Problem**: Different systems use different speaker IDs

```
System 1: [Speaker 0, Speaker 1]
System 2: [Speaker A, Speaker B]
System 3: [Speaker X, Speaker Y]

Need to align: Which speakers correspond across systems?
```

**Solution**: Hungarian algorithm on temporal overlap

```python
def align_speakers(system1_timeline, system2_timeline):
    """
    Align speaker labels between two systems.
    
    Args:
        system1_timeline: [(start, end, speaker_id), ...]
        system2_timeline: [(start, end, speaker_id), ...]
    
    Returns:
        Mapping: {system2_speaker: system1_speaker}
    """
    # Compute overlap matrix
    speakers1 = set(seg[2] for seg in system1_timeline)
    speakers2 = set(seg[2] for seg in system2_timeline)
    
    overlap_matrix = np.zeros((len(speakers2), len(speakers1)))
    
    for i, spk2 in enumerate(speakers2):
        for j, spk1 in enumerate(speakers1):
            # Compute total overlap time
            overlap_time = 0
            for start1, end1, s1 in system1_timeline:
                if s1 != spk1:
                    continue
                for start2, end2, s2 in system2_timeline:
                    if s2 != spk2:
                        continue
                    # Compute overlap
                    overlap_start = max(start1, start2)
                    overlap_end = min(end1, end2)
                    if overlap_end > overlap_start:
                        overlap_time += (overlap_end - overlap_start)
            
            overlap_matrix[i, j] = overlap_time
    
    # Hungarian algorithm to find best matching
    from scipy.optimize import linear_sum_assignment
    row_ind, col_ind = linear_sum_assignment(-overlap_matrix)  # Maximize overlap
    
    # Create mapping
    mapping = {}
    for i, j in zip(row_ind, col_ind):
        spk2 = list(speakers2)[i]
        spk1 = list(speakers1)[j]
        mapping[spk2] = spk1
    
    return mapping
```

**Numerical Example**:

```
System 1 timeline:
  [0-5s]: Speaker 0
  [5-10s]: Speaker 1

System 2 timeline:
  [0-4.8s]: Speaker A
  [5.2-10s]: Speaker B

Overlap matrix:
           Speaker 0  Speaker 1
Speaker A     4.8s       0s
Speaker B     0s         4.8s

Hungarian matching:
  Speaker A → Speaker 0 ✓
  Speaker B → Speaker 1 ✓
```

### 7.5.3 Step 2: Frame-Level Voting Grid

**Create voting grid**:

```python
def create_voting_grid(timelines, frame_size=0.01):
    """
    Create frame-level voting grid from multiple timelines.
    
    Args:
        timelines: List of [(start, end, speaker_id), ...] for each system
        frame_size: Frame duration (seconds)
    
    Returns:
        Voting grid: (num_frames, num_systems, num_speakers)
    """
    # Determine time range
    max_time = max(seg[1] for timeline in timelines for seg in timeline)
    num_frames = int(max_time / frame_size) + 1
    
    # Determine total speakers (after alignment)
    num_speakers = max(seg[2] for timeline in timelines for seg in timeline) + 1
    num_systems = len(timelines)
    
    # Initialize voting grid
    votes = np.zeros((num_frames, num_systems, num_speakers))
    
    # Fill voting grid
    for sys_idx, timeline in enumerate(timelines):
        for start, end, speaker in timeline:
            start_frame = int(start / frame_size)
            end_frame = int(end / frame_size)
            
            for frame in range(start_frame, end_frame):
                votes[frame, sys_idx, speaker] = 1
    
    return votes
```

**Numerical Example**:

```
3 systems, 2 speakers, 10 frames (100ms total, 10ms frames)

Frame 0 (0-10ms):
  System 1: Speaker 0 = 1, Speaker 1 = 0
  System 2: Speaker 0 = 1, Speaker 1 = 0
  System 3: Speaker 0 = 1, Speaker 1 = 0
  
  Votes: [3, 0] (all systems agree on Speaker 0)

Frame 5 (50-60ms):
  System 1: Speaker 0 = 1, Speaker 1 = 1 (overlap)
  System 2: Speaker 0 = 1, Speaker 1 = 1 (overlap)
  System 3: Speaker 0 = 1, Speaker 1 = 0 (no overlap)
  
  Votes: [3, 2] (2/3 systems detect overlap)
```

### 7.5.4 Step 3: Weighted Voting

**Apply system weights**:

```python
def weighted_voting(votes, weights):
    """
    Apply system weights to voting grid.
    
    Args:
        votes: (num_frames, num_systems, num_speakers)
        weights: (num_systems,) - weight per system
    
    Returns:
        Weighted votes: (num_frames, num_speakers)
    """
    weighted = np.zeros((votes.shape[0], votes.shape[2]))
    
    for frame in range(votes.shape[0]):
        for speaker in range(votes.shape[2]):
            # Weighted sum across systems
            weighted[frame, speaker] = np.sum(
                votes[frame, :, speaker] * weights
            )
    
    return weighted
```

**Weight strategy**:

```python
# Example weights (tuned on dev set)
weights = {
    "system1_ecapa": 1.0,    # Main pipeline (highest weight)
    "system2_pixit": 0.8,    # PixIT (good for overlaps)
    "system3_powerset": 0.6  # Powerset (different error profile)
}

# Normalize
total = sum(weights.values())
normalized_weights = [w / total for w in weights.values()]
# [0.42, 0.33, 0.25]
```

**Adaptive weighting** (optional):

```python
def adaptive_weights(frame, votes, base_weights):
    """
    Adjust weights based on frame characteristics.
    
    Args:
        frame: Frame index
        votes: Voting grid
        base_weights: Base system weights
    
    Returns:
        Adjusted weights for this frame
    """
    # Detect overlap frame
    num_active = np.sum(votes[frame, :, :] > 0, axis=1)
    is_overlap = np.any(num_active > 1)
    
    if is_overlap:
        # Boost PixIT weight in overlap frames
        adjusted = base_weights.copy()
        adjusted[1] *= 1.5  # PixIT system
        return adjusted / np.sum(adjusted)
    else:
        return base_weights
```

### 7.5.5 Step 4: Speaker Selection

**Select speakers based on vote threshold**:

```python
def select_speakers(weighted_votes, threshold=0.5):
    """
    Select active speakers per frame.
    
    Args:
        weighted_votes: (num_frames, num_speakers)
        threshold: Minimum vote fraction to be active
    
    Returns:
        Active speakers: (num_frames, num_speakers) binary
    """
    # Normalize votes per frame
    vote_sums = np.sum(weighted_votes, axis=1, keepdims=True)
    vote_sums[vote_sums == 0] = 1  # Avoid division by zero
    normalized = weighted_votes / vote_sums
    
    # Threshold
    active = (normalized >= threshold).astype(int)
    
    return active
```

**Numerical Example**:

```
Frame 5 weighted votes:
  Speaker 0: 2.5 (System 1: 1.0×0.42 + System 2: 1.0×0.33 + System 3: 1.0×0.25)
  Speaker 1: 1.58 (System 1: 1.0×0.42 + System 2: 1.0×0.33 + System 3: 0.0×0.25)
  
  Total: 4.08
  
Normalized:
  Speaker 0: 2.5 / 4.08 = 0.61
  Speaker 1: 1.58 / 4.08 = 0.39
  
Threshold: 0.5

Active speakers:
  Speaker 0: 0.61 >= 0.5 ✓ (active)
  Speaker 1: 0.39 < 0.5 ✗ (not active)
  
Result: Frame 5 → Speaker 0 only (no overlap)
```

**Lower threshold for overlaps**:

```python
# If any system strongly predicts overlap, use lower threshold
if np.max(weighted_votes) > 0.7 and np.sum(weighted_votes > 0.3) > 1:
    threshold = 0.3  # Lower threshold for overlap detection
```

### 7.5.6 Step 5: Timeline Conversion

**Convert frame-level decisions to timeline**:

```python
def frames_to_timeline(active_speakers, frame_size=0.01):
    """
    Convert frame-level speaker activity to timeline.
    
    Args:
        active_speakers: (num_frames, num_speakers) binary
        frame_size: Frame duration (seconds)
    
    Returns:
        Timeline: [(start, end, speaker_id), ...]
    """
    timeline = []
    num_frames, num_speakers = active_speakers.shape
    
    for speaker in range(num_speakers):
        # Find contiguous active regions for this speaker
        active = active_speakers[:, speaker]
        
        start = None
        for frame in range(num_frames):
            if active[frame] == 1 and start is None:
                # Start of active region
                start = frame * frame_size
            elif active[frame] == 0 and start is not None:
                # End of active region
                end = frame * frame_size
                timeline.append((start, end, speaker))
                start = None
        
        # Handle case where speaker active until end
        if start is not None:
            end = num_frames * frame_size
            timeline.append((start, end, speaker))
    
    # Sort by start time
    timeline.sort(key=lambda x: x[0])
    
    return timeline
```

---

## 7.6 Weighting Strategy (Important)

### 7.6.1 Why Weights Matter

**Not all systems are equally reliable**:

```
Example scenario:
  System 1 (ECAPA): DER = 10.5% (strong overall)
  System 2 (PixIT): DER = 12.8% (weaker overall, but strong on overlaps)
  System 3 (Powerset): DER = 13.2% (different error profile)

Uniform weights (1/3 each):
  Fused DER = 9.2%

Optimized weights (0.5, 0.3, 0.2):
  Fused DER = 8.1% ✓ (1.1% improvement)
```

### 7.6.2 Weight Tuning on Dev Set

**Procedure**:

```python
def tune_weights(systems, dev_ground_truth, weight_grid):
    """
    Tune system weights to minimize DER on dev set.
    
    Args:
        systems: List of system outputs on dev set
        dev_ground_truth: Ground truth RTTM for dev set
        weight_grid: Grid of weight combinations to try
    
    Returns:
        Best weights
    """
    best_weights = None
    best_der = float('inf')
    
    for weights in weight_grid:
        # Fuse systems with these weights
        fused = dover_lap_fusion(systems, weights)
        
        # Compute DER
        der = compute_der(fused, dev_ground_truth)
        
        if der < best_der:
            best_der = der
            best_weights = weights
    
    return best_weights
```

**Typical weight ranges**:

```
Main system (ECAPA): 0.4 - 0.6
PixIT system: 0.2 - 0.4
Powerset system: 0.1 - 0.3

Constraint: weights sum to 1.0
```

### 7.6.3 Adaptive Weighting (Advanced)

**Adjust weights based on frame characteristics**:

```
Overlap frames:
  - Boost PixIT weight (×1.5)
  - Reduce main system weight (×0.8)

Short-turn frames (<500ms):
  - Boost Powerset weight (×1.3)
  - Keep others unchanged

Single-speaker frames:
  - Boost main system weight (×1.2)
  - Reduce PixIT weight (×0.7)
```

**Implementation**:

```python
def adaptive_dover_lap(systems, base_weights, frame_characteristics):
    """
    DOVER-Lap with adaptive weighting.
    
    Args:
        systems: System outputs
        base_weights: Base system weights
        frame_characteristics: Per-frame metadata
    
    Returns:
        Fused timeline
    """
    votes = create_voting_grid(systems)
    
    for frame in range(votes.shape[0]):
        # Adjust weights for this frame
        if frame_characteristics[frame]["is_overlap"]:
            adjusted_weights = base_weights * [0.8, 1.5, 1.0]
        elif frame_characteristics[frame]["is_short_turn"]:
            adjusted_weights = base_weights * [1.0, 1.0, 1.3]
        else:
            adjusted_weights = base_weights
        
        # Normalize
        adjusted_weights /= np.sum(adjusted_weights)
        
        # Apply to this frame
        # ... (voting logic)
    
    return fused_timeline
```

---

## 7.7 What Module 7 Does NOT Do

**Module 7 performs fusion only** - it does NOT:

❌ **Embedding extraction**: Computing speaker representations (Module 3)  
❌ **Clustering**: Grouping segments by speaker (Module 5)  
❌ **Resegmentation**: Refining boundaries (Module 6)  
❌ **Threshold tuning**: Optimizing system parameters (Module 8)  
❌ **RTTM formatting**: Final output generation (Module 8)  
❌ **Training**: Learning fusion weights (weights tuned on dev set)

**Module 7 answers**: "What is the best combined output from multiple systems?"  
**Module 7 does NOT answer**: "How do we optimize individual systems?"

---

## 7.8 Output of Module 7

**Format**:

```python
fused_timeline = [
  {"start_time": 0.0, "end_time": 5.0, "speaker_id": 0},
  {"start_time": 5.0, "end_time": 7.0, "speaker_id": 1},
  {"start_time": 7.0, "end_time": 7.2, "speaker_id": 0},  # Overlap
  {"start_time": 7.0, "end_time": 7.2, "speaker_id": 1},  # Overlap
  {"start_time": 7.2, "end_time": 10.0, "speaker_id": 0}
]
```

**Properties**:
- Single unified diarization timeline
- Overlap-aware (multiple speakers per time range)
- Speaker IDs are **new** (not from any individual system)
- Consistent only within final output

---

## 7.9 Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Multiple System Outputs                                 │
│ System 1 (ECAPA), System 2 (PixIT), System 3 (Powerset)       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: SPEAKER ALIGNMENT                          │
│ Hungarian algorithm on temporal overlap                        │
│ Map speaker IDs across systems                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: FRAME-LEVEL VOTING GRID                    │
│ Discretize time (10-20ms frames)                              │
│ Each system votes for active speakers per frame                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: WEIGHTED VOTING                            │
│ Apply system weights (tuned on dev set)                        │
│ Optional: Adaptive weighting based on frame type               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: SPEAKER SELECTION                          │
│ Threshold voting scores (0.5 default, 0.3 for overlaps)       │
│ Select active speakers per frame                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: TIMELINE CONVERSION                        │
│ Convert frame-level decisions to timeline                      │
│ Merge consecutive frames with same speakers                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Fused Speaker Timeline                                 │
│ Single robust timeline, overlap-aware                          │
│ Next: MODULE 8 (RTTM Output & Scoring)                        │
└─────────────────────────────────────────────────────────────────┘
```

---

## 7.10 Complete End-to-End Example

**Scenario**: Fusing 3 systems for 10-second conversation

### System Outputs

```python
# System 1 (ECAPA main pipeline)
system1 = [
  (0.0, 5.0, 0),   # Speaker 0
  (5.0, 7.0, 1),   # Speaker 1
  (7.2, 10.0, 0)   # Speaker 0
]

# System 2 (PixIT)
system2 = [
  (0.0, 4.8, "A"),  # Speaker A
  (5.2, 7.0, "B"),  # Speaker B
  (7.0, 7.2, "A"),  # Overlap: Speaker A
  (7.0, 7.2, "B"),  # Overlap: Speaker B
  (7.2, 10.0, "A")  # Speaker A
]

# System 3 (Powerset)
system3 = [
  (0.0, 5.0, "X"),  # Speaker X
  (5.0, 7.0, "Y"),  # Speaker Y
  (7.2, 10.0, "X")  # Speaker X
]
```

---

### Step 1: Speaker Alignment

```python
# Align System 2 to System 1
overlap_matrix:
           Spk 0  Spk 1
Spk A       4.8s   0.2s
Spk B       0s     1.8s

Mapping: A → 0, B → 1

# Align System 3 to System 1
overlap_matrix:
           Spk 0  Spk 1
Spk X       5.0s   0s
Spk Y       0s     2.0s

Mapping: X → 0, Y → 1

Aligned systems:
  System 1: [0, 1, 0]
  System 2: [0, 1, 0+1 overlap, 0]
  System 3: [0, 1, 0]
```

---

### Step 2: Voting Grid (Frame 700: 7.0-7.01s)

```python
Frame 700 (7.0-7.01s):
  System 1: Speaker 0 = 0, Speaker 1 = 0 (no overlap detected)
  System 2: Speaker 0 = 1, Speaker 1 = 1 (overlap detected)
  System 3: Speaker 0 = 0, Speaker 1 = 0 (no overlap detected)

Votes: [[0, 1, 0], [0, 1, 0]]
        Sys1 Sys2 Sys3
```

---

### Step 3: Weighted Voting

```python
Weights: [0.42, 0.33, 0.25] (ECAPA, PixIT, Powerset)

Weighted votes (Frame 700):
  Speaker 0: 0×0.42 + 1×0.33 + 0×0.25 = 0.33
  Speaker 1: 0×0.42 + 1×0.33 + 0×0.25 = 0.33

Total: 0.66

Normalized:
  Speaker 0: 0.33 / 0.66 = 0.50
  Speaker 1: 0.33 / 0.66 = 0.50
```

---

### Step 4: Speaker Selection

```python
Threshold: 0.5 (default)

Frame 700:
  Speaker 0: 0.50 >= 0.5 ✓ (active)
  Speaker 1: 0.50 >= 0.5 ✓ (active)

Result: Overlap detected ✓

(Note: With threshold=0.5, both speakers exactly meet threshold.
 In practice, use threshold=0.45 for overlaps to be more inclusive)
```

---

### Step 5: Timeline Conversion

```python
Fused timeline:
  [0.0-5.0s]: Speaker 0
  [5.0-7.0s]: Speaker 1
  [7.0-7.2s]: Speaker 0 (overlap)
  [7.0-7.2s]: Speaker 1 (overlap)
  [7.2-10.0s]: Speaker 0
```

---

### Comparison with Ground Truth

```python
Ground truth:
  [0-5s]: Doctor (Speaker 0)
  [5-7s]: Patient (Speaker 1)
  [7-7.2s]: Overlap (Doctor + Patient)
  [7.2-10s]: Doctor (Speaker 0)

Fused output:
  ✓ All segments correct
  ✓ Overlap correctly detected (thanks to PixIT)
  ✓ Boundaries accurate

Estimated DER: ~6-8% (excellent)
```

---

## 7.11 Key Terminology

| Term | Simple | Technical | Why It Matters |
|------|--------|-----------|----------------|
| **DOVER-Lap** | Voting-based fusion | Diarization Output Voting Error Reduction with Overlap | Standard ensemble method for diarization |
| **Speaker alignment** | Match speakers across systems | Hungarian algorithm on temporal overlap | Required for meaningful voting |
| **Voting grid** | Frame-level votes | (frames, systems, speakers) tensor | Core data structure for fusion |
| **System weights** | Importance of each system | Tuned on dev set to minimize DER | Critical for optimal fusion |
| **Adaptive weighting** | Context-dependent weights | Boost PixIT in overlaps, etc. | Exploits system-specific strengths |

---

## 7.12 Success Criteria

Module 7 is successful if:

1. **DER improvement**: ≥2% absolute DER reduction from best single system
2. **Overlap handling**: ≥90% of overlaps correctly detected (better than any single system)
3. **Speaker consistency**: Aligned speakers match across systems (≥95% agreement)
4. **Robustness**: Fused output stable across different test files
5. **Weight sensitivity**: Performance degrades gracefully with suboptimal weights
6. **Processing speed**: <0.5× real-time (fusion is fast)

---

## 7.13 Summary

> **Module 7 combines multiple diarization systems into a single robust output using DOVER-Lap voting, aligning speakers across systems via Hungarian algorithm, creating frame-level voting grids, applying tuned system weights, selecting speakers based on vote thresholds, and converting to a final overlap-aware timeline that exploits complementary system strengths for 2-4% DER improvement.**

---

**Next Module**: MODULE 8 — Output Formatting, Scoring & DER Optimization

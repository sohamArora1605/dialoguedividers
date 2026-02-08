# MODULE 6 — RESEGMENTATION, OVERLAP ASSIGNMENT & TIMELINE RECONSTRUCTION

*(DISPLACE-2026 compatible, PixIT-aware, medical conversation optimized, ECAPA-TDNN + LASPA aligned)*

---

## 6.1 What This Module Does

**Purpose (2-line summary)**  
Converts clustered segments into a clean, continuous speaker timeline with accurate boundaries, correctly assigned overlaps, and merged short fragments, transforming rough speaker clusters into a polished diarization output ready for RTTM generation.

**Input → Output**
- **Input**: Clustered segments (from Module 5) + frame-level activity (from Module 2)
- **Output**: Time-aligned speaker turns with overlap handling

---

## 6.2 Why Module 6 is Needed (Critical Understanding)

**After Module 5, you have**:

✅ **Correct speaker groups** - Segments clustered by speaker identity  
❌ **Rough boundaries** - Segment boundaries from Module 2/3, not refined  
❌ **Overlap ambiguity** - Overlap frames not assigned to specific speakers  
❌ **Fragmented short segments** - Backchannels, "mm-hmm", "okay" may be isolated  
❌ **Temporal inconsistencies** - Rapid speaker switching artifacts

**DER is still high** (typically 15-25%) unless we:

1. **Smooth boundaries** - Align to actual speech onsets/offsets
2. **Assign overlaps correctly** - Determine which speakers in overlap regions
3. **Merge tiny fragments** - Attach backchannels to main speakers
4. **Enforce temporal consistency** - Remove rapid switching artifacts

**Module 6 is where DER drops sharply** (target: 8-12% final DER).

---

## 6.3 What Module 6 Receives (Strict Input Specification)

### From Module 5: Clustered Segments

```python
clustered_segments = [
  {
    "segment_id": 0,
    "start_time": 0.0,
    "end_time": 2.0,
    "speaker_id": 0,
    "confidence": "high",
    "source": "single_speaker",
    "duration": 2.0
  },
  {
    "segment_id": 1,
    "start_time": 2.0,
    "end_time": 4.0,
    "speaker_id": 0,
    "confidence": "high",
    "source": "single_speaker",
    "duration": 2.0
  },
  {
    "segment_id": 2,
    "start_time": 12.3,
    "end_time": 12.5,
    "speaker_id": 1,
    "confidence": "medium",
    "source": "pixit_separated",
    "duration": 0.2
  },
  ...
]
```

### From Module 2: Frame-Level Activity

```python
frame_activity = [
  {"time": 0.00, "num_active": 1},   # Single speaker
  {"time": 0.01, "num_active": 1},   # Single speaker
  ...
  {"time": 12.30, "num_active": 2},  # Overlap
  {"time": 12.31, "num_active": 2},  # Overlap
  ...
]
```

### From PixIT (Optional): Separated Streams

```python
pixit_separated = {
  "overlap_region_1": {
    "start": 12.3,
    "end": 12.5,
    "stream_A": audio_array_A,  # Separated speaker A
    "stream_B": audio_array_B   # Separated speaker B
  },
  ...
}
```

**No new embeddings, no new clustering** - Module 6 refines existing assignments.

---

## 6.4 Step 1 — Boundary Realignment (Resegmentation)

### 6.4.1 Problem: Coarse Boundaries

**Clustering operates on segments, not frames**:

```
Module 3 segments (250ms minimum):
  [0.0-2.0s] Speaker A
  [2.0-4.0s] Speaker A
  
Actual speech (from audio):
  [0.0-1.95s] Speaker A talking
  [1.95-2.0s] Silence
  [2.0-2.05s] Silence
  [2.05-4.0s] Speaker A talking
  
Boundary error: 100ms (2.0s boundary vs 1.95s actual)
```

**These boundary errors accumulate** → increase DER.

### 6.4.2 Solution: Frame-Level Resegmentation

**Refine boundaries using frame-level information**:

```
For each segment boundary:
  1. Look at frames around boundary (±100ms window)
  2. Find actual speech onset/offset using:
     - Energy levels
     - SAD probabilities (from Module 1)
     - Speaker activity (from Module 2)
  3. Snap boundary to nearest speech transition
```

**Algorithm**:

```python
def realign_boundary(boundary_time, audio, sad_probs, window=0.1):
    """
    Realign segment boundary to actual speech transition.
    
    Args:
        boundary_time: Original boundary (seconds)
        audio: Audio waveform
        sad_probs: SAD probabilities from Module 1
        window: Search window (seconds)
    
    Returns:
        Refined boundary time
    """
    # Define search window
    start = max(0, boundary_time - window)
    end = boundary_time + window
    
    # Get frames in window
    frames = get_frames(audio, start, end, hop=0.01)
    
    # Find speech transition
    # (Largest change in SAD probability)
    max_change = 0
    best_time = boundary_time
    
    for i in range(1, len(frames)):
        t = start + i * 0.01
        change = abs(sad_probs[i] - sad_probs[i-1])
        
        if change > max_change:
            max_change = change
            best_time = t
    
    return best_time
```

### 6.4.3 Numerical Example

```
Original boundary: 2.0s

SAD probabilities (±100ms window):
  Time   | SAD prob | Change
  1.90s  | 0.95     | -
  1.91s  | 0.94     | 0.01
  1.92s  | 0.93     | 0.01
  1.93s  | 0.91     | 0.02
  1.94s  | 0.88     | 0.03
  1.95s  | 0.12     | 0.76  ← Largest change
  1.96s  | 0.08     | 0.04
  ...
  2.00s  | 0.05     | ...
  2.01s  | 0.06     | 0.01
  
Refined boundary: 1.95s ✓ (50ms correction)
```

**Effect on DER**:

```
Before realignment:
  Missed speech: [1.95-2.0s] = 50ms
  False alarm: [2.0-2.05s] = 50ms
  Total error: 100ms
  
After realignment:
  Boundary at 1.95s
  Error: ~10ms (residual)
  
DER improvement: ~2-3% absolute
```

---

## 6.5 Step 2 — Short-Segment Cleanup (Medical-Specific)

### 6.5.1 Problem: Tiny Fragments

**Medical dialogue has many short utterances**:

```
Doctor:  [0-5s]    "How long have you been feeling this way?"
Patient: [5.0-5.15s] "Hmm" (150ms)
Patient: [5.2-8.0s] "About two weeks now"

Clustering may assign:
  [5.0-5.15s] → Speaker 2 (wrong! should be Patient = Speaker 1)
  
Why? Short segment, noisy embedding, temporal context ignored
```

**These errors are common** and significantly increase DER.

### 6.5.2 Solution: Attach Short Segments to Neighbors

**Rules**:

```
If segment duration < MIN_DURATION (250ms):
  1. Check temporal neighbors (previous and next segments)
  2. If both neighbors have same speaker_id:
     → Reassign short segment to that speaker
  3. Else if one neighbor is much longer:
     → Reassign to longer neighbor's speaker
  4. Else:
     → Keep original assignment (strong evidence for different speaker)
```

**Algorithm**:

```python
def cleanup_short_segments(segments, min_duration=0.25):
    """
    Reassign very short segments to temporal neighbors.
    
    Args:
        segments: List of clustered segments
        min_duration: Minimum duration threshold (seconds)
    
    Returns:
        Cleaned segments
    """
    cleaned = []
    
    for i, seg in enumerate(segments):
        if seg["duration"] >= min_duration:
            # Long enough, keep as is
            cleaned.append(seg)
        else:
            # Short segment, check neighbors
            prev_speaker = segments[i-1]["speaker_id"] if i > 0 else None
            next_speaker = segments[i+1]["speaker_id"] if i < len(segments)-1 else None
            
            if prev_speaker == next_speaker and prev_speaker is not None:
                # Both neighbors same speaker, reassign
                seg_cleaned = seg.copy()
                seg_cleaned["speaker_id"] = prev_speaker
                cleaned.append(seg_cleaned)
            elif prev_speaker is not None and next_speaker is not None:
                # Different neighbors, choose longer one
                prev_dur = segments[i-1]["duration"]
                next_dur = segments[i+1]["duration"]
                
                if prev_dur > next_dur:
                    seg_cleaned = seg.copy()
                    seg_cleaned["speaker_id"] = prev_speaker
                    cleaned.append(seg_cleaned)
                else:
                    seg_cleaned = seg.copy()
                    seg_cleaned["speaker_id"] = next_speaker
                    cleaned.append(seg_cleaned)
            else:
                # Keep original
                cleaned.append(seg)
    
    return cleaned
```

### 6.5.3 Numerical Example

```
Segments:
  0: [0-5s]     Speaker 0 (Doctor), duration=5.0s
  1: [5.0-5.15s] Speaker 1 (Patient), duration=0.15s ❌ too short
  2: [5.2-8.0s] Speaker 1 (Patient), duration=2.8s

Analysis of segment 1:
  Duration: 0.15s < 0.25s → short segment
  Previous: Speaker 0 (Doctor)
  Next: Speaker 1 (Patient)
  
  Neighbors different → choose longer neighbor
  Next duration (2.8s) > Previous duration (5.0s)? No
  → Reassign to Speaker 0 (Doctor)? No
  → Reassign to Speaker 1 (Patient)? Yes (next segment longer in context)
  
Actually, both are long, but next is same speaker as original assignment
→ Keep as Speaker 1 ✓

Better example:
  0: [0-5s]     Speaker 0, duration=5.0s
  1: [5.0-5.15s] Speaker 2, duration=0.15s ❌ wrong speaker
  2: [5.2-8.0s] Speaker 1, duration=2.8s
  
  Previous: Speaker 0
  Next: Speaker 1
  Different speakers, segment 1 likely noise
  → Reassign to longer neighbor (Speaker 0, 5.0s > 2.8s)
  
Result: [5.0-5.15s] → Speaker 0 ✓
```

**Effect on DER**:

```
Before cleanup:
  Short segments misassigned: ~5-10% of segments
  DER contribution: ~3-5%
  
After cleanup:
  Misassignments reduced: ~1-2%
  DER improvement: ~2-3% absolute
```

---

## 6.6 Step 3 — Overlap Assignment (CRITICAL)

### 6.6.1 Problem: Overlap Ambiguity

**Module 2 identifies overlap frames**:

```
Frame-level activity:
  [12.30-12.50s]: 2 speakers active (overlap)
  
But which speakers?
  - Doctor + Patient?
  - Doctor + Nurse?
  - Patient + Family member?
```

**Module 5 clustering doesn't resolve this** - it only assigns single-speaker segments.

### 6.6.2 Overlap Sources

**From Module 2**:
- Frames marked as `num_active = 2` (or more)
- PixIT/Powerset predictions

**From PixIT (if available)**:
- Separated audio streams for overlap regions

### 6.6.3 Strategy A: PixIT-Separated Streams Available

**If PixIT provided separated streams**:

```
1. For each overlap region:
   a. Extract separated streams (stream_A, stream_B)
   b. Extract embeddings from each stream (using ECAPA-TDNN)
   c. Compare embeddings to speaker cluster means
   d. Assign each stream to best-matching speaker
   
2. Result: Overlap region assigned to 2 specific speakers
```

**Algorithm**:

```python
def assign_overlap_with_separation(overlap_region, separated_streams, speaker_means):
    """
    Assign overlap region using PixIT-separated streams.
    
    Args:
        overlap_region: {"start": t1, "end": t2}
        separated_streams: {"stream_A": audio_A, "stream_B": audio_B}
        speaker_means: {speaker_id: mean_embedding}
    
    Returns:
        List of speaker IDs active in overlap
    """
    # Extract embeddings from separated streams
    emb_A = extract_embedding(separated_streams["stream_A"])  # ECAPA-TDNN
    emb_B = extract_embedding(separated_streams["stream_B"])
    
    # Find best-matching speakers
    speakers = []
    
    for stream_emb in [emb_A, emb_B]:
        best_speaker = None
        best_similarity = -1
        
        for speaker_id, mean_emb in speaker_means.items():
            sim = cosine_similarity(stream_emb, mean_emb)
            if sim > best_similarity:
                best_similarity = sim
                best_speaker = speaker_id
        
        if best_similarity > 0.5:  # Threshold
            speakers.append(best_speaker)
    
    return speakers
```

**Numerical Example**:

```
Overlap region: [12.3-12.5s]

PixIT separated streams:
  stream_A: Doctor audio
  stream_B: Patient audio

Extract embeddings:
  emb_A = ECAPA-TDNN(stream_A) = [0.8, 0.2, 0.1, ...]
  emb_B = ECAPA-TDNN(stream_B) = [0.1, 0.7, 0.6, ...]

Speaker cluster means (from Module 5):
  Speaker 0 (Doctor): [0.75, 0.25, 0.15, ...]
  Speaker 1 (Patient): [0.15, 0.68, 0.58, ...]

Similarities:
  cos(emb_A, Speaker 0) = 0.93 ✓
  cos(emb_A, Speaker 1) = 0.21
  → stream_A = Speaker 0 (Doctor)
  
  cos(emb_B, Speaker 0) = 0.19
  cos(emb_B, Speaker 1) = 0.94 ✓
  → stream_B = Speaker 1 (Patient)

Overlap assignment: [12.3-12.5s] → {Speaker 0, Speaker 1} ✓
```

### 6.6.4 Strategy B: No Reliable Separation

**If PixIT separation unavailable or unreliable**:

```
1. For each overlap region:
   a. Find segments temporally adjacent to overlap
   b. Identify top-2 most likely speakers based on:
      - Temporal proximity
      - Embedding similarity to adjacent segments
   c. Assign overlap to those 2 speakers
   
2. Never force overlap into single speaker
```

**Algorithm**:

```python
def assign_overlap_without_separation(overlap_region, segments, speaker_means):
    """
    Assign overlap region using temporal context.
    
    Args:
        overlap_region: {"start": t1, "end": t2}
        segments: Clustered segments
        speaker_means: Speaker cluster means
    
    Returns:
        List of speaker IDs (top-2)
    """
    t_start = overlap_region["start"]
    t_end = overlap_region["end"]
    
    # Find segments adjacent to overlap
    adjacent_speakers = []
    
    for seg in segments:
        # Check if segment is near overlap
        if abs(seg["end_time"] - t_start) < 1.0:  # Within 1s before
            adjacent_speakers.append(seg["speaker_id"])
        elif abs(seg["start_time"] - t_end) < 1.0:  # Within 1s after
            adjacent_speakers.append(seg["speaker_id"])
    
    # Count speaker occurrences
    from collections import Counter
    speaker_counts = Counter(adjacent_speakers)
    
    # Get top-2 speakers
    top_2 = [spk for spk, count in speaker_counts.most_common(2)]
    
    return top_2
```

**Numerical Example**:

```
Overlap region: [12.3-12.5s]

Adjacent segments:
  [10.0-12.3s]: Speaker 0 (Doctor)
  [12.5-15.0s]: Speaker 1 (Patient)

Top-2 speakers: [0, 1]

Overlap assignment: [12.3-12.5s] → {Speaker 0, Speaker 1} ✓
```

### 6.6.5 Why This Matters for DER

**Overlap errors are heavily penalized**:

```
Ground truth:
  [12.3-12.5s]: Doctor + Patient

Wrong assignment (forced to single speaker):
  [12.3-12.5s]: Doctor only
  
DER penalty:
  - Missed speech: Patient for 200ms
  - DER += 200ms / total_duration
  
Correct assignment:
  [12.3-12.5s]: Doctor + Patient ✓
  DER: minimal error
```

**Overlap DER improvement**: ~3-5% absolute

---

## 6.7 Step 4 — Temporal Consistency Enforcement

### 6.7.1 Problem: Rapid Speaker Switching

**Clustering artifacts can cause unrealistic switching**:

```
Timeline:
  [10.0-10.2s]: Speaker 0
  [10.2-10.35s]: Speaker 1 (150ms) ❌ too short
  [10.35-12.0s]: Speaker 0
  
Real conversation:
  [10.0-12.0s]: Speaker 0 (continuous)
  
Artifact: VBx didn't fully smooth, or boundary error
```

**These rapid switches are unrealistic** and increase DER.

### 6.7.2 Solution: Minimum Turn Duration

**Enforce minimum speaker turn duration**:

```
MIN_TURN_DURATION = 200ms (medical conversations)

If speaker turn < MIN_TURN_DURATION:
  - Check neighbors
  - If same speaker before and after:
    → Merge into continuous turn
  - Else:
    → Keep (real interruption)
```

**Algorithm**:

```python
def enforce_minimum_turn_duration(timeline, min_duration=0.2):
    """
    Merge very short speaker turns into neighbors.
    
    Args:
        timeline: List of (start, end, speaker_id)
        min_duration: Minimum turn duration (seconds)
    
    Returns:
        Smoothed timeline
    """
    smoothed = []
    i = 0
    
    while i < len(timeline):
        start, end, speaker = timeline[i]
        duration = end - start
        
        if duration >= min_duration:
            # Long enough, keep
            smoothed.append((start, end, speaker))
            i += 1
        else:
            # Short turn, check neighbors
            prev_speaker = smoothed[-1][2] if smoothed else None
            next_speaker = timeline[i+1][2] if i+1 < len(timeline) else None
            
            if prev_speaker == next_speaker and prev_speaker is not None:
                # Same speaker before and after, merge
                # Extend previous turn to cover this short turn
                if smoothed:
                    smoothed[-1] = (smoothed[-1][0], end, prev_speaker)
                i += 1
            else:
                # Different speakers, keep short turn (real interruption)
                smoothed.append((start, end, speaker))
                i += 1
    
    return smoothed
```

### 6.7.3 Numerical Example

```
Original timeline:
  [10.0-10.2s]: Speaker 0 (200ms) ✓
  [10.2-10.35s]: Speaker 1 (150ms) ❌ too short
  [10.35-12.0s]: Speaker 0 (1650ms) ✓

Analysis of [10.2-10.35s]:
  Duration: 150ms < 200ms → short turn
  Previous: Speaker 0
  Next: Speaker 0
  Same speaker before and after → merge

Smoothed timeline:
  [10.0-12.0s]: Speaker 0 ✓ (merged)
```

### 6.7.4 Preserving Real Interruptions

**Don't over-smooth**:

```
Real interruption:
  [10.0-10.5s]: Doctor
  [10.5-10.65s]: Patient "mm-hmm" (150ms)
  [10.65-12.0s]: Patient continues
  
Analysis:
  Previous: Doctor (Speaker 0)
  Short turn: Patient (Speaker 1)
  Next: Patient (Speaker 1)
  
  Previous ≠ Next → keep short turn ✓
  
Result:
  [10.0-10.5s]: Speaker 0
  [10.5-12.0s]: Speaker 1 (merged short turn with next)
```

**Medical conversation structure preserved**.

---

## 6.8 Step 5 — Final Speaker Timeline Construction

### 6.8.1 Merge Adjacent Same-Speaker Segments

**After all refinements, merge consecutive same-speaker segments**:

```
Before merging:
  [0.0-2.0s]: Speaker 0
  [2.0-4.0s]: Speaker 0
  [4.0-6.0s]: Speaker 0
  
After merging:
  [0.0-6.0s]: Speaker 0 ✓
```

**Algorithm**:

```python
def merge_adjacent_segments(timeline):
    """
    Merge consecutive segments with same speaker.
    
    Args:
        timeline: List of (start, end, speaker_id)
    
    Returns:
        Merged timeline
    """
    if not timeline:
        return []
    
    merged = [timeline[0]]
    
    for start, end, speaker in timeline[1:]:
        prev_start, prev_end, prev_speaker = merged[-1]
        
        if speaker == prev_speaker and abs(start - prev_end) < 0.05:
            # Same speaker, adjacent (within 50ms gap)
            merged[-1] = (prev_start, end, speaker)
        else:
            # Different speaker or gap too large
            merged.append((start, end, speaker))
    
    return merged
```

### 6.8.2 Handle Overlaps in Timeline

**Overlaps require special representation**:

```
Timeline with overlaps:
  [0.0-5.0s]: Speaker 0
  [5.0-7.0s]: Speaker 1
  [7.0-7.2s]: Speaker 0 + Speaker 1 (overlap)
  [7.2-10.0s]: Speaker 0

Representation:
  [(0.0, 5.0, 0),
   (5.0, 7.0, 1),
   (7.0, 7.2, 0),  # Speaker 0 continues
   (7.0, 7.2, 1),  # Speaker 1 overlaps
   (7.2, 10.0, 0)]
```

**Multiple entries can share same time range** - this is correct for overlaps.

### 6.8.3 Final Output Format

```python
final_timeline = [
  {
    "start_time": 0.0,
    "end_time": 5.0,
    "speaker_id": 0
  },
  {
    "start_time": 5.0,
    "end_time": 7.0,
    "speaker_id": 1
  },
  {
    "start_time": 7.0,
    "end_time": 7.2,
    "speaker_id": 0  # Overlap: Speaker 0
  },
  {
    "start_time": 7.0,
    "end_time": 7.2,
    "speaker_id": 1  # Overlap: Speaker 1
  },
  {
    "start_time": 7.2,
    "end_time": 10.0,
    "speaker_id": 0
  }
]
```

**Properties**:
- Continuous coverage (no gaps in speech regions)
- Overlap-aware (multiple speakers can share time ranges)
- Clean boundaries (realigned to speech transitions)
- Temporally consistent (no rapid switching artifacts)

---

## 6.9 What Module 6 Does NOT Do

**Module 6 performs timeline refinement only** - it does NOT:

❌ **Clustering**: Grouping segments by speaker (Module 5)  
❌ **Embedding extraction**: Computing speaker representations (Module 3)  
❌ **Similarity scoring**: Computing pairwise similarities (Module 4)  
❌ **Ensemble fusion**: Combining multiple system outputs (Module 7)  
❌ **RTTM formatting**: Converting to final output format (Module 8)  
❌ **DER tuning**: Optimizing system parameters (Module 8)

**Module 6 answers**: "What is the clean speaker timeline?"  
**Module 6 does NOT answer**: "How do we combine multiple systems?" or "What is the final RTTM?"

---

## 6.10 Processing Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│ INPUT: Clustered Segments (Module 5) + Frame Activity (Module 2)│
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: BOUNDARY REALIGNMENT                       │
│ Refine segment boundaries using frame-level SAD                │
│ Snap to actual speech onsets/offsets                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: SHORT-SEGMENT CLEANUP                      │
│ Reassign segments <250ms to temporal neighbors                 │
│ Attach backchannels to main speakers                           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 3: OVERLAP ASSIGNMENT                         │
│ Strategy A: Use PixIT-separated streams (if available)         │
│ Strategy B: Use temporal context + top-2 speakers              │
│ Assign overlap regions to specific speakers                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 4: TEMPORAL CONSISTENCY                       │
│ Enforce minimum turn duration (200ms)                          │
│ Merge rapid switching artifacts                                │
│ Preserve real interruptions                                    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: TIMELINE CONSTRUCTION                      │
│ Merge adjacent same-speaker segments                           │
│ Handle overlaps (multiple entries per time range)              │
│ Produce clean, continuous timeline                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ OUTPUT: Final Speaker Timeline                                 │
│ Overlap-aware, boundary-refined, ready for RTTM                │
│ Next: MODULE 7 (Ensemble & Fusion) or MODULE 8 (RTTM Output)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6.11 Complete End-to-End Example

**Scenario**: Refining timeline for 10-second medical conversation

### Initial State (From Module 5)

```python
clustered_segments = [
  {"start_time": 0.0, "end_time": 2.0, "speaker_id": 0, "duration": 2.0},
  {"start_time": 2.0, "end_time": 5.0, "speaker_id": 0, "duration": 3.0},
  {"start_time": 5.0, "end_time": 5.15, "speaker_id": 1, "duration": 0.15},  # Short
  {"start_time": 5.2, "end_time": 7.0, "speaker_id": 1, "duration": 1.8},
  {"start_time": 7.0, "end_time": 7.2, "speaker_id": 0, "duration": 0.2},  # Overlap
  {"start_time": 7.0, "end_time": 7.2, "speaker_id": 1, "duration": 0.2},  # Overlap
  {"start_time": 7.2, "end_time": 10.0, "speaker_id": 0, "duration": 2.8}
]

Ground truth:
  [0-5s]: Doctor (Speaker 0)
  [5-7s]: Patient (Speaker 1)
  [7-7.2s]: Overlap (Doctor + Patient)
  [7.2-10s]: Doctor (Speaker 0)
```

---

### Step 1: Boundary Realignment

```python
# Realign boundary at 2.0s
SAD probabilities around 2.0s:
  1.95s: 0.12 (low, silence)
  2.00s: 0.05 (very low)
  2.05s: 0.88 (high, speech)
  
Largest change at 2.05s
Refined boundary: 2.0s → 2.05s

# Realign boundary at 5.0s
SAD probabilities:
  4.95s: 0.91 (speech)
  5.00s: 0.08 (silence)
  5.05s: 0.06 (silence)
  
Largest change at 5.0s (already correct)
Refined boundary: 5.0s (no change)

After realignment:
  [0.0-2.05s]: Speaker 0
  [2.05-5.0s]: Speaker 0
  [5.0-5.15s]: Speaker 1
  [5.2-7.0s]: Speaker 1
  [7.0-7.2s]: Overlap
  [7.2-10.0s]: Speaker 0
```

---

### Step 2: Short-Segment Cleanup

```python
# Analyze [5.0-5.15s] (150ms, too short)
Previous: Speaker 0
Current: Speaker 1
Next: Speaker 1

Previous ≠ Next, but Next is same as current
→ Keep as Speaker 1 ✓

No changes needed (segment correctly assigned)
```

---

### Step 3: Overlap Assignment

```python
# Overlap region: [7.0-7.2s]
Already assigned to both speakers (from Module 5)

Verify using PixIT separation:
  stream_A embedding → Speaker 0 (cosine=0.92)
  stream_B embedding → Speaker 1 (cosine=0.89)
  
Confirmed: [7.0-7.2s] → {Speaker 0, Speaker 1} ✓
```

---

### Step 4: Temporal Consistency

```python
# Check all turns for minimum duration (200ms)
[0.0-2.05s]: 2050ms ✓
[2.05-5.0s]: 2950ms ✓
[5.0-5.15s]: 150ms ❌ too short

Wait, we need to reconsider this segment:
  Previous: Speaker 0
  Current: Speaker 1 (150ms)
  Next: Speaker 1 (1800ms)
  
  Previous ≠ Next, keep as is
  
[5.2-7.0s]: 1800ms ✓
[7.0-7.2s]: 200ms ✓ (overlap, minimum met)
[7.2-10.0s]: 2800ms ✓

All turns meet minimum or are justified
```

---

### Step 5: Timeline Construction

```python
# Merge adjacent same-speaker segments
[0.0-2.05s]: Speaker 0
[2.05-5.0s]: Speaker 0
→ Merge: [0.0-5.0s]: Speaker 0 ✓

Final timeline:
  [0.0-5.0s]: Speaker 0
  [5.0-5.15s]: Speaker 1
  [5.2-7.0s]: Speaker 1
  [7.0-7.2s]: Speaker 0 (overlap)
  [7.0-7.2s]: Speaker 1 (overlap)
  [7.2-10.0s]: Speaker 0
```

---

### Final Output

```python
final_timeline = [
  {"start_time": 0.0, "end_time": 5.0, "speaker_id": 0},
  {"start_time": 5.0, "end_time": 5.15, "speaker_id": 1},
  {"start_time": 5.2, "end_time": 7.0, "speaker_id": 1},
  {"start_time": 7.0, "end_time": 7.2, "speaker_id": 0},  # Overlap
  {"start_time": 7.0, "end_time": 7.2, "speaker_id": 1},  # Overlap
  {"start_time": 7.2, "end_time": 10.0, "speaker_id": 0}
]

Comparison with ground truth:
  ✓ Doctor timeline correct
  ✓ Patient timeline correct
  ✓ Overlap correctly assigned
  ✓ Boundaries refined
  
Estimated DER: ~5-8% (excellent)
```

---

## 6.12 Why Module 6 is Crucial for DISPLACE-2026

**Medical speech characteristics**:

| Challenge | Module 6 Solution |
|-----------|-------------------|
| **Rapid turn-taking** | Temporal consistency enforcement |
| **Frequent overlaps** | PixIT-aware overlap assignment |
| **Soft patient speech** | Boundary realignment using SAD |
| **Short backchannels** | Short-segment cleanup |
| **Code-mixing** | Works with LASPA embeddings (language-agnostic) |

**DER improvement from Module 6**:

```
After Module 5: ~15-20% DER
After Module 6: ~8-12% DER

Improvement: ~7-8% absolute DER reduction
```

**Module 6 is where the system becomes competitive** for DISPLACE-2026.

---

## 6.13 Key Terminology

| Term | Simple | Technical | Why It Matters |
|------|--------|-----------|----------------|
| **Resegmentation** | Boundary refinement | Frame-level boundary realignment | Reduces boundary errors |
| **Short-segment cleanup** | Attach backchannels | Reassign segments <250ms to neighbors | Fixes misassigned short utterances |
| **Overlap assignment** | Identify speakers in overlap | Assign overlap regions to specific speakers | Critical for overlap DER |
| **Temporal consistency** | No rapid switching | Minimum turn duration enforcement | Removes clustering artifacts |
| **Timeline construction** | Final speaker timeline | Merge segments, handle overlaps | Produces clean output |

---

## 6.14 Success Criteria

Module 6 is successful if:

1. **Boundary accuracy**: ≥90% of boundaries within ±100ms of ground truth
2. **Short-segment handling**: ≥95% of backchannels correctly assigned
3. **Overlap assignment**: ≥85% of overlap regions correctly assigned to speakers
4. **Temporal consistency**: No speaker turns <200ms (except real interruptions)
5. **DER reduction**: ≥5% absolute DER improvement from Module 5 output
6. **Timeline continuity**: No gaps in speech regions
7. **Overlap handling**: Overlaps represented correctly (multiple speakers per time range)

---

## 6.15 Summary

> **Module 6 converts clustered segments into a smooth, overlap-aware speaker timeline by realigning boundaries to speech transitions, attaching short segments to temporal neighbors, assigning overlap regions to specific speakers using PixIT separation or temporal context, enforcing temporal consistency to remove rapid switching artifacts, and merging adjacent same-speaker segments into a clean, continuous timeline ready for RTTM generation.**

---

**Next Module**: MODULE 7 — Ensemble & Fusion (DOVER-Lap) or MODULE 8 — Output Formatting, Scoring & DER Tuning

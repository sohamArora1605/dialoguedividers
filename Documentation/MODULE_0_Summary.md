# MODULE 0 — AUDIO STANDARDIZATION (SUMMARY)

## What We Do
Convert raw audio + labels into standardized audio (16 kHz, mono, normalized) with aligned labels

## Input and Output
- **Input**: Raw audio (any format) + Speaker labels + Language labels + Overlap annotations
- **Output**: 16 kHz, mono, 16-bit PCM, RMS-normalized audio + Time-aligned labels

---

## Processing Pipeline

```
Raw Audio + Labels
        |
        v
Audio Format Standardization
(Resample to 16kHz, Mono, 16-bit PCM, DC removal)
        |
        v
Speech-Safe Trimming
(Label-aware, preserve <150ms, add ±300ms padding)
        |
        v
Conversation-Level Normalization
(RMS scaling to 0.1)
        |
        v
Label Preservation Check
(Verify alignment, no drift, overlaps intact)
        |
        v
Optional Domain Tagging
(Metadata only, no audio modification)
        |
        v
Standardized Audio + Aligned Labels
```

---

## Processing Steps

### 1. Audio Format Standardization
**What**: Convert all audio to uniform format  
**How**: 
- Resample to 16 kHz
- Convert to mono (channel averaging)
- Standardize to 16-bit PCM
- Remove DC offset

**Why**: Models require consistent input format (PyAnnote, PixIT, WavLM expect 16 kHz mono)  
*Example: A 44.1 kHz stereo file becomes 16 kHz mono, reducing size by 64% while preserving speech*

---

### 2. Speech-Safe Trimming (Label-Aware)
**What**: Remove unnecessary silence while preserving all speech  
**How**:
- Only trim unlabeled silence at boundaries
- Never trim inside labeled regions
- Preserve all segments below 150 ms
- Add ±300 ms padding around speech

**Why**: Medical conversations have short backchannels that must be preserved for accurate diarization  
*Example: A 180ms "mm-hmm" at [12.5s-12.68s] is kept intact, even though it's very short*

---

### 3. Conversation-Level Normalization
**What**: Standardize loudness across recordings  
**How**:
- Calculate RMS (Root Mean Square) energy
- Scale entire conversation to target RMS = 0.1
- Apply same scaling to all speakers

**Why**: Makes recordings comparable while preserving natural speaker loudness differences  
*Example: Quiet recording (RMS=0.02) and loud recording (RMS=0.8) both scaled to RMS=0.1, making them directly comparable*

---

### 4. NO Chunking
**What**: Keep audio continuous  
**How**: Do NOT split into fixed-length segments  
**Why**: PixIT/powerset models need continuous context for overlap detection and turn-taking analysis  
*Example: Speaker overlap at [25.5s-27.2s] stays intact instead of being split across 3-second chunks*

---

### 5. Label Preservation Check
**What**: Verify labels still align after processing  
**How**:
- Check all timestamps within audio duration
- Verify no timestamp drift (under 1 ms error)
- Confirm overlap annotations intact

**Why**: Resampling and trimming can introduce timing errors that break label alignment  
*Example: After resampling, verify Speaker A label [10.0s-12.0s] still maps to exact same audio content*

---

### 6. Short-Utterance Protection
**What**: Guarantee segments under 250 ms are preserved  
**How**: Never merge, delete, or smooth segments below threshold  
**Why**: Backchannels ("mm-hmm", "okay") are critical for speaker identification and turn-taking  
*Example: Doctor says "okay" (150ms) between patient sentences - losing this breaks conversation flow*

---

### 7. Optional Domain Tagging
**What**: Attach metadata tags  
**How**: Add domain/environment/noise_level tags to metadata (no audio modification)  
**Why**: Enables batch analysis and error diagnosis by recording environment  
*Example: Tag file as "clinic" environment to later analyze if model performs worse in clinic vs. hospital*

---

## Complete Processing Pipeline (Detailed)

```
┌─────────────────────────────────────────────────────────────────┐
│                   INPUT: Raw Audio + Labels                     │
│                                                                 │
│  Audio: meeting.mp3 (44.1 kHz, Stereo, 24-bit, 180 seconds)   │
│  Speaker Labels: 156 segments                                  │
│  Language Labels: 156 segments                                 │
│  Overlap Annotations: 23 regions                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 1: AUDIO FORMAT STANDARDIZATION               │
│                                                                 │
│  1a. Decode MP3 → PCM samples                                  │
│  1b. Resample: 44.1 kHz → 16 kHz                              │
│  1c. Convert: Stereo → Mono (channel averaging)               │
│  1d. Bit depth: 24-bit → 16-bit PCM                           │
│  1e. Remove DC offset (center around zero)                     │
│                                                                 │
│  Result: 16 kHz, Mono, 16-bit PCM, 180 seconds                │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 2: SPEECH-SAFE TRIMMING                       │
│                                                                 │
│  2a. Identify unlabeled silence regions                        │
│  2b. Check: Is silence at start/end only?                      │
│  2c. Add ±300ms padding to all labeled regions                │
│  2d. Trim only padded silence at boundaries                    │
│  2e. Verify: No labeled segments < 150ms removed               │
│                                                                 │
│  Result: Trimmed to 175 seconds (5s silence removed)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         STEP 3: CONVERSATION-LEVEL NORMALIZATION                │
│                                                                 │
│  3a. Calculate current RMS: 0.045                              │
│  3b. Set target RMS: 0.1                                       │
│  3c. Scale factor: 0.1 / 0.045 = 2.22                         │
│  3d. Apply scaling to entire conversation                      │
│  3e. Verify: New RMS = 0.1 ✓                                  │
│                                                                 │
│  Result: Normalized amplitude (RMS = 0.1)                      │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│         STEP 4: LABEL PRESERVATION & ALIGNMENT CHECK            │
│                                                                 │
│  4a. Verify all 156 speaker labels within [0, 175s] ✓         │
│  4b. Verify all 156 language labels aligned ✓                  │
│  4c. Verify all 23 overlap regions intact ✓                    │
│  4d. Check timestamp precision (< 1ms error) ✓                 │
│  4e. Validate no segments < 250ms were lost ✓                  │
│                                                                 │
│  Result: All labels perfectly aligned                          │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│              STEP 5: OPTIONAL DOMAIN TAGGING                    │
│                                                                 │
│  5a. Attach metadata:                                          │
│      - domain: "medical_far_field"                             │
│      - environment: "clinic"                                   │
│      - noise_level: "medium"                                   │
│  5b. No audio modification ✓                                   │
│                                                                 │
│  Result: Tagged metadata (audio unchanged)                     │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                   OUTPUT: STANDARDIZED AUDIO                    │
│                                                                 │
│  Audio: 16 kHz, Mono, 16-bit PCM, 175 seconds, RMS=0.1        │
│  Speaker Labels: 156 segments (time-aligned)                   │
│  Language Labels: 156 segments (time-aligned)                  │
│  Overlap Annotations: 23 regions (preserved)                   │
│  Metadata: Domain tags attached                                │
│                                                                 │
│  Next: MODULE 1 (Speech Activity Detection)                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Complete End-to-End Example

**Scenario**: Processing a 3-minute medical consultation recording

### Initial State

```
File: clinic_visit_2026_01_15.mp3
Format: 44.1 kHz, Stereo, 24-bit, 180.5 seconds
File size: 18.2 MB

Labels (provided):
- Speaker A (Doctor): 78 segments
- Speaker B (Patient): 78 segments
- Language: English throughout
- Overlaps: 23 regions where both speak simultaneously

Sample segments:
  Doctor:  [0.0-4.5s] "How long have you been feeling this way?"
  Patient: [5.0-5.18s] "Um" (180ms - short backchannel!)
  Patient: [5.5-9.2s] "about two weeks I think"
  Doctor:  [9.5-9.65s] "Okay" (150ms - short backchannel!)
  Doctor:  [10.0-13.5s] "And where exactly is the pain?"
  ...
  Overlap: [45.2-46.8s] Both speaking (Doctor interrupts)
```

---

### Step 1: Audio Format Standardization

**1a. Decode MP3**
```
MP3 compressed → PCM samples
Result: Raw audio data in memory
```

**1b. Resample 44.1 kHz → 16 kHz**
```
Original: 44,100 samples/second × 180.5s = 7,960,050 samples
Resampled: 16,000 samples/second × 180.5s = 2,888,000 samples
Reduction: 63.7% fewer samples
File size: 18.2 MB → 5.5 MB (stereo, 16-bit)
```

**1c. Stereo → Mono**
```
Before (stereo):
  Left channel:  [0.42, 0.58, -0.31, 0.67, ...]
  Right channel: [0.38, 0.54, -0.35, 0.61, ...]

After (mono - channel averaging):
  Mono: [(0.42+0.38)/2, (0.58+0.54)/2, (-0.31-0.35)/2, ...]
  Mono: [0.40, 0.56, -0.33, 0.64, ...]

File size: 5.5 MB → 2.75 MB
```

**1d. 24-bit → 16-bit PCM**
```
24-bit: 16,777,216 amplitude levels
16-bit: 65,536 amplitude levels
Still sufficient for speech (96 dB dynamic range)

File size: 2.75 MB → 2.75 MB (already 16-bit after conversion)
```

**1e. Remove DC Offset**
```
Calculate mean of all samples:
  Mean = 0.0023 (small DC offset detected)

Subtract from all samples:
  Before: [0.40, 0.56, -0.33, 0.64, ...]
  After:  [0.3977, 0.5577, -0.3323, 0.6377, ...]
  New mean: 0.0000 ✓
```

**Result after Step 1**:
```
Format: 16 kHz, Mono, 16-bit PCM
Duration: 180.5 seconds
Samples: 2,888,000
File size: 5.5 MB (from original 18.2 MB)
```

---

### Step 2: Speech-Safe Trimming

**2a. Identify unlabeled silence**
```
Timeline analysis:
  [0.0-0.5s]: Silence (unlabeled) ← Can trim
  [0.5-4.5s]: Doctor speaking (labeled)
  [4.5-5.0s]: Silence (unlabeled, but between speech)
  [5.0-5.18s]: Patient "Um" (labeled, 180ms!)
  ...
  [178.5-180.5s]: Silence (unlabeled) ← Can trim
```

**2b. Check boundaries**
```
Start silence: [0.0-0.5s] = 500ms ✓ Safe to trim
End silence: [178.5-180.5s] = 2000ms ✓ Safe to trim
Internal gaps: Keep all (between labeled speech)
```

**2c. Add ±300ms padding**
```
Original segments:
  Doctor:  [0.5-4.5s]
  Patient: [5.0-5.18s]
  Patient: [5.5-9.2s]

With padding:
  Doctor:  [0.2-4.8s]  (added -300ms, +300ms)
  Patient: [4.7-5.48s] (added -300ms, +300ms)
  Patient: [5.2-9.5s]  (added -300ms, +300ms)

Overlapping padded regions merge:
  [0.2-4.8s] + [4.7-5.48s] + [5.2-9.5s] → [0.2-9.5s]
```

**2d. Trim boundaries**
```
Original: [0.0-180.5s]
After padding analysis:
  First speech (with padding): starts at 0.2s
  Last speech (with padding): ends at 178.2s

Trim:
  Remove [0.0-0.2s]: 200ms
  Remove [178.2-180.5s]: 2300ms
  Total removed: 2.5 seconds
```

**2e. Verify short segments**
```
Check all segments < 250ms:
  Patient "Um" [5.0-5.18s]: 180ms ✓ PRESERVED
  Doctor "Okay" [9.5-9.65s]: 150ms ✓ PRESERVED
  Patient "Mm-hmm" [23.1-23.25s]: 150ms ✓ PRESERVED
  ... (all 12 short segments preserved)
```

**Result after Step 2**:
```
Duration: 178.0 seconds (from 180.5s)
Trimmed: 2.5 seconds of boundary silence
All labeled speech preserved ✓
All short segments (≥150ms) intact ✓
```

---

### Step 3: Conversation-Level Normalization

**3a. Calculate current RMS**
```
RMS = √(1/N × Σ(x[n]²))

Sample calculation (first 1000 samples):
  Squared: [0.158, 0.311, 0.110, 0.407, ...]
  Sum: 45.2
  Mean: 45.2 / 1000 = 0.0452
  RMS: √0.0452 = 0.213

Full audio RMS: 0.045 (quiet recording)
```

**3b. Set target**
```
Target RMS: 0.1 (standard)
```

**3c. Calculate scale factor**
```
Scale = Target / Current
Scale = 0.1 / 0.045 = 2.22
```

**3d. Apply scaling**
```
Before: [0.3977, 0.5577, -0.3323, 0.6377, ...]
After:  [0.883, 1.238, -0.737, 1.416, ...]
        (each sample × 2.22)
```

**3e. Verify**
```
New RMS calculation:
  RMS = √(1/N × Σ((x[n] × 2.22)²))
  RMS = 2.22 × √(1/N × Σ(x[n]²))
  RMS = 2.22 × 0.045
  RMS = 0.0999 ≈ 0.1 ✓
```

**Result after Step 3**:
```
All amplitudes scaled by 2.22
RMS: 0.1 (standardized)
Relative loudness between speakers preserved
```

---

### Step 4: Label Preservation & Alignment Check

**4a. Verify speaker labels**
```
Original labels (before trimming):
  Doctor [0.5-4.5s] → After trim: [0.3-4.3s]
  Patient [5.0-5.18s] → After trim: [4.8-4.98s]
  ...

Check: All 156 segments within [0, 178.0s]? ✓
```

**4b. Verify language labels**
```
All 156 language labels checked
All within audio duration ✓
All aligned with speaker labels ✓
```

**4c. Verify overlaps**
```
Original overlap [45.2-46.8s]
After trim (0.2s removed from start): [45.0-46.6s]

Check overlap integrity:
  Doctor segment: [44.5-46.2s]
  Patient segment: [45.0-47.1s]
  Overlap region: [45.0-46.2s] ✓
  Duration: 1.2s (unchanged) ✓
```

**4d. Check timestamp precision**
```
Sample rate: 16,000 Hz
Time resolution: 1/16,000 = 0.0000625s = 0.0625ms

Verify label [4.8-4.98s]:
  Start: 4.8s × 16,000 = 76,800 samples (exact)
  End: 4.98s × 16,000 = 79,680 samples (exact)
  Precision: 0.0625ms < 1ms ✓
```

**4e. Validate short segments**
```
All segments < 250ms:
  12 segments found
  All preserved ✓
  Shortest: 150ms ✓
```

**Result after Step 4**:
```
All labels verified and aligned
Timestamp precision: < 0.1ms
No segments lost
Overlaps intact
```

---

### Step 5: Optional Domain Tagging

**5a. Attach metadata**
```json
{
  "file_id": "clinic_visit_2026_01_15",
  "domain": "medical_far_field",
  "environment": "clinic",
  "noise_level": "medium",
  "recording_date": "2026-01-15",
  "duration_seconds": 178.0,
  "num_speakers": 2,
  "speaker_labels": ["Doctor", "Patient"],
  "languages": ["English"],
  "has_overlaps": true,
  "num_overlap_regions": 23,
  "short_segments_count": 12,
  "processing_timestamp": "2026-01-29T19:52:00Z"
}
```

**5b. Verify no audio modification**
```
Audio samples: Unchanged ✓
Only metadata added ✓
```

---

### Final Output

```
File: clinic_visit_2026_01_15_processed.wav
Format: 16 kHz, Mono, 16-bit PCM
Duration: 178.0 seconds
File size: 5.3 MB
RMS: 0.1

Labels:
- Speaker labels: 156 segments (perfectly aligned)
- Language labels: 156 segments (perfectly aligned)
- Overlap annotations: 23 regions (preserved)

Metadata: Attached (JSON sidecar file)

Quality checks:
✓ All speech preserved (especially 12 short segments)
✓ Labels aligned (< 0.1ms precision)
✓ Overlaps intact
✓ No chunking
✓ Deterministic (same input → same output)

Ready for: MODULE 1 (Speech Activity Detection)
```

---

## Key Constraints

**MUST DO**:
- Preserve all labeled speech (especially under 250 ms)
- Maintain label alignment (under 1 ms error)
- Keep audio continuous (no chunking)
- Normalize at conversation level (not per-speaker)

**MUST NOT DO**:
- Speaker diarization
- Speech activity detection
- Language identification
- Overlap detection (only preserve existing annotations)
- Acoustic augmentation

---

## Technical Specifications

| Parameter | Value | Reason |
|-----------|-------|--------|
| Sample Rate | 16 kHz | Captures 0-8 kHz (full speech range) |
| Channels | Mono | Model compatibility |
| Bit Depth | 16-bit PCM | Standard for speech (96 dB dynamic range) |
| Target RMS | 0.1 | Standard normalization level |
| Min Segment | 150 ms | Shortest meaningful speech |
| Protection Threshold | 250 ms | Backchannel preservation |
| Context Padding | ±300 ms | Preserves speech transitions |

---

## Summary
Loss-free audio standardization with strict label preservation and no inference decisions.

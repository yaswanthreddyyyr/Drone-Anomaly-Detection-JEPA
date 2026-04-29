# Drone Telemetry Tampering Dataset v2 - Summary

## Dataset Overview
**Source:** Kaggle - `rasikaekanayakadevlk/drone-telemetry-tampering-dataset-v2`  
**Size:** 663 MB (uncompressed ~1.3 GB)  
**License:** CC-BY-SA-4.0

## Dataset Structure

### Three Difficulty Levels:
1. **Balanced** - Medium difficulty anomalies
2. **Strong** - Severe, obvious anomalies  
3. **Subtle** - Difficult-to-detect anomalies

Each level contains:
- 4 replicates (rep_00 to rep_03)
- 60 cases per replicate
- **Total: 720 cases** (240 per difficulty level)

### Anomaly Type Distribution (per replicate):
| Anomaly Type            | Count | Description |
|------------------------|-------|-------------|
| deletion_gap           | 10    | Missing telemetry chunks |
| **injection**          | **9** | **GPS spoofing attacks** ✓ |
| timestamp_drift        | 7     | Time inconsistencies |
| altitude_spike         | 7     | Sudden altitude changes |
| precision_rounding     | 6     | Coordinate precision errors |
| **normal**             | **5** | **Clean flight data** ✓ |
| coordinate_jump        | 5     | GPS coordinate jumps |
| speed_inconsistency    | 4     | Unrealistic speed changes |
| heading_inconsistency  | 4     | Direction anomalies |
| combined               | 3     | Multiple simultaneous faults |

## File Format

Each case contains:
```
case_XXXX_<anomaly_type>/
├── case_meta.json           # Metadata (anomaly type, rate, etc.)
├── decoded_flightlog.csv    # Telemetry data (7 features)
└── labels.csv               # Binary labels (0=normal, 1=anomaly)
```

### Flight Log Features (7 total):
1. `timestamp` - UTC timestamp
2. `latitude` - GPS latitude  
3. `longitude` - GPS longitude
4. `altitude` - Altitude in meters
5. `speed` - Velocity (m/s or km/h)
6. `heading` - Direction in degrees
7. `source` - Data source identifier

**Note:** Proposal mentions 9 features - may need to derive additional features (e.g., acceleration, velocity components, distance traveled).

## Data Split Recommendation

### For JEPA Training (Self-Supervised):
**Use:** All normal cases across all replicates
- Normal cases per replicate: 5
- Total normal cases: 5 × 4 replicates × 3 difficulties = **60 normal flights**
- Additional normal data may be embedded in non-tampered portions of anomaly cases

**Estimated chunks:** ~11,000 rows/flight × 60 flights ÷ 20 rows/chunk = **~33,000 chunks**

### For Validation (20%):
**Balanced difficulty recommended**
- Use: `rep_00` and `rep_01` from balanced dataset
- Cases: 60 × 2 = 120 cases
- Stratified by anomaly type

### For Testing (80%):
- **Balanced:** `rep_02`, `rep_03` (120 cases)
- **Strong:** All 4 replicates (240 cases) - test robustness to severe faults
- **Subtle:** All 4 replicates (240 cases) - test sensitivity
- **Total test cases:** ~600 cases

### GPS Spoofing Zero-Shot Test:
- All **injection** cases from test split
- Expected: 9 cases/replicate × test replicates = **~54 injection cases**

## Advantages for JEPA-DRONE Project

✅ **Perfect Ground Truth:** Binary labels for every waypoint  
✅ **Multiple Anomaly Types:** Covers 10 distinct fault scenarios  
✅ **GPS Spoofing Data:** Direct support for injection attack detection  
✅ **Difficulty Gradation:** Balanced/Strong/Subtle allows robustness testing  
✅ **Real DJI Data:** Decoded from actual DJI flight logs  
✅ **Sufficient Scale:** 720 cases with thousands of waypoints each  
✅ **Labeled Data:** No synthetic generation needed for validation  

## Next Steps

1. ✅ Dataset downloaded and extracted
2. [ ] Create data preprocessing pipeline
3. [ ] Extract normal-flight chunks for JEPA training
4. [ ] Stratify validation/test splits by anomaly type
5. [ ] Compute dataset statistics (chunk count, anomaly rates)
6. [ ] Derive additional features if needed (acceleration, jerk, etc.)

---

**Dataset Ready!** This is an excellent choice for your JEPA-DRONE course project.

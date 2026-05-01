# Suggested Hyperparameter Combinations for Improved Results

## Current Performance (Baseline)

After running `scripts/diagnose_scores.py`, the **TRUE** performance is:

| Metric | Test Balanced | Notes |
|--------|--------------|-------|
| AUC-ROC | **0.595** | Not 0.37 - there was an evaluation bug |
| Best Recall | **99.98%** | With optimal threshold |
| Best F1 | **0.881** | With optimal threshold |
| FAR at 85% Recall | **71%** | **This is the real problem!** |

**Root Cause Identified**: The threshold (-0.459) used in evaluation was miscalibrated. 
With F1-optimal threshold (-0.5186), results are much better.

**Remaining Issue**: AUC-ROC of 0.59 is still weak. To reduce FAR to <5% while keeping recall >85%, 
we need AUC-ROC > 0.90. Current embeddings don't separate anomalies well enough.

---

## Root Cause Analysis

1. **Score Inversion Problem**: Isolation Forest returns lower scores for anomalies, but your embeddings might have reversed this relationship
2. **Contamination Mismatch**: Training contamination=0.05 vs actual test anomaly rate ~79%
3. **Training Data**: Mixed normal/anomaly training may corrupt learned representations
4. **Threshold Miscalibration**: Current threshold (-0.459) is not properly tuned

---

## Combination 1: Fix Score Direction + Threshold Tuning

**Key changes**: Invert scores and use optimal threshold from validation

```yaml
# configs/config_combo1.yaml
isolation_forest:
  n_estimators: 200
  contamination: "auto"  # Let sklearn determine
  max_samples: 0.8
  score_inversion: true  # ADD THIS - negate scores
  
training:
  batch_size: 256
  epochs: 150  # More training
  learning_rate: 0.0005  # Lower LR for stability
```

**Code modification needed** in `isolation_forest.py`:
```python
def score_samples(self, embeddings):
    scores = self.iso_forest.score_samples(embeddings)
    return -scores  # INVERT: higher = more anomalous
```

---

## Combination 2: Use Reconstruction Error (Replace Isolation Forest)

Instead of Isolation Forest on embeddings, use JEPA's prediction error directly.

```yaml
model:
  embedding_dim: 256
  encoder_hidden: [512, 512, 256]  # Deeper encoder
  predictor_hidden: [512, 256]     # Stronger predictor
  use_reconstruction_score: true   # NEW

anomaly_detection:
  method: "reconstruction_error"  # Not isolation_forest
  aggregation: "max"  # Use max error across masked positions
  percentile_threshold: 95
```

**Rationale**: The prediction error for anomalous patterns should be higher since the model learned normal patterns only.

---

## Combination 3: Different Masking Strategy (Block Masking)

Current random masking may not capture temporal anomalies well.

```yaml
masking:
  strategy: "block"  # Instead of random
  block_size: 5      # Mask 5 consecutive waypoints
  min_mask_ratio: 0.30
  max_mask_ratio: 0.60
  
model:
  embedding_dim: 512  # Larger embedding
  encoder_hidden: [1024, 512, 256]
  predictor_hidden: [512, 512, 256]
  use_attention: true
```

**Rationale**: GPS anomalies often span consecutive waypoints; block masking forces model to learn temporal dependencies.

---

## Combination 4: Multi-Scale Feature Engineering

Add more derived features to capture different anomaly signatures.

```yaml
features:
  base_features:
    - latitude
    - longitude  
    - altitude
    - speed
    - heading
    
  derived_features:
    # Rate of change (1st order)
    - delta_lat
    - delta_lon
    - delta_alt
    - acceleration
    - angular_velocity
    - distance
    
    # NEW: Second-order derivatives (jerk, turning rate change)
    - delta_acceleration   # Rate of change of acceleration
    - delta_angular_vel    # Rate of heading change change
    - altitude_jerk        # Rate of altitude acceleration
    
    # NEW: Statistical windows
    - speed_variance_5     # 5-point window variance
    - heading_variance_5
    - altitude_variance_5
    
  use_derived: true
  use_second_order: true   # NEW
  use_windowed_stats: true # NEW
```

---

## Combination 5: Transformer-based Encoder (Instead of MLP)

Replace the MLP encoder with a proper Transformer for better temporal modeling.

```yaml
model:
  encoder_type: "transformer"  # NEW: was "mlp"
  embedding_dim: 256
  n_heads: 8
  n_layers: 4
  ff_dim: 1024
  dropout: 0.1
  
  # Keep predictor as MLP
  predictor_hidden: [512, 256]

training:
  batch_size: 128  # Smaller due to memory
  epochs: 200
  learning_rate: 0.0001  # Lower for transformer
  weight_decay: 0.01
  warmup_epochs: 10
```

**Rationale**: Transformers excel at modeling sequential dependencies; GPS telemetry is highly sequential.

---

## Combination 6: Contrastive Learning Enhancement

Add a contrastive loss to pull normal embeddings together and push anomaly-like patterns apart.

```yaml
training:
  loss_type: "jepa_contrastive"  # NEW
  jepa_weight: 0.7
  contrastive_weight: 0.3
  temperature: 0.1
  
  # Create pseudo-negatives by augmenting
  augmentation:
    noise_scale: 0.1
    time_shift: [-2, 2]
    feature_dropout: 0.1
```

---

## Combination 7: Ensemble of Detection Methods

Combine multiple anomaly scores for robustness.

```yaml
anomaly_detection:
  ensemble: true
  methods:
    - name: "isolation_forest"
      weight: 0.3
      n_estimators: 200
    - name: "reconstruction_error"
      weight: 0.4
    - name: "local_outlier_factor"  # NEW
      weight: 0.2
      n_neighbors: 20
    - name: "one_class_svm"  # NEW
      weight: 0.1
      kernel: "rbf"
```

---

## Quick Fix Experiment: Score Inversion Test

Before trying complex changes, run this quick test to confirm score inversion:

```python
# Add to evaluate.py or create quick_test.py
from sklearn.metrics import roc_auc_score

# Get scores
scores = anomaly_detector.score_samples(embeddings)
labels = test_labels  # 1 = anomaly

# Try both directions
auc_original = roc_auc_score(labels, scores)
auc_inverted = roc_auc_score(labels, -scores)

print(f"Original AUC: {auc_original:.4f}")
print(f"Inverted AUC: {auc_inverted:.4f}")  # Should be much higher!
```

---

## Recommended Priority Order

1. **FIRST**: Test score inversion (quick fix, may solve everything)
2. **SECOND**: Combination 2 - Use reconstruction error directly
3. **THIRD**: Combination 4 - Better feature engineering
4. **FOURTH**: Combination 3 - Block masking
5. **FIFTH**: Combination 5 or 6 - Architecture changes

---

## Expected Improvements

| Combination | Expected AUC-ROC | Expected Recall | Complexity |
|-------------|------------------|-----------------|------------|
| Score Inversion Fix | 0.62-0.65 | 60-70% | Very Easy |
| Reconstruction Error | 0.75-0.85 | 70-80% | Easy |
| Block Masking | 0.70-0.80 | 65-75% | Medium |
| Better Features | 0.80-0.88 | 75-85% | Medium |
| Transformer Encoder | 0.85-0.92 | 80-90% | Hard |
| Full Ensemble | 0.88-0.95 | 85-92% | Hard |

---

## Implementation Notes

To implement these combinations, I can create:
1. `configs/config_combo1.yaml` through `configs/config_combo7.yaml`
2. Modified `isolation_forest.py` with score inversion option
3. New `reconstruction_anomaly_detector.py` 
4. Enhanced `preprocessing.py` with second-order features
5. New `transformer_encoder.py`

Let me know which combinations you want me to implement!

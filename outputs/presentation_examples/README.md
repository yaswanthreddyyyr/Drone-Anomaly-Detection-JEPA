# Presentation Examples (Anomaly vs Normal)

This folder contains slide-ready examples selected by model confidence.

## Folder structure

- `test_balanced/`
- `test_strong/`
- `test_subtle/`

Each split has:
- `anomaly/`: strongest true-positive anomaly examples (`TP`)
- `normal/`: strongest true-negative normal examples (`TN`)
- `hard_cases/`: confident errors and edge cases (`FP`/`FN`) for discussion
- `examples_table.csv`: per-example metadata
- `examples_summary.json`: run config + metrics + selected examples

## Confidence definition

Confidence is based on distance from threshold:

- Let score be LOF anomaly score and threshold be calibrated on validation data.
- `confidence = clip(|score - threshold| / p95(|score-threshold|), 0, 1)`.

Interpretation:
- High confidence anomaly: score much greater than threshold
- High confidence normal: score much lower than threshold

## Recommended slide selection

For each split:
1. Pick 2-3 images from `anomaly/` (strong detections)
2. Pick 2-3 images from `normal/` (clean normal behavior)
3. Optionally add 1 image from `hard_cases/` to explain model limits

## Regeneration command

```bash
/Users/deepank/Downloads/AI-JEPA/.venv/bin/python scripts/generate_presentation_examples.py \
  --checkpoint smoke_outputs/run_20260420_103424/best_model.pt \
  --data-dir processed_data \
  --split test_balanced \
  --output-dir outputs/presentation_examples \
  --n-examples-per-class 8 \
  --batch-size 512
```

Repeat with `--split test_strong` and `--split test_subtle`.

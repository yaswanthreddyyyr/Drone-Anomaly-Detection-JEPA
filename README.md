# Drone-Anomaly-Detection-JEPA

Local setup and run notes for the JEPA-based drone anomaly detection project.

## Setup

```bash
cd "/Users/deepank/Downloads/AI-JEPA"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Data

The training scripts expect:

- raw telemetry under `data/drone_temparing_dataset_v2`
- processed chunks under `processed_data`

If `processed_data` does not exist yet, run preprocessing first after placing the raw dataset in the expected location.

## Run

Show available CLI options:

```bash
python scripts/preprocess_data.py --help
python scripts/train_jepa.py --help
python scripts/train_full_pipeline.py --help
```

Preprocess the dataset:

```bash
python scripts/preprocess_data.py --config configs/config.yaml
```

Train JEPA only:

```bash
python scripts/train_jepa.py --config configs/config.yaml --data-dir processed_data
```

Run the full pipeline:

```bash
python scripts/train_full_pipeline.py --config configs/config.yaml --data-dir processed_data
```

## Notes

- The repo is now cloned locally in this workspace.
- If you want, I can next install the Python dependencies and try the script smoke tests for you.

# 🚁 Drone Telemetry Anomaly Detection — JEPA Pipeline

> Unsupervised anomaly detection for drone telemetry using Joint Embedding Predictive Architecture (JEPA) + Local Outlier Factor. Trained only on normal flight data. No attack labels needed.

---

## 📊 Results

| Test Split | AUC | Recall | F1 | FAR |
|---|---|---|---|---|
| test_balanced | 0.8078 | 0.5602 | 0.7105 | 0.0620 |
| test_strong | 0.7602 | 0.5072 | 0.6637 | 0.0956 |
| test_subtle | 0.7675 | 0.5268 | 0.6793 | 0.0923 |

**Composite score: 2.60** (28% improvement over baseline)

---

## 🧠 How It Works

1. **JEPA Encoder** — Takes 11-feature telemetry windows, randomly masks portions, and learns to predict masked content in embedding space. Trained only on normal flights.
2. **LOF Detector** — Local Outlier Factor scores each embedding by comparing its local density to k nearest neighbors. Anomalies sit in sparse regions.
3. **Threshold Calibration** — Threshold tuned on validation set under a strict 20% FAR cap.

**Best detector config:** `StandardScaler → PCA(130) → LOF(manhattan, k=21)`

---

## 📁 Repository Structure

```
Drone-Anomaly-Detection-JEPA/
├── configs/
│   ├── config.yaml                    # base config
│   └── config.aggressive_v3.yaml     # JEPA v3 config (used for best model)
├── scripts/
│   ├── preprocess_data.py
│   ├── train_jepa.py
│   ├── train_full_pipeline.py
│   └── evaluate.py
├── src/
│   ├── data/
│   │   └── dataset.py
│   ├── models/
│   │   ├── jepa.py
│   │   └── isolation_forest.py
├── colab_gpu_pipeline.ipynb           # main notebook
└── README.md
```

---

## ⚡ Quick Start — Use Pretrained Models (Recommended)

If you just want to run inference with the saved models, follow these steps.

### 1. Download Pretrained Models

All pretrained model files are available in the shared Google Drive folder:

📂 **[Download All Models — Google Drive](https://drive.google.com/drive/folders/1ufz2Xjnt8EFKXe8-9-xVtBdD_H-c_jII?usp=sharing)**

| File | Description |
|---|---|
| `best_model.pt` | JEPA v3 encoder checkpoint (384-dim, trained 140 epochs) |
| `best_lof_model.pkl` | LOF detector (pca=130, manhattan, k=21) |
| `lof_local_refinement_summary.json` | Best config + calibrated threshold (1.2534) |

Download all three files to your Google Drive's `MyDrive/` root folder.

### 2. Open the Notebook in Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yaswanthreddyyyr/Drone-Anomaly-Detection-JEPA/blob/main/colab_gpu_pipeline.ipynb)

### 3. Set Runtime to GPU

`Runtime → Change runtime type → T4 GPU`

### 4. Run Setup Cells Only

Run these cells in order — they set up the environment and dataset:

| Cell | What it does |
|---|---|
| `# 1)` | Import libraries |
| `# 2)` | Check GPU availability |
| `# 3)` | Verify CUDA/cuDNN |
| `# 4)` | Configure GPU memory |
| `# 5)` | Set project paths |
| `# 6)` | Clone repo + install dependencies |
| `# 6.5)` | Restore dataset from Drive |
| `# 7)` | Wire dataset to project |
| `# 8)` | Patch config + preprocess data |
| `# 9)` | Smoke test forward pass |

### 5. Place Pretrained Models

After running setup cells, move the downloaded files into place:

```python
import shutil
from google.colab import drive
drive.mount('/content/drive')

shutil.copy("/content/drive/MyDrive/best_model.pt",
            "/content/Drone-Anomaly-Detection-JEPA/outputs/full_run_latest/jepa/run_latest/best_model.pt")
shutil.copy("/content/drive/MyDrive/best_lof_model.pkl",
            "/content/Drone-Anomaly-Detection-JEPA/outputs/best_lof_model.pkl")
shutil.copy("/content/drive/MyDrive/lof_local_refinement_summary.json",
            "/content/Drone-Anomaly-Detection-JEPA/outputs/lof_local_refinement_summary.json")
```

### 6. Run Inference

Paste and run this cell:

```python
import pickle, torch, json, numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from torch.utils.data import DataLoader
from src.data.dataset import DroneChunkDataset
from src.models.jepa import JEPA
from src.models.isolation_forest import EmbeddingExtractor
import yaml

# Load config
with open("configs/config.aggressive_v3.yaml") as f:
    cfg = yaml.safe_load(f)

# Load JEPA
device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = sorted(Path("outputs").glob("full_run_*/jepa/run_*/best_model.pt"))[-1]
train_ds = DroneChunkDataset("processed_data", split="train", return_labels=True)
train_normal = train_ds.get_normal_chunks()

model = JEPA(
    input_dim=train_ds.num_features,
    embed_dim=cfg["model"]["embedding_dim"],
    encoder_hidden=cfg["model"]["encoder_hidden"],
    predictor_hidden=cfg["model"]["predictor_hidden"],
    dropout=cfg["model"]["dropout"],
    min_mask_ratio=cfg["masking"]["min_mask_ratio"],
    max_mask_ratio=cfg["masking"]["max_mask_ratio"],
)
state = torch.load(ckpt, map_location="cpu")
model.load_state_dict(state["model_state_dict"] if "model_state_dict" in state else state)
model.eval()

# Load LOF + threshold
with open("outputs/best_lof_model.pkl", "rb") as f:
    lof = pickle.load(f)
with open("outputs/lof_local_refinement_summary.json") as f:
    summary = json.load(f)
threshold = summary["best"]["threshold"]

# Fit scaler + PCA on normal train embeddings
extractor = EmbeddingExtractor(model, device=device, pooling="cls")
train_loader = DataLoader(train_normal, batch_size=512, shuffle=False, num_workers=2)
train_emb, train_lbl, _ = extractor.extract_from_dataloader(train_loader, return_labels=True, verbose=False)
train_emb = train_emb[train_lbl == 0]

sc = StandardScaler()
pca = PCA(n_components=130, random_state=42)
pca.fit_transform(sc.fit_transform(train_emb))

# Run inference on all test splits
for split in ["test_balanced", "test_strong", "test_subtle"]:
    ds = DroneChunkDataset("processed_data", split=split, return_labels=True)
    dl = DataLoader(ds, batch_size=512, shuffle=False, num_workers=2)
    emb_s, lbl_s, _ = extractor.extract_from_dataloader(dl, return_labels=True, verbose=False)
    emb_st = pca.transform(sc.transform(emb_s))
    scores_s = -lof.score_samples(emb_st)
    preds_s = (scores_s >= threshold).astype(int)
    tn_s, fp_s, fn_s, tp_s = confusion_matrix(lbl_s, preds_s).ravel()
    auc_s = roc_auc_score(lbl_s, scores_s)
    print(f"\n{split}:")
    print(classification_report(lbl_s, preds_s, target_names=["Normal", "Anomaly"]))
    print(f"AUC={auc_s:.4f}  Recall={tp_s/(tp_s+fn_s):.4f}  FAR={fp_s/(fp_s+tn_s):.4f}")
    print(f"Anomalies caught: {tp_s} / {tp_s+fn_s}")
```

---

## 🔁 Full Training From Scratch

If you want to retrain the model yourself from scratch:

### Required
- Google Colab with GPU (T4 minimum, A100 recommended)
- Dataset: `drone-telemetry-tampering-dataset-v2.zip` on your Google Drive
- ~3–4 hours of GPU time for full JEPA v3 training

### Steps

Run cells in this order:

**Environment Setup**
- `# 1)` through `# 9)` — setup, dataset, preprocess, smoke test

**JEPA v3 Training**
- `# 9.5)` — set `RUN_SMOKE_TRAIN = True`, `REAL_EPOCHS = 60`
- `# 47)` — trains JEPA v3 (140 epochs, ~2–3 hrs on T4)

**LOF Detector Search**
- `# 62)` — sweeps 225 LOF configurations, saves best to `lof_local_refinement_summary.json`

**Save Models**
```python
import pickle, shutil
from google.colab import drive
drive.mount('/content/drive')

# Save LOF
with open("outputs/best_lof_model.pkl", "wb") as f:
    pickle.dump(lof, f)

# Copy to Drive
for f in ["outputs/best_lof_model.pkl", "outputs/lof_local_refinement_summary.json"]:
    shutil.copy(f, f"/content/drive/MyDrive/{Path(f).name}")

# Also save JEPA checkpoint
from pathlib import Path
ckpt = sorted(Path("outputs").glob("full_run_*/jepa/run_*/best_model.pt"))[-1]
shutil.copy(ckpt, "/content/drive/MyDrive/best_model.pt")
print("All saved!")
```

---

## 📦 Dataset

Dataset: **Drone Telemetry Tampering Dataset v2**

- 11 telemetry features per timestep
- Splits: `train` (normal only), `validation`, `test_balanced`, `test_strong`, `test_subtle`
- Upload `drone-telemetry-tampering-dataset-v2.zip` to your Google Drive
- The notebook auto-downloads and extracts it via Cell `# 6.5)`

---

## 🛠️ Requirements

All dependencies are installed automatically by Cell `# 6)`. Key packages:

```
torch
scikit-learn
numpy
pyyaml
tqdm
gdown
```

Python 3.10+ recommended. Tested on Google Colab with Python 3.12.

---

## ⚠️ Common Issues

| Problem | Fix |
|---|---|
| `CUDA out of memory` | Reduce batch size in Cell `# 9.5)` to 128 |
| `Dataset not found` | Check your Drive folder URL in Cell `# 5)` |
| `best_model.pt not found` | Run Cell `# 47)` first or load from Drive |
| `LOF expecting N features` | Make sure PCA n_components matches saved LOF (130) |
| Session disconnected mid-run | Re-run cells `# 1)` to `# 9)`, then load checkpoint from Drive |

---

## 📜 Citation

If you use this work, please cite:

```
@misc{drone-jepa-2026,
  author = {Yaswanth Reddy},
  title = {Drone Telemetry Anomaly Detection using JEPA},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/yaswanthreddyyyr/Drone-Anomaly-Detection-JEPA}
}
```

---

## 📬 Contact

For questions or issues, open a GitHub issue or reach out via the repository.

#!/usr/bin/env python3
"""
Generate presentation-ready anomaly/normal examples with confidence scores.

Pipeline:
1) Load JEPA checkpoint
2) Extract embeddings (train_normal, validation, target split)
3) Fit StandardScaler + PCA + LOF (champion defaults)
4) Calibrate threshold on validation with FAR cap
5) Score target split, compute confidence
6) Export top confident examples as figures and metadata

Usage:
  python scripts/generate_presentation_examples.py \
    --checkpoint smoke_outputs/run_20260420_103424/best_model.pt \
    --data-dir processed_data \
    --split test_balanced \
    --output-dir outputs/presentation_examples
"""

import argparse
import json
import sys
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import DroneChunkDataset  # noqa: E402
from src.models.jepa import JEPA  # noqa: E402
from src.models.isolation_forest import EmbeddingExtractor  # noqa: E402


@dataclass
class ExampleRow:
    split: str
    sample_idx: int
    true_label: int
    pred_label: int
    outcome: str
    anomaly_type: str
    score: float
    threshold: float
    confidence: float
    anomaly_fraction: float
    figure_path: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate anomaly/normal presentation examples")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to JEPA checkpoint")
    parser.add_argument("--data-dir", type=str, default="processed_data", help="Processed data directory")
    parser.add_argument("--split", type=str, default="test_balanced", help="Split to visualize")
    parser.add_argument("--output-dir", type=str, default="outputs/presentation_examples", help="Output folder")
    parser.add_argument("--batch-size", type=int, default=512, help="Embedding extraction batch size")
    parser.add_argument("--device", type=str, default="auto", help="cuda|cpu|mps|auto")
    parser.add_argument("--n-examples-per-class", type=int, default=8, help="Number of top confident examples for normal and anomaly")
    parser.add_argument("--pca-components", type=int, default=124, help="PCA components")
    parser.add_argument("--n-neighbors", type=int, default=22, help="LOF neighbors")
    parser.add_argument("--metric", type=str, default="manhattan", help="LOF metric")
    parser.add_argument("--far-cap", type=float, default=0.20, help="False alarm rate cap for threshold calibration")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    return parser.parse_args()


def choose_device(device_arg: str) -> str:
    if device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_model_from_checkpoint(checkpoint_path: Path, data_dir: Path) -> JEPA:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    config = checkpoint.get("config", {})

    model_cfg = config.get("model", {})
    masking_cfg = config.get("masking", {})

    train_ds = DroneChunkDataset(data_dir, split="train", return_labels=False)

    model = JEPA(
        input_dim=train_ds.num_features,
        embed_dim=model_cfg.get("embedding_dim", 256),
        encoder_hidden=model_cfg.get("encoder_hidden", [512, 256, 256]),
        predictor_hidden=model_cfg.get("predictor_hidden", [256, 256]),
        chunk_size=train_ds.chunk_size,
        dropout=model_cfg.get("dropout", 0.1),
        adaptive_masking=masking_cfg.get("adaptive", True),
        min_mask_ratio=masking_cfg.get("min_mask_ratio", 0.2),
        max_mask_ratio=masking_cfg.get("max_mask_ratio", 0.5),
        fixed_mask_ratio=masking_cfg.get("fixed_mask_ratio", 0.3),
    )

    state = checkpoint["model_state_dict"] if "model_state_dict" in checkpoint else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model


def get_embeddings(extractor: EmbeddingExtractor, dataset: DroneChunkDataset, batch_size: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    emb, labels, anomaly_types = extractor.extract_from_dataloader(loader, return_labels=True, verbose=False)
    return emb, labels, anomaly_types


def sanitize_embeddings(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = np.nan_to_num(x, nan=0.0, posinf=1e6, neginf=-1e6)
    x = np.clip(x, -1e4, 1e4)
    return x.astype(np.float32)


def calc_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = y_true.astype(int)
    y_pred = y_pred.astype(int)

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())

    recall = tp / (tp + fn) if (tp + fn) else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    far = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "far": far,
    }


def score_formula(recall: float, f1: float, far: float, auc_proxy: float = 0.0, cap: float = 0.20) -> float:
    return 1.60 * auc_proxy + 2.30 * recall + 0.20 * f1 - 3.2 * max(0.0, far - cap)


def calibrate_threshold(val_scores: np.ndarray, val_labels: np.ndarray, far_cap: float) -> Tuple[float, Dict[str, float]]:
    thresholds = np.quantile(val_scores, np.linspace(0.55, 0.998, 260))
    best_thr = float(np.quantile(val_scores, 0.95))
    best_score = -1e9
    best_metrics: Dict[str, float] = {}

    for thr in thresholds:
        pred = (val_scores >= thr).astype(int)
        m = calc_metrics(val_labels, pred)
        if m["far"] <= far_cap:
            sc = score_formula(m["recall"], m["f1"], m["far"], auc_proxy=0.0, cap=far_cap)
            if sc > best_score:
                best_score = sc
                best_thr = float(thr)
                best_metrics = m

    if not best_metrics:
        pred = (val_scores >= best_thr).astype(int)
        best_metrics = calc_metrics(val_labels, pred)

    return best_thr, best_metrics


def make_confidence(scores: np.ndarray, threshold: float) -> np.ndarray:
    dist = np.abs(scores - threshold)
    dist = np.log1p(dist)
    median = np.median(dist)
    mad = np.median(np.abs(dist - median)) + 1e-8
    z = np.clip((dist - median) / (1.4826 * mad), -8.0, 8.0)
    conf = 1.0 / (1.0 + np.exp(-z))
    return conf


def ensure_dirs(base: Path) -> Dict[str, Path]:
    paths = {
        "base": base,
        "normal": base / "normal",
        "anomaly": base / "anomaly",
        "hard_cases": base / "hard_cases",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def outcome_name(y_true: int, y_pred: int) -> str:
    if y_true == 1 and y_pred == 1:
        return "TP"
    if y_true == 0 and y_pred == 0:
        return "TN"
    if y_true == 0 and y_pred == 1:
        return "FP"
    return "FN"


def plot_chunk_figure(
    sample: Dict,
    row: ExampleRow,
    out_path: Path,
) -> None:
    x = sample["features"].numpy()
    t = np.arange(x.shape[0])

    lat = x[:, 0]
    lon = x[:, 1]
    alt = x[:, 2]
    speed = x[:, 3]
    heading = x[:, 4]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))

    axes[0, 0].plot(lon, lat, linewidth=1.5)
    axes[0, 0].set_title("Trajectory (Lon vs Lat)")
    axes[0, 0].set_xlabel("Longitude")
    axes[0, 0].set_ylabel("Latitude")
    axes[0, 0].grid(alpha=0.3)

    axes[0, 1].plot(t, alt, linewidth=1.2)
    axes[0, 1].set_title("Altitude over Time")
    axes[0, 1].set_xlabel("Timestep")
    axes[0, 1].set_ylabel("Altitude")
    axes[0, 1].grid(alpha=0.3)

    axes[1, 0].plot(t, speed, linewidth=1.2)
    axes[1, 0].set_title("Speed over Time")
    axes[1, 0].set_xlabel("Timestep")
    axes[1, 0].set_ylabel("Speed")
    axes[1, 0].grid(alpha=0.3)

    axes[1, 1].plot(t, heading, linewidth=1.2)
    axes[1, 1].set_title("Heading over Time")
    axes[1, 1].set_xlabel("Timestep")
    axes[1, 1].set_ylabel("Heading")
    axes[1, 1].grid(alpha=0.3)

    title = (
        f"{row.outcome} | true={row.true_label} pred={row.pred_label} | "
        f"score={row.score:.4f} thr={row.threshold:.4f} conf={row.confidence:.3f} | "
        f"type={row.anomaly_type}"
    )
    fig.suptitle(title, fontsize=11)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def select_indices(y_true: np.ndarray, y_pred: np.ndarray, conf: np.ndarray, scores: np.ndarray, n: int) -> Dict[str, np.ndarray]:
    idx = np.arange(len(y_true))

    tp = idx[(y_true == 1) & (y_pred == 1)]
    tn = idx[(y_true == 0) & (y_pred == 0)]
    fp = idx[(y_true == 0) & (y_pred == 1)]
    fn = idx[(y_true == 1) & (y_pred == 0)]

    def top_by_conf(indices: np.ndarray) -> np.ndarray:
        if len(indices) == 0:
            return indices
        finite = np.isfinite(scores[indices])
        candidate = indices[finite]
        if len(candidate) == 0:
            return candidate

        score_cap = np.percentile(np.abs(scores[candidate]), 99.5)
        candidate = candidate[np.abs(scores[candidate]) <= score_cap]
        if len(candidate) == 0:
            candidate = indices[finite]

        order = np.argsort(conf[candidate])[::-1]
        return candidate[order[:n]]

    return {
        "tp": top_by_conf(tp),
        "tn": top_by_conf(tn),
        "fp": top_by_conf(fp),
        "fn": top_by_conf(fn),
    }


def main() -> None:
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    warnings.filterwarnings("ignore", category=RuntimeWarning, module=r"sklearn.*")

    data_dir = Path(args.data_dir)
    ckpt_path = Path(args.checkpoint)
    out_dir = Path(args.output_dir)

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = choose_device(args.device)

    print("=" * 100)
    print("GENERATE PRESENTATION EXAMPLES")
    print("=" * 100)
    print(f"Checkpoint: {ckpt_path}")
    print(f"Data dir:    {data_dir}")
    print(f"Split:       {args.split}")
    print(f"Device:      {device}")

    model = build_model_from_checkpoint(ckpt_path, data_dir)
    extractor = EmbeddingExtractor(model, device=device, pooling="cls")

    train_ds = DroneChunkDataset(data_dir, split="train", return_labels=True)
    train_normal = train_ds.get_normal_chunks()
    val_ds = DroneChunkDataset(data_dir, split="validation", return_labels=True)
    target_ds = DroneChunkDataset(data_dir, split=args.split, return_labels=True)

    print("Extracting embeddings...")
    train_emb, train_lbl, _ = get_embeddings(extractor, train_normal, args.batch_size)
    val_emb, val_lbl, _ = get_embeddings(extractor, val_ds, args.batch_size)
    test_emb, test_lbl, test_types = get_embeddings(extractor, target_ds, args.batch_size)

    train_emb = sanitize_embeddings(train_emb)
    val_emb = sanitize_embeddings(val_emb)
    test_emb = sanitize_embeddings(test_emb)

    train_emb = train_emb[train_lbl == 0]

    scaler = StandardScaler()
    train_s = scaler.fit_transform(train_emb)
    val_s = scaler.transform(val_emb)
    test_s = scaler.transform(test_emb)

    train_s = sanitize_embeddings(train_s)
    val_s = sanitize_embeddings(val_s)
    test_s = sanitize_embeddings(test_s)

    finite_mask = np.isfinite(train_s).all(axis=1)
    if finite_mask.sum() < len(train_s):
        train_s = train_s[finite_mask]

    n_comp = min(args.pca_components, train_s.shape[1], train_s.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=args.seed, svd_solver="randomized")
    train_t = pca.fit_transform(train_s)
    val_t = pca.transform(val_s)
    test_t = pca.transform(test_s)

    train_t = sanitize_embeddings(train_t)
    val_t = sanitize_embeddings(val_t)
    test_t = sanitize_embeddings(test_t)

    lof = LocalOutlierFactor(
        n_neighbors=min(args.n_neighbors, len(train_t) - 1),
        novelty=True,
        metric=args.metric,
    )
    lof.fit(train_t)

    val_scores = -lof.score_samples(val_t)
    threshold, val_metrics = calibrate_threshold(val_scores, val_lbl, args.far_cap)

    test_scores = -lof.score_samples(test_t)
    test_pred = (test_scores >= threshold).astype(int)
    test_metrics = calc_metrics(test_lbl, test_pred)
    conf = make_confidence(test_scores, threshold)

    print(f"Calibrated threshold: {threshold:.6f}")
    print("Validation metrics:", json.dumps(val_metrics, indent=2))
    print("Target split metrics:", json.dumps(test_metrics, indent=2))

    paths = ensure_dirs(out_dir / args.split)
    selected = select_indices(test_lbl, test_pred, conf, test_scores, args.n_examples_per_class)

    rows: List[ExampleRow] = []

    # Strong positives (pred anomaly) and strong normals (pred normal)
    key_map = {
        "tp": (paths["anomaly"], "anomaly"),
        "tn": (paths["normal"], "normal"),
        "fp": (paths["hard_cases"], "hard_case"),
        "fn": (paths["hard_cases"], "hard_case"),
    }

    for bucket in ["tp", "tn", "fp", "fn"]:
        save_dir, prefix = key_map[bucket]
        for rank, i in enumerate(selected[bucket], start=1):
            sample = target_ds[int(i)]
            y_true = int(test_lbl[i])
            y_pred = int(test_pred[i])
            outcome = outcome_name(y_true, y_pred)

            fig_name = f"{prefix}_{outcome.lower()}_{rank:02d}_idx{i}.png"
            fig_path = save_dir / fig_name

            row = ExampleRow(
                split=args.split,
                sample_idx=int(i),
                true_label=y_true,
                pred_label=y_pred,
                outcome=outcome,
                anomaly_type=str(test_types[i]),
                score=float(test_scores[i]),
                threshold=float(threshold),
                confidence=float(conf[i]),
                anomaly_fraction=float(sample["labels"].float().mean().item()),
                figure_path=str(fig_path.relative_to(out_dir)),
            )

            plot_chunk_figure(sample, row, fig_path)
            rows.append(row)

    summary = {
        "checkpoint": str(ckpt_path),
        "split": args.split,
        "params": {
            "pca_components": int(n_comp),
            "n_neighbors": int(args.n_neighbors),
            "metric": args.metric,
            "far_cap": float(args.far_cap),
            "threshold": float(threshold),
        },
        "validation_metrics": val_metrics,
        "target_metrics": test_metrics,
        "counts": {
            "tp_selected": int(len(selected["tp"])),
            "tn_selected": int(len(selected["tn"])),
            "fp_selected": int(len(selected["fp"])),
            "fn_selected": int(len(selected["fn"])),
        },
        "examples": [asdict(r) for r in rows],
    }

    with open(out_dir / args.split / "examples_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # CSV export
    csv_path = out_dir / args.split / "examples_table.csv"
    with open(csv_path, "w") as f:
        f.write(
            "split,sample_idx,true_label,pred_label,outcome,anomaly_type,score,threshold,confidence,anomaly_fraction,figure_path\n"
        )
        for r in rows:
            f.write(
                f"{r.split},{r.sample_idx},{r.true_label},{r.pred_label},{r.outcome},{r.anomaly_type},"
                f"{r.score:.8f},{r.threshold:.8f},{r.confidence:.6f},{r.anomaly_fraction:.6f},{r.figure_path}\n"
            )

    print("=" * 100)
    print(f"Saved figures and metadata to: {out_dir / args.split}")
    print(f"- JSON: {out_dir / args.split / 'examples_summary.json'}")
    print(f"- CSV:  {csv_path}")
    print(f"- Normal figures:  {paths['normal']}")
    print(f"- Anomaly figures: {paths['anomaly']}")
    print(f"- Hard cases:      {paths['hard_cases']}")
    print("=" * 100)


if __name__ == "__main__":
    main()

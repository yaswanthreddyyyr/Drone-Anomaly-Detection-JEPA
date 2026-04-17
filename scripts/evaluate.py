#!/usr/bin/env python3
"""
JEPA-DRONE Evaluation Script

Comprehensive evaluation of trained models including:
- Per-anomaly-type analysis
- GPS spoofing (injection) detection metrics
- Difficulty-level analysis (balanced/strong/subtle)
- Visualization of results

Usage:
    python scripts/evaluate.py --run-dir outputs/full_run_xxx
    python scripts/evaluate.py --jepa-checkpoint path/to/jepa.pt --detector-path path/to/detector.pkl
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate JEPA-DRONE anomaly detection"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        default=None,
        help="Path to training run directory (contains jepa/ and anomaly_detector.pkl)"
    )
    parser.add_argument(
        "--jepa-checkpoint",
        type=str,
        default=None,
        help="Path to JEPA checkpoint (alternative to --run-dir)"
    )
    parser.add_argument(
        "--detector-path",
        type=str,
        default=None,
        help="Path to anomaly detector (alternative to --run-dir)"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="processed_data",
        help="Path to processed data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: run-dir or outputs/evaluation)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for inference"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save per-sample predictions"
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        default=True,
        help="Generate evaluation plots"
    )
    return parser.parse_args()


def print_table(headers, rows, col_widths=None):
    """Print a formatted table."""
    if col_widths is None:
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) + 2 
                      for i in range(len(headers))]
    
    header_str = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
    print(header_str)
    print("-" * len(header_str))
    
    for row in rows:
        row_str = "".join(str(v).ljust(w) for v, w in zip(row, col_widths))
        print(row_str)


def main():
    args = parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                    JEPA-DRONE                             ║
    ║          Anomaly Detection Evaluation                     ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    from sklearn.metrics import (
        roc_auc_score, precision_recall_curve, average_precision_score,
        f1_score, confusion_matrix, roc_curve
    )
    
    from src.data.dataset import DroneChunkDataset
    from src.models.jepa import JEPA
    from src.models.isolation_forest import AnomalyDetector, EmbeddingExtractor
    
    # Determine paths
    if args.run_dir:
        run_dir = Path(args.run_dir)
        jepa_path = run_dir / "jepa" / "best.pt"
        detector_path = run_dir / "anomaly_detector.pkl"
        output_dir = args.output_dir or run_dir / "evaluation"
    else:
        if not args.jepa_checkpoint or not args.detector_path:
            print("❌ Error: Either --run-dir or both --jepa-checkpoint and --detector-path required")
            sys.exit(1)
        jepa_path = Path(args.jepa_checkpoint)
        detector_path = Path(args.detector_path)
        output_dir = Path(args.output_dir or "outputs/evaluation")
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_dir = Path(args.data_dir)
    
    print(f"📁 JEPA checkpoint: {jepa_path}")
    print(f"📁 Anomaly detector: {detector_path}")
    print(f"📁 Data directory: {data_dir}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # Load model and detector
    print("Loading models...")
    checkpoint = torch.load(jepa_path, map_location="cpu")
    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    
    # Determine input dimension from data
    sample_ds = DroneChunkDataset(data_dir, split="train", return_labels=False)
    input_dim = sample_ds.num_features
    masking_config = config.get("masking", {})
    
    model = JEPA(
        input_dim=input_dim,
        embed_dim=model_config.get("embedding_dim", 256),
        encoder_hidden=model_config.get("encoder_hidden", [512, 256, 256]),
        predictor_hidden=model_config.get("predictor_hidden", [256, 256]),
        chunk_size=sample_ds.chunk_size,
        dropout=model_config.get("dropout", 0.1),
        adaptive_masking=masking_config.get("adaptive", True),
        min_mask_ratio=masking_config.get("min_mask_ratio", 0.20),
        max_mask_ratio=masking_config.get("max_mask_ratio", 0.50),
        fixed_mask_ratio=masking_config.get("fixed_mask_ratio", 0.30)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    detector = AnomalyDetector.load(detector_path)
    extractor = EmbeddingExtractor(model, device=args.device)
    
    print("✅ Models loaded successfully")
    print()
    
    # Load test datasets
    print("Loading test datasets...")
    test_splits = ["validation", "test", "test_balanced", "test_strong", "test_subtle"]
    datasets = {}
    
    for split in test_splits:
        try:
            ds = DroneChunkDataset(data_dir, split=split, return_labels=True)
            datasets[split] = ds
            print(f"  ✅ {split}: {len(ds):,} chunks")
        except Exception as e:
            print(f"  ⚠️  {split}: not found")
    
    if not datasets:
        print("❌ Error: No test datasets found")
        sys.exit(1)
    
    print()
    
    # Evaluate each dataset
    all_results = {}
    
    for split_name, dataset in datasets.items():
        print("="*60)
        print(f"Evaluating: {split_name}")
        print("="*60)
        
        # Create dataloader
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=4
        )
        
        # Extract embeddings
        embeddings, labels, anomaly_types = extractor.extract_from_dataloader(
            loader,
            return_labels=True,
            verbose=True
        )
        
        # Get scores and predictions
        scores = detector.score_samples(embeddings)
        predictions = detector.predict(embeddings)
        
        # Overall metrics
        print(f"\n📊 Overall Metrics:")
        
        if len(np.unique(labels)) > 1:
            auc_roc = roc_auc_score(labels, -scores)
            auc_pr = average_precision_score(labels, -scores)
        else:
            auc_roc = auc_pr = 0.0
        
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results = {
            "n_samples": len(labels),
            "n_normal": int((labels == 0).sum()),
            "n_anomaly": int((labels == 1).sum()),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "recall": float(recall),
            "precision": float(precision),
            "f1_score": float(f1),
            "false_alarm_rate": float(far),
            "specificity": float(specificity),
            "confusion_matrix": {
                "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
            }
        }
        
        print(f"  Samples: {results['n_samples']:,} (Normal: {results['n_normal']:,}, Anomaly: {results['n_anomaly']:,})")
        print(f"  AUC-ROC: {auc_roc:.4f}")
        print(f"  AUC-PR: {auc_pr:.4f}")
        print(f"  Recall: {recall:.4f}")
        print(f"  Precision: {precision:.4f}")
        print(f"  F1 Score: {f1:.4f}")
        print(f"  False Alarm Rate: {far:.4f}")
        
        # Per-anomaly-type analysis
        if anomaly_types is not None:
            print(f"\n📋 Per-Anomaly-Type Analysis:")
            
            unique_types = np.unique(anomaly_types)
            type_results = {}
            
            headers = ["Anomaly Type", "Samples", "AUC-ROC", "Recall", "Precision", "F1"]
            rows = []
            
            for atype in sorted(unique_types):
                if atype == "normal":
                    continue
                
                mask = anomaly_types == atype
                if mask.sum() == 0:
                    continue
                
                type_labels = labels[mask]
                type_scores = scores[mask]
                type_preds = predictions[mask]
                
                # Calculate metrics
                if len(np.unique(type_labels)) > 1:
                    type_auc = roc_auc_score(type_labels, -type_scores)
                else:
                    type_auc = 0.0
                
                type_tp = ((type_preds == 1) & (type_labels == 1)).sum()
                type_fp = ((type_preds == 1) & (type_labels == 0)).sum()
                type_fn = ((type_preds == 0) & (type_labels == 1)).sum()
                
                type_recall = type_tp / (type_tp + type_fn) if (type_tp + type_fn) > 0 else 0
                type_precision = type_tp / (type_tp + type_fp) if (type_tp + type_fp) > 0 else 0
                type_f1 = 2 * type_precision * type_recall / (type_precision + type_recall) if (type_precision + type_recall) > 0 else 0
                
                type_results[atype] = {
                    "n_samples": int(mask.sum()),
                    "auc_roc": float(type_auc),
                    "recall": float(type_recall),
                    "precision": float(type_precision),
                    "f1_score": float(type_f1)
                }
                
                rows.append([
                    atype,
                    f"{mask.sum():,}",
                    f"{type_auc:.4f}",
                    f"{type_recall:.4f}",
                    f"{type_precision:.4f}",
                    f"{type_f1:.4f}"
                ])
            
            print()
            print_table(headers, rows, col_widths=[25, 10, 10, 10, 10, 10])
            results["per_type"] = type_results
            
            # GPS Spoofing (injection) specific analysis
            if "injection" in type_results:
                print(f"\n🎯 GPS Spoofing (Injection) Detection:")
                inj_results = type_results["injection"]
                print(f"  - AUC-ROC: {inj_results['auc_roc']:.4f}")
                print(f"  - Recall: {inj_results['recall']:.4f} (Detection Rate)")
                print(f"  - Zero-shot detection: {'✅ YES' if inj_results['recall'] > 0.5 else '⚠️ PARTIAL'}")
        
        all_results[split_name] = results
        
        # Save predictions if requested
        if args.save_predictions:
            preds_data = {
                "scores": scores.tolist(),
                "predictions": predictions.tolist(),
                "labels": labels.tolist(),
                "anomaly_types": anomaly_types.tolist() if anomaly_types is not None else None
            }
            with open(output_dir / f"{split_name}_predictions.json", "w") as f:
                json.dump(preds_data, f)
    
    # Generate comparison table
    print("\n" + "="*60)
    print("SUMMARY: All Test Sets")
    print("="*60 + "\n")
    
    headers = ["Dataset", "Samples", "AUC-ROC", "Recall", "Precision", "F1", "FAR"]
    rows = []
    for name, res in all_results.items():
        rows.append([
            name,
            f"{res['n_samples']:,}",
            f"{res['auc_roc']:.4f}",
            f"{res['recall']:.4f}",
            f"{res['precision']:.4f}",
            f"{res['f1_score']:.4f}",
            f"{res['false_alarm_rate']:.4f}"
        ])
    
    print_table(headers, rows, col_widths=[20, 12, 10, 10, 10, 10, 10])
    
    # Check against proposal targets
    print("\n" + "="*60)
    print("TARGET COMPARISON (from proposal)")
    print("="*60)
    
    # Use first available test set for comparison
    primary_results = all_results.get("test_balanced", all_results.get("test", list(all_results.values())[0]))
    
    targets = {
        "AUC-ROC": (">0.90", 0.90, primary_results["auc_roc"]),
        "Recall": (">85%", 0.85, primary_results["recall"]),
        "Precision": (">80%", 0.80, primary_results["precision"]),
        "FAR": ("<5%", 0.05, primary_results["false_alarm_rate"])
    }
    
    print(f"\n{'Metric':<15} {'Target':<10} {'Achieved':<10} {'Status':<10}")
    print("-"*50)
    
    all_met = True
    for metric, (target_str, target_val, achieved) in targets.items():
        if metric == "FAR":
            met = achieved <= target_val
        else:
            met = achieved >= target_val
        
        status = "✅ MET" if met else "❌ MISS"
        all_met = all_met and met
        
        if metric == "FAR":
            print(f"{metric:<15} {target_str:<10} {achieved:.2%}{'':>4} {status:<10}")
        else:
            print(f"{metric:<15} {target_str:<10} {achieved:.4f}{'':>4} {status:<10}")
    
    print("-"*50)
    print(f"Overall: {'✅ ALL TARGETS MET!' if all_met else '⚠️ Some targets missed'}")
    
    # Save all results
    with open(output_dir / "evaluation_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n📁 Results saved to: {output_dir / 'evaluation_results.json'}")
    
    # Generate plots
    if args.generate_plots:
        try:
            generate_plots(all_results, output_dir)
            print(f"📊 Plots saved to: {output_dir}")
        except ImportError:
            print("⚠️  matplotlib not available, skipping plots")
    
    print("\n✅ Evaluation complete!")


def generate_plots(results: dict, output_dir: Path):
    """Generate evaluation visualizations."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 1. Metrics comparison across datasets
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    datasets = list(results.keys())
    metrics = ["auc_roc", "recall", "precision", "f1_score"]
    metric_labels = ["AUC-ROC", "Recall", "Precision", "F1"]
    
    x = np.arange(len(datasets))
    width = 0.2
    
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[d][metric] for d in datasets]
        axes[0].bar(x + i * width, values, width, label=label)
    
    axes[0].set_ylabel("Score")
    axes[0].set_xlabel("Dataset")
    axes[0].set_title("Performance Metrics by Dataset")
    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(datasets, rotation=45, ha="right")
    axes[0].legend()
    axes[0].set_ylim(0, 1.1)
    axes[0].axhline(y=0.90, color='r', linestyle='--', alpha=0.5, label="AUC Target")
    axes[0].axhline(y=0.85, color='g', linestyle='--', alpha=0.5, label="Recall Target")
    
    # 2. Per-type performance (if available)
    primary_ds = list(results.keys())[0]
    if "per_type" in results[primary_ds]:
        per_type = results[primary_ds]["per_type"]
        types = list(per_type.keys())
        recalls = [per_type[t]["recall"] for t in types]
        aucs = [per_type[t]["auc_roc"] for t in types]
        
        x = np.arange(len(types))
        axes[1].bar(x - 0.2, aucs, 0.4, label="AUC-ROC", color="steelblue")
        axes[1].bar(x + 0.2, recalls, 0.4, label="Recall", color="coral")
        
        axes[1].set_ylabel("Score")
        axes[1].set_xlabel("Anomaly Type")
        axes[1].set_title("Per-Anomaly-Type Performance")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(types, rotation=45, ha="right")
        axes[1].legend()
        axes[1].set_ylim(0, 1.1)
        axes[1].axhline(y=0.85, color='r', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "evaluation_metrics.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 3. Confusion matrices
    fig, axes = plt.subplots(1, min(len(datasets), 3), figsize=(5*min(len(datasets), 3), 4))
    if len(datasets) == 1:
        axes = [axes]
    
    for ax, (name, res) in zip(axes, list(results.items())[:3]):
        cm = res["confusion_matrix"]
        cm_array = np.array([[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]])
        
        im = ax.imshow(cm_array, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Normal", "Anomaly"])
        ax.set_yticklabels(["Normal", "Anomaly"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"{name}")
        
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm_array[i, j]:,}", ha="center", va="center",
                       color="white" if cm_array[i, j] > cm_array.max()/2 else "black")
    
    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrices.png", dpi=150, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

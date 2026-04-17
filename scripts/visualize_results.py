#!/usr/bin/env python3
"""
JEPA-DRONE: Visualization and Analysis Script

Generates comprehensive visualizations for the trained model:
1. Training curves
2. Embedding visualization (t-SNE/PCA)
3. Score distributions
4. Per-anomaly-type analysis
5. ROC and PR curves
"""

import argparse
import json
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Visualize JEPA-DRONE training and evaluation results"
    )
    parser.add_argument(
        "--run-dir",
        type=str,
        required=True,
        help="Path to training run directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for visualizations (default: run-dir/visualizations)"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Number of samples for embedding visualization"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device for embedding extraction"
    )
    return parser.parse_args()


def plot_training_curves(history: dict, output_dir: Path):
    """Plot training and validation loss curves."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history["train_losses"]) + 1)
    
    # Training loss
    axes[0].plot(epochs, history["train_losses"], 'b-', label='Train Loss', linewidth=2)
    axes[0].plot(epochs, history["val_losses"], 'r-', label='Val Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Log scale
    axes[1].semilogy(epochs, history["train_losses"], 'b-', label='Train Loss', linewidth=2)
    axes[1].semilogy(epochs, history["val_losses"], 'r-', label='Val Loss', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss (log scale)')
    axes[1].set_title('Training and Validation Loss (Log Scale)')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved training_curves.png")


def plot_embedding_visualization(embeddings, labels, anomaly_types, output_dir: Path, method='tsne'):
    """Visualize embeddings using t-SNE or PCA."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    
    print(f"  Computing {method.upper()} projection...")
    
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, max_iter=1000)
    else:
        reducer = PCA(n_components=2, random_state=42)
    
    coords = reducer.fit_transform(embeddings)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # By normal vs anomaly
    normal_mask = labels == 0
    axes[0].scatter(coords[normal_mask, 0], coords[normal_mask, 1], 
                   c='blue', alpha=0.3, s=10, label='Normal')
    axes[0].scatter(coords[~normal_mask, 0], coords[~normal_mask, 1], 
                   c='red', alpha=0.3, s=10, label='Anomaly')
    axes[0].set_title(f'{method.upper()}: Normal vs Anomaly')
    axes[0].legend()
    axes[0].set_xlabel('Component 1')
    axes[0].set_ylabel('Component 2')
    
    # By anomaly type
    unique_types = np.unique(anomaly_types)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_types)))
    
    for atype, color in zip(unique_types, colors):
        mask = anomaly_types == atype
        axes[1].scatter(coords[mask, 0], coords[mask, 1], 
                       c=[color], alpha=0.4, s=10, label=atype)
    
    axes[1].set_title(f'{method.upper()}: By Anomaly Type')
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].set_xlabel('Component 1')
    axes[1].set_ylabel('Component 2')
    
    plt.tight_layout()
    plt.savefig(output_dir / f"embedding_{method}.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved embedding_{method}.png")


def plot_score_distribution(scores, labels, anomaly_types, output_dir: Path):
    """Plot anomaly score distributions."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    normal_scores = scores[labels == 0]
    anomaly_scores = scores[labels == 1]
    
    # Overall distribution
    axes[0, 0].hist(normal_scores, bins=50, alpha=0.6, label='Normal', density=True, color='blue')
    axes[0, 0].hist(anomaly_scores, bins=50, alpha=0.6, label='Anomaly', density=True, color='red')
    axes[0, 0].set_xlabel('Anomaly Score')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Score Distribution: Normal vs Anomaly')
    axes[0, 0].legend()
    axes[0, 0].axvline(x=np.median(normal_scores), color='blue', linestyle='--', alpha=0.8)
    axes[0, 0].axvline(x=np.median(anomaly_scores), color='red', linestyle='--', alpha=0.8)
    
    # Box plot
    data = [normal_scores, anomaly_scores]
    bp = axes[0, 1].boxplot(data, labels=['Normal', 'Anomaly'], patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    bp['boxes'][1].set_facecolor('lightcoral')
    axes[0, 1].set_ylabel('Anomaly Score')
    axes[0, 1].set_title('Score Distribution (Box Plot)')
    
    # Per anomaly type
    unique_types = [t for t in np.unique(anomaly_types) if t != 'normal']
    type_scores = [scores[anomaly_types == t] for t in unique_types]
    
    axes[1, 0].boxplot(type_scores, labels=[t[:10] for t in unique_types], patch_artist=True)
    axes[1, 0].set_ylabel('Anomaly Score')
    axes[1, 0].set_title('Score by Anomaly Type')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Separation analysis
    separations = []
    for t in unique_types:
        type_s = scores[anomaly_types == t]
        # Cohen's d effect size
        d = (np.mean(normal_scores) - np.mean(type_s)) / np.sqrt((np.std(normal_scores)**2 + np.std(type_s)**2) / 2)
        separations.append(abs(d))
    
    colors = ['green' if s > 0.5 else 'orange' if s > 0.2 else 'red' for s in separations]
    axes[1, 1].barh(range(len(unique_types)), separations, color=colors)
    axes[1, 1].set_yticks(range(len(unique_types)))
    axes[1, 1].set_yticklabels([t[:15] for t in unique_types])
    axes[1, 1].set_xlabel("Cohen's d (effect size)")
    axes[1, 1].set_title('Score Separation from Normal')
    axes[1, 1].axvline(x=0.5, color='green', linestyle='--', alpha=0.5, label='Medium effect')
    axes[1, 1].axvline(x=0.2, color='orange', linestyle='--', alpha=0.5, label='Small effect')
    
    plt.tight_layout()
    plt.savefig(output_dir / "score_distribution.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved score_distribution.png")


def plot_roc_pr_curves(scores, labels, output_dir: Path):
    """Plot ROC and Precision-Recall curves."""
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import roc_curve, precision_recall_curve, auc
    
    # Negate scores (lower = more anomalous)
    fpr, tpr, _ = roc_curve(labels, -scores)
    precision, recall, _ = precision_recall_curve(labels, -scores)
    
    roc_auc = auc(fpr, tpr)
    pr_auc = auc(recall, precision)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # ROC curve
    axes[0].plot(fpr, tpr, 'b-', linewidth=2, label=f'JEPA-DRONE (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random (AUC = 0.5)')
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title('ROC Curve')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # PR curve
    baseline = labels.sum() / len(labels)
    axes[1].plot(recall, precision, 'b-', linewidth=2, label=f'JEPA-DRONE (AUC = {pr_auc:.3f})')
    axes[1].axhline(y=baseline, color='k', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.2f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title('Precision-Recall Curve')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_dir / "roc_pr_curves.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved roc_pr_curves.png")


def plot_per_type_results(results: dict, output_dir: Path):
    """Plot per-anomaly-type results comparison."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Get per-type results from first available test set
    per_type = None
    for key in ['test_balanced', 'test_strong', 'test_subtle', 'validation']:
        if key in results and 'per_type' in results[key]:
            per_type = results[key]['per_type']
            break
    
    if per_type is None:
        print("  ⚠️ No per-type results found")
        return
    
    types = list(per_type.keys())
    auc_rocs = [per_type[t]['auc_roc'] for t in types]
    recalls = [per_type[t]['recall'] for t in types]
    
    # Sort by AUC-ROC
    sorted_idx = np.argsort(auc_rocs)[::-1]
    types = [types[i] for i in sorted_idx]
    auc_rocs = [auc_rocs[i] for i in sorted_idx]
    recalls = [recalls[i] for i in sorted_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # AUC-ROC by type
    colors = ['green' if a > 0.6 else 'orange' if a > 0.5 else 'red' for a in auc_rocs]
    axes[0].barh(range(len(types)), auc_rocs, color=colors)
    axes[0].set_yticks(range(len(types)))
    axes[0].set_yticklabels(types)
    axes[0].set_xlabel('AUC-ROC')
    axes[0].set_title('AUC-ROC by Anomaly Type')
    axes[0].axvline(x=0.5, color='black', linestyle='--', alpha=0.5, label='Random')
    axes[0].axvline(x=0.9, color='green', linestyle='--', alpha=0.5, label='Target')
    axes[0].set_xlim([0, 1])
    axes[0].legend()
    
    # Recall by type
    colors = ['green' if r > 0.5 else 'orange' if r > 0.1 else 'red' for r in recalls]
    axes[1].barh(range(len(types)), recalls, color=colors)
    axes[1].set_yticks(range(len(types)))
    axes[1].set_yticklabels(types)
    axes[1].set_xlabel('Recall')
    axes[1].set_title('Recall by Anomaly Type')
    axes[1].axvline(x=0.85, color='green', linestyle='--', alpha=0.5, label='Target (85%)')
    axes[1].set_xlim([0, 1])
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "per_type_results.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved per_type_results.png")


def plot_difficulty_comparison(results: dict, output_dir: Path):
    """Compare performance across difficulty levels."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    difficulties = ['test_balanced', 'test_strong', 'test_subtle']
    available = [d for d in difficulties if d in results]
    
    if len(available) < 2:
        print("  ⚠️ Not enough difficulty levels for comparison")
        return
    
    metrics = ['auc_roc', 'recall', 'precision', 'f1_score', 'false_alarm_rate']
    metric_labels = ['AUC-ROC', 'Recall', 'Precision', 'F1 Score', 'FAR']
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, diff in enumerate(available):
        values = [results[diff][m] for m in metrics]
        label = diff.replace('test_', '').capitalize()
        ax.bar(x + i * width, values, width, label=label)
    
    ax.set_ylabel('Score')
    ax.set_xlabel('Metric')
    ax.set_title('Performance by Difficulty Level')
    ax.set_xticks(x + width)
    ax.set_xticklabels(metric_labels)
    ax.legend()
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add target lines
    ax.axhline(y=0.9, color='green', linestyle='--', alpha=0.3, xmin=0, xmax=0.2)
    ax.axhline(y=0.85, color='green', linestyle='--', alpha=0.3, xmin=0.2, xmax=0.4)
    
    plt.tight_layout()
    plt.savefig(output_dir / "difficulty_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved difficulty_comparison.png")


def main():
    args = parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                    JEPA-DRONE                             ║
    ║              Results Visualization                        ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    
    from src.data.dataset import DroneChunkDataset
    from src.models.jepa import JEPA
    from src.models.isolation_forest import AnomalyDetector, EmbeddingExtractor
    
    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "visualizations"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Run directory: {run_dir}")
    print(f"📁 Output directory: {output_dir}")
    print()
    
    # ==================== Training Curves ====================
    print("📈 Generating training curves...")
    
    # Find history file
    history_files = list(run_dir.glob("**/history.json"))
    if history_files:
        with open(history_files[0]) as f:
            history = json.load(f)
        plot_training_curves(history, output_dir)
    else:
        print("  ⚠️ No history.json found, skipping training curves")
    
    # ==================== Load Results ====================
    print("\n📊 Loading evaluation results...")
    
    results_file = run_dir / "evaluation" / "evaluation_results.json"
    if not results_file.exists():
        results_file = run_dir / "test_results.json"
    
    if results_file.exists():
        with open(results_file) as f:
            results = json.load(f)
        
        # Plot per-type results
        print("\n📋 Generating per-type analysis...")
        plot_per_type_results(results, output_dir)
        
        # Plot difficulty comparison
        print("\n📊 Generating difficulty comparison...")
        plot_difficulty_comparison(results, output_dir)
    else:
        print("  ⚠️ No evaluation results found")
        results = {}
    
    # ==================== Embedding Visualization ====================
    print("\n🔮 Generating embedding visualizations...")
    
    # Find checkpoint
    checkpoint_files = list(run_dir.glob("**/best.pt"))
    detector_file = run_dir / "anomaly_detector.pkl"
    
    if checkpoint_files and detector_file.exists():
        checkpoint_path = checkpoint_files[0]
        print(f"  Loading checkpoint: {checkpoint_path}")
        
        # Load model
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        config = checkpoint.get("config", {})
        model_config = config.get("model", {})
        masking_config = config.get("masking", {})
        
        # Load dataset
        dataset = DroneChunkDataset("processed_data", split="validation", return_labels=True)
        
        model = JEPA(
            input_dim=dataset.num_features,
            embed_dim=model_config.get("embedding_dim", 256),
            encoder_hidden=model_config.get("encoder_hidden", [512, 256, 256]),
            predictor_hidden=model_config.get("predictor_hidden", [256, 256]),
            chunk_size=dataset.chunk_size,
            dropout=model_config.get("dropout", 0.1),
            adaptive_masking=masking_config.get("adaptive", True),
            min_mask_ratio=masking_config.get("min_mask_ratio", 0.20),
            max_mask_ratio=masking_config.get("max_mask_ratio", 0.50),
            fixed_mask_ratio=masking_config.get("fixed_mask_ratio", 0.30)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        
        # Load detector
        detector = AnomalyDetector.load(detector_file)
        
        # Extract embeddings
        extractor = EmbeddingExtractor(model, device=args.device)
        
        # Sample subset for visualization
        n_samples = min(args.n_samples, len(dataset))
        indices = np.random.choice(len(dataset), n_samples, replace=False)
        
        sampled_features = dataset.features[indices]
        sampled_labels = dataset.chunk_labels[indices].numpy()
        sampled_types = dataset.anomaly_types[indices]
        
        # Extract embeddings
        print(f"  Extracting embeddings for {n_samples} samples...")
        with torch.no_grad():
            embeddings = extractor.extract(sampled_features)
        embeddings = embeddings.cpu().numpy() if torch.is_tensor(embeddings) else embeddings
        
        # Get anomaly scores
        scores = detector.score_samples(embeddings)
        
        # Embedding visualization
        print("\n  Creating t-SNE visualization...")
        plot_embedding_visualization(embeddings, sampled_labels, sampled_types, output_dir, method='tsne')
        
        print("  Creating PCA visualization...")
        plot_embedding_visualization(embeddings, sampled_labels, sampled_types, output_dir, method='pca')
        
        # Score distribution
        print("\n📊 Generating score distribution plots...")
        plot_score_distribution(scores, sampled_labels, sampled_types, output_dir)
        
        # ROC and PR curves
        print("\n📈 Generating ROC and PR curves...")
        plot_roc_pr_curves(scores, sampled_labels, output_dir)
    else:
        print("  ⚠️ Checkpoint or detector not found, skipping embedding visualization")
    
    # ==================== Summary ====================
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"\n📁 All visualizations saved to: {output_dir}")
    
    # List generated files
    print("\nGenerated files:")
    for f in output_dir.glob("*.png"):
        print(f"  - {f.name}")
    
    # Summary statistics
    if results:
        print("\n📊 Quick Summary:")
        primary = results.get('test_balanced', results.get('validation', {}))
        if primary:
            print(f"  AUC-ROC: {primary.get('auc_roc', 0):.4f} (Target: >0.90)")
            print(f"  Recall: {primary.get('recall', 0):.4f} (Target: >0.85)")
            print(f"  Precision: {primary.get('precision', 0):.4f} (Target: >0.80)")
            print(f"  F1 Score: {primary.get('f1_score', 0):.4f}")
            print(f"  FAR: {primary.get('false_alarm_rate', 0):.4f} (Target: <0.05)")


if __name__ == "__main__":
    main()

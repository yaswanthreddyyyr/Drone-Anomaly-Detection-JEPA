#!/usr/bin/env python3
"""
Visualize Adaptive Masking Behavior

Creates visualizations showing how entropy-guided masking works:
1. Sample chunks with different entropy levels
2. Mask ratio distribution
3. Entropy vs mask ratio correlation

Usage:
    python scripts/visualize_masking.py [--samples N]
"""

import argparse
import sys
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import DroneChunkDataset
from src.models.adaptive_masking import AdaptiveMaskingModule


def visualize_sample_masks(
    masking: AdaptiveMaskingModule,
    dataset: DroneChunkDataset,
    num_samples: int = 6,
    save_path: Path = None
):
    """Visualize masking on sample chunks."""
    
    fig, axes = plt.subplots(num_samples, 2, figsize=(14, 2.5 * num_samples))
    
    # Get diverse samples
    indices = np.linspace(0, len(dataset) - 1, num_samples).astype(int)
    
    for idx, sample_idx in enumerate(indices):
        sample = dataset[sample_idx]
        features = sample["features"].unsqueeze(0)
        
        # Generate mask
        masks, _, _, entropy = masking(features, return_entropy=True)
        
        mask = masks[0].numpy()
        feat = features[0].numpy()
        ent = entropy[0].item()
        mask_ratio = mask.mean()
        
        # Plot features
        ax1 = axes[idx, 0]
        time = np.arange(feat.shape[0])
        
        ax1.plot(time, feat[:, 2], label='Altitude', alpha=0.8, linewidth=2)
        ax1.plot(time, feat[:, 3], label='Speed', alpha=0.8, linewidth=2)
        ax1.plot(time, feat[:, 4], label='Heading', alpha=0.8, linewidth=2)
        
        # Highlight masked regions
        for i, m in enumerate(mask):
            if m:
                ax1.axvspan(i - 0.5, i + 0.5, alpha=0.3, color='red')
        
        ax1.set_ylabel('Value')
        if idx == 0:
            ax1.legend(loc='upper right', fontsize=8)
        if idx == num_samples - 1:
            ax1.set_xlabel('Waypoint')
        
        ax1.set_title(f'Sample {sample_idx} | Entropy: {ent:.3f} | Mask: {mask_ratio:.1%}')
        ax1.grid(True, alpha=0.3)
        
        # Plot mask
        ax2 = axes[idx, 1]
        mask_img = mask.reshape(1, -1)
        ax2.imshow(mask_img, aspect='auto', cmap='Reds', vmin=0, vmax=1)
        ax2.set_yticks([])
        ax2.set_xlabel('Waypoint' if idx == num_samples - 1 else '')
        ax2.set_title(f'Mask Pattern ({int(mask.sum())}/{len(mask)} masked)')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def visualize_entropy_distribution(
    masking: AdaptiveMaskingModule,
    dataset: DroneChunkDataset,
    num_samples: int = 1000,
    save_path: Path = None
):
    """Visualize entropy and mask ratio distributions."""
    
    entropies = []
    mask_ratios = []
    
    # Sample from dataset
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        features = sample["features"].unsqueeze(0)
        
        masks, _, _, entropy = masking(features, return_entropy=True)
        
        entropies.append(entropy[0].item())
        mask_ratios.append(masks[0].float().mean().item())
    
    entropies = np.array(entropies)
    mask_ratios = np.array(mask_ratios)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Entropy distribution
    ax1 = axes[0, 0]
    ax1.hist(entropies, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(entropies.mean(), color='red', linestyle='--', label=f'Mean: {entropies.mean():.3f}')
    ax1.set_xlabel('Entropy Score')
    ax1.set_ylabel('Count')
    ax1.set_title('Entropy Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Mask ratio distribution
    ax2 = axes[0, 1]
    ax2.hist(mask_ratios, bins=30, edgecolor='black', alpha=0.7, color='coral')
    ax2.axvline(mask_ratios.mean(), color='red', linestyle='--', label=f'Mean: {mask_ratios.mean():.2f}')
    ax2.set_xlabel('Mask Ratio')
    ax2.set_ylabel('Count')
    ax2.set_title('Mask Ratio Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Entropy vs Mask Ratio scatter
    ax3 = axes[1, 0]
    ax3.scatter(entropies, mask_ratios, alpha=0.5, s=10, c='purple')
    
    # Add trend line
    z = np.polyfit(entropies, mask_ratios, 1)
    p = np.poly1d(z)
    x_line = np.linspace(entropies.min(), entropies.max(), 100)
    ax3.plot(x_line, p(x_line), "r--", linewidth=2, label=f'Trend (slope={z[0]:.3f})')
    
    # Correlation
    corr = np.corrcoef(entropies, mask_ratios)[0, 1]
    ax3.set_xlabel('Entropy Score')
    ax3.set_ylabel('Mask Ratio')
    ax3.set_title(f'Entropy vs Mask Ratio (Correlation: {corr:.3f})')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Expected relationship annotation
    ax4 = axes[1, 1]
    ax4.text(0.5, 0.7, "Adaptive Masking Strategy", fontsize=14, fontweight='bold',
             ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.5, "Low Entropy (Stable Cruise)\n→ High Mask Ratio (50%)\n→ Harder prediction task",
             fontsize=11, ha='center', transform=ax4.transAxes)
    ax4.text(0.5, 0.25, "High Entropy (Turbulent)\n→ Low Mask Ratio (20%)\n→ More context for learning",
             fontsize=11, ha='center', transform=ax4.transAxes)
    ax4.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()
    
    return entropies, mask_ratios


def visualize_feature_contributions(
    masking: AdaptiveMaskingModule,
    dataset: DroneChunkDataset,
    num_samples: int = 500,
    save_path: Path = None
):
    """Visualize which features contribute most to entropy."""
    
    from src.data.dataset import get_feature_names
    feature_names = get_feature_names()
    
    # Compute per-feature variance for samples
    variances = []
    entropies = []
    
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    for idx in indices:
        sample = dataset[idx]
        features = sample["features"]  # (seq_len, num_features)
        
        # Per-feature variance
        var = features.var(dim=0).numpy()
        variances.append(var)
        
        # Entropy
        _, _, _, entropy = masking(features.unsqueeze(0), return_entropy=True)
        entropies.append(entropy[0].item())
    
    variances = np.array(variances)  # (num_samples, num_features)
    entropies = np.array(entropies)
    
    # Correlation of each feature's variance with overall entropy
    correlations = []
    for i in range(variances.shape[1]):
        corr = np.corrcoef(variances[:, i], entropies)[0, 1]
        correlations.append(corr)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(feature_names))
    colors = ['green' if c > 0.1 else 'gray' if abs(c) <= 0.1 else 'red' for c in correlations]
    
    bars = ax.bar(x, correlations, color=colors, edgecolor='black', alpha=0.7)
    
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(feature_names, rotation=45, ha='right')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Correlation with Entropy')
    ax.set_title('Feature Contribution to Entropy Score')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, corr in zip(bars, correlations):
        height = bar.get_height()
        ax.annotate(f'{corr:.2f}',
                   xy=(bar.get_x() + bar.get_width() / 2, height),
                   xytext=(0, 3 if height > 0 else -10),
                   textcoords="offset points",
                   ha='center', va='bottom' if height > 0 else 'top',
                   fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize adaptive masking")
    parser.add_argument("--data-dir", type=str, default="processed_data")
    parser.add_argument("--output-dir", type=str, default="outputs/masking_analysis")
    parser.add_argument("--samples", type=int, default=500, help="Number of samples to analyze")
    args = parser.parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║              ADAPTIVE MASKING ANALYSIS                    ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    print("Loading dataset...")
    dataset = DroneChunkDataset(args.data_dir, split="train_normal", return_labels=False)
    print(f"Dataset size: {len(dataset):,}")
    
    # Create adaptive masking module
    with open("configs/config.yaml") as f:
        config = yaml.safe_load(f)
    
    masking = AdaptiveMaskingModule(
        chunk_size=config["chunking"]["chunk_size"],
        min_mask_ratio=config["masking"]["min_mask_ratio"],
        max_mask_ratio=config["masking"]["max_mask_ratio"],
        adaptive=True,
        entropy_method="combined"
    )
    
    # Generate visualizations
    print("\n1. Generating sample mask visualizations...")
    visualize_sample_masks(
        masking, dataset, num_samples=6,
        save_path=output_dir / "sample_masks.png"
    )
    
    print("2. Generating entropy distribution...")
    entropies, mask_ratios = visualize_entropy_distribution(
        masking, dataset, num_samples=args.samples,
        save_path=output_dir / "entropy_distribution.png"
    )
    
    print("3. Generating feature contribution analysis...")
    visualize_feature_contributions(
        masking, dataset, num_samples=args.samples,
        save_path=output_dir / "feature_contributions.png"
    )
    
    # Print statistics
    print("\n" + "="*50)
    print("MASKING STATISTICS")
    print("="*50)
    print(f"Entropy - Mean: {entropies.mean():.3f}, Std: {entropies.std():.3f}")
    print(f"Mask Ratio - Mean: {mask_ratios.mean():.3f}, Std: {mask_ratios.std():.3f}")
    print(f"Range: [{mask_ratios.min():.2f}, {mask_ratios.max():.2f}]")
    print(f"Correlation (entropy vs mask): {np.corrcoef(entropies, mask_ratios)[0,1]:.3f}")
    
    print(f"\n✅ Visualizations saved to: {output_dir}")


if __name__ == "__main__":
    main()

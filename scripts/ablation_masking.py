#!/usr/bin/env python3
"""
JEPA-DRONE Masking Ablation Study

Compare fixed masking vs adaptive entropy-guided masking.

Experiments:
1. Fixed 30% masking (baseline)
2. Fixed 20% masking
3. Fixed 50% masking  
4. Adaptive 20-50% masking (our method)

Metrics:
- Training loss convergence
- Validation loss
- Masking statistics (entropy correlation)

Usage:
    python scripts/ablation_masking.py [--epochs EPOCHS] [--quick]
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import DroneChunkDataset
from src.models.jepa import create_jepa_model, JEPA
from src.models.trainer import CosineWarmupScheduler


def train_one_config(
    config_name: str,
    model: JEPA,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    device: torch.device,
    output_dir: Path
) -> dict:
    """
    Train model with a specific masking configuration.
    
    Returns:
        Training history dict
    """
    print(f"\n{'='*60}")
    print(f"Training: {config_name}")
    print(f"{'='*60}")
    
    model = model.to(device)
    
    # Optimizer
    optimizer = optim.AdamW(
        [
            {"params": model.context_encoder.parameters()},
            {"params": model.predictor.parameters()}
        ],
        lr=0.001,
        weight_decay=0.0001
    )
    
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=2,
        max_epochs=epochs
    )
    
    train_losses = []
    val_losses = []
    masking_stats_history = []
    
    start_time = time.time()
    
    for epoch in range(epochs):
        # Training
        model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            features = batch["features"].to(device)
            
            loss = model(features)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            model.update_target_encoder()
            
            epoch_loss += loss.item()
            num_batches += 1
        
        avg_train_loss = epoch_loss / num_batches
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                features = batch["features"].to(device)
                loss = model(features)
                val_loss += loss.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        val_losses.append(avg_val_loss)
        
        # Masking statistics
        masking_stats = model.get_masking_statistics()
        masking_stats_history.append(masking_stats.copy())
        
        scheduler.step()
        
        print(f"  Epoch {epoch+1}/{epochs} | "
              f"Train: {avg_train_loss:.4f} | "
              f"Val: {avg_val_loss:.4f} | "
              f"Mask Ratio: {masking_stats.get('avg_mask_ratio', 0):.3f}")
    
    total_time = time.time() - start_time
    
    # Save model
    model_path = output_dir / f"{config_name}_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config_name": config_name,
        "epochs": epochs
    }, model_path)
    
    results = {
        "config_name": config_name,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": min(val_losses),
        "final_val_loss": val_losses[-1],
        "masking_stats": masking_stats_history[-1] if masking_stats_history else {},
        "training_time_seconds": total_time
    }
    
    print(f"  Best Val Loss: {results['best_val_loss']:.4f}")
    print(f"  Time: {total_time:.1f}s")
    
    return results


def run_ablation(
    data_dir: str = "processed_data",
    output_dir: str = "outputs/ablation_masking",
    epochs: int = 10,
    batch_size: int = 128,
    device: str = "auto",
    quick: bool = False
):
    """
    Run complete masking ablation study.
    """
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                    JEPA-DRONE                             ║
    ║            MASKING ABLATION STUDY                         ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Setup
    output_dir = Path(output_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    if device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(device)
    
    print(f"📁 Output: {run_dir}")
    print(f"🖥️  Device: {device}")
    print(f"🎯 Epochs: {epochs}")
    
    # Load data
    print("\n🔄 Loading datasets...")
    train_dataset = DroneChunkDataset(data_dir, split="train_normal", return_labels=False)
    val_dataset = DroneChunkDataset(data_dir, split="train_normal", return_labels=False)
    
    # Use subset for quick testing
    if quick:
        from torch.utils.data import Subset
        train_indices = torch.randperm(len(train_dataset))[:2000]
        train_dataset = Subset(train_dataset, train_indices)
        val_indices = torch.randperm(len(val_dataset))[:500]
        val_dataset = Subset(val_dataset, val_indices)
    
    print(f"   Train: {len(train_dataset):,} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2
    )
    
    # Load base config
    with open("configs/config.yaml") as f:
        base_config = yaml.safe_load(f)
    
    # Ablation configurations
    ablation_configs = [
        ("fixed_20", {"adaptive": False, "fixed_mask_ratio": 0.20}),
        ("fixed_30", {"adaptive": False, "fixed_mask_ratio": 0.30}),
        ("fixed_50", {"adaptive": False, "fixed_mask_ratio": 0.50}),
        ("adaptive_20_50", {"adaptive": True, "min_mask_ratio": 0.20, "max_mask_ratio": 0.50}),
    ]
    
    results = {}
    
    for config_name, masking_overrides in ablation_configs:
        # Create config
        config = base_config.copy()
        config["masking"].update(masking_overrides)
        
        # Create fresh model
        model = create_jepa_model(config)
        
        # Train
        result = train_one_config(
            config_name=config_name,
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            device=device,
            output_dir=run_dir
        )
        
        results[config_name] = result
        
        # Clear GPU memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Save results
    results_path = run_dir / "ablation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print("\n" + "="*70)
    print("ABLATION STUDY RESULTS")
    print("="*70)
    print(f"{'Config':<20} {'Best Val Loss':<15} {'Final Val Loss':<15} {'Time (s)':<10}")
    print("-"*70)
    
    for config_name, result in sorted(results.items(), key=lambda x: x[1]["best_val_loss"]):
        print(f"{config_name:<20} {result['best_val_loss']:<15.4f} "
              f"{result['final_val_loss']:<15.4f} {result['training_time_seconds']:<10.1f}")
    
    print("-"*70)
    
    # Find best
    best_config = min(results.keys(), key=lambda k: results[k]["best_val_loss"])
    print(f"\n🏆 Best Configuration: {best_config}")
    print(f"   Best Val Loss: {results[best_config]['best_val_loss']:.4f}")
    
    # Create comparison plot
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Training loss
        ax1 = axes[0]
        for config_name, result in results.items():
            ax1.plot(result["train_losses"], label=config_name, linewidth=2)
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Loss")
        ax1.set_title("Training Loss Comparison")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Validation loss
        ax2 = axes[1]
        for config_name, result in results.items():
            ax2.plot(result["val_losses"], label=config_name, linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Validation Loss")
        ax2.set_title("Validation Loss Comparison")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(run_dir / "ablation_comparison.png", dpi=150, bbox_inches="tight")
        print(f"\n📊 Plot saved to: {run_dir / 'ablation_comparison.png'}")
        plt.close()
        
    except ImportError:
        print("\n⚠️ matplotlib not available, skipping plot")
    
    print(f"\n✅ Ablation study complete!")
    print(f"   Results saved to: {run_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Run masking ablation study")
    parser.add_argument("--data-dir", type=str, default="processed_data")
    parser.add_argument("--output-dir", type=str, default="outputs/ablation_masking")
    parser.add_argument("--epochs", type=int, default=10, help="Epochs per config")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--quick", action="store_true", help="Quick test with subset")
    
    args = parser.parse_args()
    
    run_ablation(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device,
        quick=args.quick
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
JEPA-DRONE Training Script

Train the JEPA model for self-supervised drone anomaly detection.

Usage:
    python scripts/train_jepa.py [--config CONFIG] [--epochs EPOCHS] [--device DEVICE]
    
Examples:
    python scripts/train_jepa.py
    python scripts/train_jepa.py --epochs 50 --device cuda
    python scripts/train_jepa.py --resume outputs/run_xxx/checkpoint.pt
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train JEPA model for drone anomaly detection"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="Path to configuration file"
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
        default="outputs",
        help="Output directory for checkpoints and logs"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to train on"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--adaptive-masking",
        action="store_true",
        help="Use adaptive entropy-guided masking instead of fixed"
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=None,
        help="Fixed mask ratio (default: 0.30)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                    JEPA-DRONE                             ║
    ║     Self-Supervised Drone Anomaly Detection               ║
    ║                                                           ║
    ║                  MODEL TRAINING                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    import yaml
    import torch
    from torch.utils.data import DataLoader
    
    from src.data.dataset import DroneChunkDataset
    from src.models.jepa import create_jepa_model
    from src.models.trainer import JEPATrainer
    
    # Check paths
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        print("   Run 'python scripts/preprocess_data.py' first")
        sys.exit(1)
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.lr:
        config["training"]["learning_rate"] = args.lr
    if args.adaptive_masking:
        config["masking"]["adaptive"] = True
    if args.mask_ratio:
        config["masking"]["fixed_mask_ratio"] = args.mask_ratio
    
    print(f"📁 Config: {config_path}")
    print(f"📂 Data: {data_dir}")
    print(f"📤 Output: {args.output_dir}")
    print(f"🎯 Epochs: {config['training']['epochs']}")
    print(f"📦 Batch size: {config['training']['batch_size']}")
    print(f"📈 Learning rate: {config['training']['learning_rate']}")
    print(f"🎭 Masking: {'Adaptive' if config['masking']['adaptive'] else 'Fixed'} ({config['masking']['fixed_mask_ratio']*100:.0f}%)")
    
    # Create datasets
    print("\n🔄 Loading datasets...")
    train_dataset = DroneChunkDataset(str(data_dir), split="train_normal", return_labels=False)
    
    # Use train set for validation during JEPA training (we don't use labels anyway)
    # Or use a subset of train for validation
    val_dataset = DroneChunkDataset(str(data_dir), split="train_normal", return_labels=False)
    
    print(f"   Train samples: {len(train_dataset):,}")
    
    # Create data loaders
    batch_size = config["training"]["batch_size"]
    num_workers = min(config.get("num_workers", 4), 4)  # Cap at 4 for safety
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # For validation, use a subset (every 5th batch)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Create model
    print("\n🔧 Creating JEPA model...")
    model = create_jepa_model(config)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    # Create trainer
    trainer = JEPATrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=args.device,
        output_dir=args.output_dir
    )
    
    # Resume from checkpoint if specified
    if args.resume:
        print(f"\n📥 Resuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Train
    print("\n🚀 Starting training...\n")
    try:
        history = trainer.train()
        
        print("\n" + "="*60)
        print("✅ Training completed successfully!")
        print("="*60)
        print(f"\n📊 Results:")
        print(f"   Best validation loss: {history['best_val_loss']:.4f}")
        print(f"   Training time: {history['total_time_seconds']/60:.1f} minutes")
        print(f"\n📁 Model saved to: {trainer.run_dir}")
        print(f"   - best_model.pt (best validation loss)")
        print(f"   - final_model.pt (final epoch)")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print("Saving checkpoint...")
        trainer.save_checkpoint("interrupted_checkpoint.pt")
        print(f"Checkpoint saved to: {trainer.run_dir}/interrupted_checkpoint.pt")
        sys.exit(0)
    
    except Exception as e:
        print(f"\n❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

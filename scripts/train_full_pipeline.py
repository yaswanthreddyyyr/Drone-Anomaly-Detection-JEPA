#!/usr/bin/env python3
"""
JEPA-DRONE Full Training Pipeline

Complete training pipeline that:
1. Trains JEPA model on normal flight data (self-supervised)
2. Extracts embeddings from trained model
3. Fits Isolation Forest on normal embeddings
4. Tunes threshold on validation data
5. Evaluates on test sets

Usage:
    python scripts/train_full_pipeline.py [--epochs EPOCHS] [--device DEVICE]
    
Examples:
    python scripts/train_full_pipeline.py --epochs 50
    python scripts/train_full_pipeline.py --epochs 100 --skip-jepa --jepa-checkpoint outputs/run_xxx/best.pt
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full JEPA-DRONE training pipeline (JEPA + Isolation Forest)"
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
    # JEPA training options
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Number of JEPA training epochs"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Batch size for training"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--adaptive-masking",
        action="store_true",
        default=True,
        help="Use adaptive entropy-guided masking"
    )
    parser.add_argument(
        "--no-adaptive-masking",
        action="store_false",
        dest="adaptive_masking",
        help="Use fixed masking instead"
    )
    # Isolation Forest options
    parser.add_argument(
        "--n-estimators",
        type=int,
        default=100,
        help="Number of trees in Isolation Forest"
    )
    parser.add_argument(
        "--contamination",
        type=float,
        default=0.05,
        help="Expected anomaly rate for Isolation Forest"
    )
    # Pipeline control
    parser.add_argument(
        "--skip-jepa",
        action="store_true",
        help="Skip JEPA training (use existing checkpoint)"
    )
    parser.add_argument(
        "--jepa-checkpoint",
        type=str,
        default=None,
        help="Path to existing JEPA checkpoint (required if --skip-jepa)"
    )
    parser.add_argument(
        "--skip-isolation-forest",
        action="store_true",
        help="Skip Isolation Forest fitting"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device to train on"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    return parser.parse_args()


def print_banner():
    print("""
    ╔═══════════════════════════════════════════════════════════════╗
    ║                        JEPA-DRONE                             ║
    ║        Self-Supervised Drone Anomaly Detection                ║
    ║                                                               ║
    ║              FULL TRAINING PIPELINE                           ║
    ║    [JEPA Training] → [Isolation Forest] → [Evaluation]        ║
    ╚═══════════════════════════════════════════════════════════════╝
    """)


def main():
    args = parse_args()
    print_banner()
    
    import yaml
    import numpy as np
    import torch
    from torch.utils.data import DataLoader
    
    from src.data.dataset import DroneChunkDataset
    from src.models.jepa import JEPA
    from src.models.trainer import JEPATrainer
    from src.models.isolation_forest import AnomalyDetector, EmbeddingExtractor, fit_anomaly_detector
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Check paths
    config_path = Path(args.config)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not config_path.exists():
        print(f"❌ Error: Config file not found: {config_path}")
        sys.exit(1)
    
    if not data_dir.exists():
        print(f"❌ Error: Data directory not found: {data_dir}")
        print("Run: python scripts/preprocess_data.py first")
        sys.exit(1)
    
    # Load config
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"full_run_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)
    
    print(f"📁 Output directory: {run_dir}")
    print(f"⚙️  Device: {args.device}")
    print(f"🎲 Seed: {args.seed}")
    print()
    
    # ==================== STEP 1: Load Data ====================
    print("="*60)
    print("STEP 1: Loading Data")
    print("="*60)
    
    # Training data (normal chunks only for self-supervised learning)
    train_dataset = DroneChunkDataset(data_dir, split="train", return_labels=True)
    # For JEPA training, we only use normal data
    train_normal = train_dataset.get_normal_chunks()
    
    # Validation and test data (includes anomalies for evaluation)
    val_dataset = DroneChunkDataset(data_dir, split="validation", return_labels=True)
    
    # Try to load test sets
    test_datasets = {}
    for test_split in ["test_balanced", "test_strong", "test_subtle", "test"]:
        try:
            test_ds = DroneChunkDataset(data_dir, split=test_split, return_labels=True)
            test_datasets[test_split] = test_ds
            print(f"  ✅ Loaded {test_split}: {len(test_ds):,} chunks")
        except:
            pass
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  - Training (normal only): {len(train_normal):,} chunks")
    print(f"  - Training (all): {len(train_dataset):,} chunks")
    print(f"  - Validation: {len(val_dataset):,} chunks")
    print(f"    - Normal: {(val_dataset.chunk_labels == 0).sum():,}")
    print(f"    - Anomaly: {(val_dataset.chunk_labels == 1).sum():,}")
    print(f"  - Features: {train_dataset.num_features}")
    print(f"  - Chunk size: {train_dataset.chunk_size}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_normal,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4
    )
    
    # ==================== STEP 2: JEPA Training ====================
    print("\n" + "="*60)
    print("STEP 2: JEPA Model Training")
    print("="*60)
    
    if args.skip_jepa:
        if not args.jepa_checkpoint:
            print("❌ Error: --jepa-checkpoint required when using --skip-jepa")
            sys.exit(1)
        
        print(f"⏭️  Skipping JEPA training, loading from: {args.jepa_checkpoint}")
        checkpoint = torch.load(args.jepa_checkpoint, map_location="cpu")
        
        # Create model
        model_config = config.get("model", {})
        masking_config = config.get("masking", {})
        model = JEPA(
            input_dim=train_dataset.num_features,
            embed_dim=model_config.get("embedding_dim", 256),
            encoder_hidden=model_config.get("encoder_hidden", [512, 256, 256]),
            predictor_hidden=model_config.get("predictor_hidden", [256, 256]),
            chunk_size=train_dataset.chunk_size,
            dropout=model_config.get("dropout", 0.1),
            adaptive_masking=args.adaptive_masking,
            min_mask_ratio=masking_config.get("min_mask_ratio", 0.20),
            max_mask_ratio=masking_config.get("max_mask_ratio", 0.50),
            fixed_mask_ratio=masking_config.get("fixed_mask_ratio", 0.30)
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        jepa_path = args.jepa_checkpoint
    else:
        print(f"\n🔧 Model Configuration:")
        print(f"  - Adaptive masking: {args.adaptive_masking}")
        print(f"  - Epochs: {args.epochs}")
        print(f"  - Batch size: {args.batch_size}")
        print(f"  - Learning rate: {args.lr}")
        
        # Create JEPA model
        model_config = config.get("model", {})
        masking_config = config.get("masking", {})
        model = JEPA(
            input_dim=train_dataset.num_features,
            embed_dim=model_config.get("embedding_dim", 256),
            encoder_hidden=model_config.get("encoder_hidden", [512, 256, 256]),
            predictor_hidden=model_config.get("predictor_hidden", [256, 256]),
            chunk_size=train_dataset.chunk_size,
            dropout=model_config.get("dropout", 0.1),
            adaptive_masking=args.adaptive_masking,
            min_mask_ratio=masking_config.get("min_mask_ratio", 0.20),
            max_mask_ratio=masking_config.get("max_mask_ratio", 0.50),
            fixed_mask_ratio=masking_config.get("fixed_mask_ratio", 0.30)
        )
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\n📐 Model Parameters:")
        print(f"  - Total: {total_params:,}")
        print(f"  - Trainable: {trainable_params:,}")
        
        # Training config
        training_config = {
            "training": {
                "epochs": args.epochs,
                "learning_rate": args.lr,
                "weight_decay": config.get("training", {}).get("weight_decay", 0.0001),
                "warmup_epochs": config.get("training", {}).get("warmup_epochs", 5)
            }
        }
        
        # Initialize trainer
        trainer = JEPATrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            device=args.device,
            output_dir=str(run_dir / "jepa")
        )
        
        # Train
        print("\n🚀 Starting JEPA training...\n")
        history = trainer.train()
        
        # Save final model
        jepa_path = run_dir / "jepa" / "best.pt"
        trainer.save_checkpoint("best.pt")
        
        print(f"\n✅ JEPA training complete!")
        print(f"   Best validation loss: {min(history['val_losses']):.4f}")
        print(f"   Model saved to: {jepa_path}")
    
    # ==================== STEP 3: Isolation Forest ====================
    print("\n" + "="*60)
    print("STEP 3: Isolation Forest Fitting")
    print("="*60)
    
    if args.skip_isolation_forest:
        print("⏭️  Skipping Isolation Forest fitting")
    else:
        print(f"\n🌲 Isolation Forest Configuration:")
        print(f"  - n_estimators: {args.n_estimators}")
        print(f"  - contamination: {args.contamination}")
        
        # Fit anomaly detector
        detector, if_results = fit_anomaly_detector(
            jepa_model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config={
                "isolation_forest": {
                    "n_estimators": args.n_estimators,
                    "contamination": args.contamination,
                    "random_state": args.seed
                }
            },
            device=args.device,
            output_dir=str(run_dir)
        )
        
        print(f"\n✅ Isolation Forest fitted!")
        if "validation" in if_results:
            val_metrics = if_results["validation"]
            print(f"   Validation AUC-ROC: {val_metrics['auc_roc']:.4f}")
            print(f"   Validation Recall: {val_metrics['recall']:.4f}")
            print(f"   Validation F1: {val_metrics['f1_score']:.4f}")
    
    # ==================== STEP 4: Test Evaluation ====================
    print("\n" + "="*60)
    print("STEP 4: Test Set Evaluation")
    print("="*60)
    
    if not args.skip_isolation_forest and test_datasets:
        # Extract embeddings and evaluate on test sets
        extractor = EmbeddingExtractor(model, device=args.device)
        
        all_test_results = {}
        
        for test_name, test_ds in test_datasets.items():
            print(f"\n📊 Evaluating on {test_name}...")
            
            test_loader = DataLoader(
                test_ds,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=4
            )
            
            # Extract embeddings
            embeddings, labels, types = extractor.extract_from_dataloader(
                test_loader,
                return_labels=True,
                verbose=False
            )
            
            # Evaluate
            results = detector.evaluate(embeddings, labels, anomaly_types=types)
            all_test_results[test_name] = results
            
            print(f"   - Samples: {results['n_samples']:,} ({results['n_anomalies']} anomalies)")
            print(f"   - AUC-ROC: {results['auc_roc']:.4f}")
            print(f"   - Recall: {results['recall']:.4f}")
            print(f"   - Precision: {results['precision']:.4f}")
            print(f"   - F1 Score: {results['f1_score']:.4f}")
            print(f"   - False Alarm Rate: {results['false_alarm_rate']:.4f}")
        
        # Save all results
        with open(run_dir / "test_results.json", "w") as f:
            json.dump(all_test_results, f, indent=2)
        
        # Per-anomaly-type analysis
        print("\n📋 Per-Anomaly-Type Results:")
        print("-"*60)
        if "per_type" in all_test_results.get(list(test_datasets.keys())[0], {}):
            per_type = all_test_results[list(test_datasets.keys())[0]]["per_type"]
            print(f"{'Type':<25} {'Samples':>10} {'AUC-ROC':>10} {'Recall':>10}")
            print("-"*60)
            for atype, metrics in per_type.items():
                print(f"{atype:<25} {metrics['n_samples']:>10,} {metrics['auc_roc']:>10.4f} {metrics['recall']:>10.4f}")
    
    # ==================== Summary ====================
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    print(f"\n📁 All outputs saved to: {run_dir}")
    print("\nFiles created:")
    for f in run_dir.rglob("*"):
        if f.is_file():
            rel_path = f.relative_to(run_dir)
            print(f"  - {rel_path}")
    
    print("\n🎯 Next steps:")
    print("  1. Review results in test_results.json")
    print("  2. Run evaluation: python scripts/evaluate.py --run-dir", run_dir)
    print("  3. Generate visualizations: python scripts/visualize_results.py --run-dir", run_dir)


if __name__ == "__main__":
    main()

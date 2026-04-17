#!/usr/bin/env python3
"""
JEPA-DRONE Data Inspection Script

Inspect processed data splits and visualize sample chunks.

Usage:
    python scripts/inspect_data.py [--split SPLIT_NAME]
"""

import argparse
import sys
from pathlib import Path

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.dataset import DroneChunkDataset, get_feature_names


def inspect_split(data_dir: str, split: str):
    """Print detailed statistics for a data split."""
    try:
        dataset = DroneChunkDataset(data_dir, split=split)
    except ValueError as e:
        print(f"❌ Error loading {split}: {e}")
        return
    
    stats = dataset.get_statistics()
    
    print(f"\n{'='*60}")
    print(f"📊 DATASET: {split}")
    print(f"{'='*60}")
    
    # Basic stats
    print(f"\n📈 Basic Statistics:")
    print(f"   Total chunks:     {stats['total_chunks']:,}")
    print(f"   Chunk size:       {stats['chunk_size']} waypoints")
    print(f"   Feature dims:     {stats['num_features']}")
    
    # Anomaly distribution
    normal_pct = 100 * stats['normal_chunks'] / stats['total_chunks']
    anom_pct = 100 * stats['anomalous_chunks'] / stats['total_chunks']
    
    print(f"\n🔹 Label Distribution:")
    print(f"   Normal chunks:    {stats['normal_chunks']:>8,} ({normal_pct:5.1f}%)")
    print(f"   Anomalous chunks: {stats['anomalous_chunks']:>8,} ({anom_pct:5.1f}%)")
    
    # Anomaly types
    print(f"\n🔸 Anomaly Type Distribution:")
    for atype, count in sorted(stats['anomaly_types'].items(), key=lambda x: -x[1]):
        pct = 100 * count / stats['total_chunks']
        print(f"   {atype:25s}: {count:>7,} ({pct:5.1f}%)")
    
    # Feature stats
    print(f"\n📐 Feature Statistics (normalized):")
    feature_names = get_feature_names()
    for i, name in enumerate(feature_names[:stats['num_features']]):
        mean = stats['feature_mean'][i]
        std = stats['feature_std'][i]
        print(f"   {name:20s}: μ={mean:>8.4f}, σ={std:>8.4f}")
    
    # Sample data
    sample = dataset[0]
    print(f"\n🔍 Sample Chunk:")
    print(f"   Shape: {sample['features'].shape}")
    print(f"   First 3 waypoints:")
    for i in range(min(3, len(sample['features']))):
        feat = sample['features'][i].numpy()
        print(f"     [{i}]: {feat[:5]}...")


def inspect_all(data_dir: str):
    """Inspect all available splits."""
    splits = ["train", "train_normal", "validation", "test_balanced", "test_strong", "test_subtle"]
    
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                    JEPA-DRONE                             ║
    ║              DATA INSPECTION REPORT                       ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    # Summary table
    print("📋 SUMMARY TABLE")
    print("-" * 80)
    print(f"{'Split':20s} {'Total':>10s} {'Normal':>10s} {'Anomalous':>10s} {'Features':>10s}")
    print("-" * 80)
    
    for split in splits:
        try:
            dataset = DroneChunkDataset(data_dir, split=split)
            stats = dataset.get_statistics()
            print(f"{split:20s} {stats['total_chunks']:>10,} {stats['normal_chunks']:>10,} {stats['anomalous_chunks']:>10,} {stats['num_features']:>10}")
        except:
            pass
    
    print("-" * 80)
    
    # Detailed inspection
    for split in splits:
        inspect_split(data_dir, split)


def main():
    parser = argparse.ArgumentParser(description="Inspect JEPA-DRONE processed data")
    parser.add_argument("--data-dir", type=str, default="processed_data", help="Processed data directory")
    parser.add_argument("--split", type=str, default=None, help="Specific split to inspect (default: all)")
    args = parser.parse_args()
    
    if args.split:
        inspect_split(args.data_dir, args.split)
    else:
        inspect_all(args.data_dir)


if __name__ == "__main__":
    main()

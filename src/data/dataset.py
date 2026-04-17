"""
PyTorch Dataset Classes for JEPA-DRONE

Provides efficient data loading for training and evaluation.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class DroneChunkDataset(Dataset):
    """
    PyTorch Dataset for drone flight chunks.
    
    Loads preprocessed chunks from disk for efficient training.
    
    Attributes:
        features: Tensor of shape (N, chunk_size, num_features)
        labels: Tensor of shape (N, chunk_size) - waypoint-level labels
        chunk_labels: Tensor of shape (N,) - chunk-level labels
        anomaly_types: Array of anomaly type strings
    """
    
    def __init__(
        self,
        data_dir: Union[str, Path],
        split: str = "train",
        return_labels: bool = True,
        transform: Optional[callable] = None
    ):
        """
        Initialize dataset.
        
        Args:
            data_dir: Path to processed data directory
            split: Which split to load ("train", "validation", "test_balanced", etc.)
            return_labels: Whether to return labels (False for self-supervised training)
            transform: Optional transform to apply to features
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.return_labels = return_labels
        self.transform = transform
        
        # Load data
        split_dir = self.data_dir / split
        
        if not split_dir.exists():
            raise ValueError(f"Split directory not found: {split_dir}")
        
        self.features = np.load(split_dir / "features.npy")
        self.labels = np.load(split_dir / "labels.npy")
        self.chunk_labels = np.load(split_dir / "chunk_labels.npy")
        self.anomaly_types = np.load(split_dir / "anomaly_types.npy", allow_pickle=True)
        
        # Convert to tensors
        self.features = torch.from_numpy(self.features).float()
        self.labels = torch.from_numpy(self.labels).long()
        self.chunk_labels = torch.from_numpy(self.chunk_labels).long()
        
        # Load normalization stats if available
        norm_file = self.data_dir / "normalization_stats.json"
        if norm_file.exists():
            with open(norm_file) as f:
                self.normalization_stats = json.load(f)
        else:
            self.normalization_stats = None
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a single chunk.
        
        Returns:
            Dictionary containing:
                - features: (chunk_size, num_features) tensor
                - labels: (chunk_size,) tensor of waypoint labels
                - chunk_label: scalar tensor (0 or 1)
                - anomaly_type: string
                - idx: sample index
        """
        features = self.features[idx]
        
        if self.transform is not None:
            features = self.transform(features)
        
        sample = {
            "features": features,
            "idx": idx
        }
        
        if self.return_labels:
            sample["labels"] = self.labels[idx]
            sample["chunk_label"] = self.chunk_labels[idx]
            sample["anomaly_type"] = self.anomaly_types[idx]
        
        return sample
    
    @property
    def num_features(self) -> int:
        return self.features.shape[-1]
    
    @property
    def chunk_size(self) -> int:
        return self.features.shape[1]
    
    def get_normal_chunks(self) -> "DroneChunkDataset":
        """Return a subset containing only normal (non-anomalous) chunks."""
        normal_mask = self.chunk_labels == 0
        
        subset = DroneChunkDataset.__new__(DroneChunkDataset)
        subset.data_dir = self.data_dir
        subset.split = self.split
        subset.return_labels = self.return_labels
        subset.transform = self.transform
        subset.normalization_stats = self.normalization_stats
        
        subset.features = self.features[normal_mask]
        subset.labels = self.labels[normal_mask]
        subset.chunk_labels = self.chunk_labels[normal_mask]
        subset.anomaly_types = self.anomaly_types[normal_mask.numpy()]
        
        return subset
    
    def get_anomalous_chunks(self) -> "DroneChunkDataset":
        """Return a subset containing only anomalous chunks."""
        anomaly_mask = self.chunk_labels == 1
        
        subset = DroneChunkDataset.__new__(DroneChunkDataset)
        subset.data_dir = self.data_dir
        subset.split = self.split
        subset.return_labels = self.return_labels
        subset.transform = self.transform
        subset.normalization_stats = self.normalization_stats
        
        subset.features = self.features[anomaly_mask]
        subset.labels = self.labels[anomaly_mask]
        subset.chunk_labels = self.chunk_labels[anomaly_mask]
        subset.anomaly_types = self.anomaly_types[anomaly_mask.numpy()]
        
        return subset
    
    def get_by_anomaly_type(self, anomaly_type: str) -> "DroneChunkDataset":
        """Return a subset containing only chunks of a specific anomaly type."""
        type_mask = np.array([t == anomaly_type for t in self.anomaly_types])
        
        subset = DroneChunkDataset.__new__(DroneChunkDataset)
        subset.data_dir = self.data_dir
        subset.split = self.split
        subset.return_labels = self.return_labels
        subset.transform = self.transform
        subset.normalization_stats = self.normalization_stats
        
        subset.features = self.features[type_mask]
        subset.labels = self.labels[type_mask]
        subset.chunk_labels = self.chunk_labels[type_mask]
        subset.anomaly_types = self.anomaly_types[type_mask]
        
        return subset
    
    def get_statistics(self) -> Dict:
        """Get dataset statistics."""
        from collections import Counter
        
        return {
            "total_chunks": len(self),
            "chunk_size": self.chunk_size,
            "num_features": self.num_features,
            "normal_chunks": int((self.chunk_labels == 0).sum()),
            "anomalous_chunks": int((self.chunk_labels == 1).sum()),
            "anomaly_types": dict(Counter(self.anomaly_types)),
            "feature_mean": self.features.mean(dim=(0, 1)).tolist(),
            "feature_std": self.features.std(dim=(0, 1)).tolist()
        }


class DroneDataset(Dataset):
    """
    Dataset for full flight sequences (not chunked).
    
    Useful for sequence-to-sequence models or evaluation.
    """
    
    def __init__(
        self,
        cases: List,
        max_length: Optional[int] = None,
        normalize: bool = True,
        normalization_stats: Optional[Dict] = None
    ):
        """
        Initialize from list of FlightCase objects.
        
        Args:
            cases: List of FlightCase objects from preprocessing
            max_length: Maximum sequence length (truncate longer sequences)
            normalize: Whether to normalize features
            normalization_stats: Pre-computed normalization statistics
        """
        self.cases = cases
        self.max_length = max_length
        self.normalize = normalize
        self.normalization_stats = normalization_stats
    
    def __len__(self) -> int:
        return len(self.cases)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        case = self.cases[idx]
        
        # Extract features (implement feature extraction here)
        features = self._extract_features(case.data)
        labels = case.labels
        
        # Truncate if necessary
        if self.max_length and len(features) > self.max_length:
            features = features[:self.max_length]
            labels = labels[:self.max_length]
        
        # Normalize
        if self.normalize and self.normalization_stats:
            features = self._normalize(features)
        
        return {
            "features": torch.from_numpy(features).float(),
            "labels": torch.from_numpy(labels).long(),
            "anomaly_type": case.anomaly_type,
            "case_id": case.case_id,
            "length": len(features)
        }
    
    def _extract_features(self, df) -> np.ndarray:
        """Extract features from DataFrame."""
        base_features = ["latitude", "longitude", "altitude", "speed", "heading"]
        return df[base_features].values.astype(np.float32)
    
    def _normalize(self, features: np.ndarray) -> np.ndarray:
        """Apply normalization."""
        stats = self.normalization_stats
        center = np.array(stats["center"])
        scale = np.array(stats["scale"])
        return (features - center) / scale


def create_data_loaders(
    data_dir: Union[str, Path],
    batch_size: int = 256,
    num_workers: int = 4,
    pin_memory: bool = True
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for all splits.
    
    Args:
        data_dir: Path to processed data directory
        batch_size: Batch size for training
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory (faster GPU transfer)
        
    Returns:
        Dictionary of DataLoaders for each split
    """
    data_dir = Path(data_dir)
    loaders = {}
    
    # Training loader (shuffle, drop_last for consistent batch sizes)
    train_dataset = DroneChunkDataset(data_dir, split="train", return_labels=False)
    loaders["train"] = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True
    )
    
    # Validation loader
    if (data_dir / "validation").exists():
        val_dataset = DroneChunkDataset(data_dir, split="validation", return_labels=True)
        loaders["validation"] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    # Test loaders
    for test_split in ["test_balanced", "test_strong", "test_subtle"]:
        if (data_dir / test_split).exists():
            test_dataset = DroneChunkDataset(data_dir, split=test_split, return_labels=True)
            loaders[test_split] = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=pin_memory
            )
    
    return loaders


def get_feature_names(use_derived: bool = True) -> List[str]:
    """Get ordered list of feature names."""
    names = ["latitude", "longitude", "altitude", "speed", "heading"]
    
    if use_derived:
        names.extend([
            "delta_lat", "delta_lon", "delta_alt",
            "acceleration", "angular_velocity", "distance"
        ])
    
    return names


# Convenience functions for quick data inspection
def inspect_dataset(data_dir: Union[str, Path], split: str = "train") -> None:
    """Print dataset statistics and sample data."""
    dataset = DroneChunkDataset(data_dir, split=split)
    stats = dataset.get_statistics()
    
    print(f"\n{'='*60}")
    print(f"Dataset: {split}")
    print(f"{'='*60}")
    print(f"Total chunks: {stats['total_chunks']:,}")
    print(f"Chunk size: {stats['chunk_size']} waypoints")
    print(f"Num features: {stats['num_features']}")
    print(f"Normal chunks: {stats['normal_chunks']:,} ({100*stats['normal_chunks']/stats['total_chunks']:.1f}%)")
    print(f"Anomalous chunks: {stats['anomalous_chunks']:,} ({100*stats['anomalous_chunks']/stats['total_chunks']:.1f}%)")
    
    print("\nAnomaly type distribution:")
    for atype, count in sorted(stats['anomaly_types'].items(), key=lambda x: -x[1]):
        print(f"  {atype:25s}: {count:,}")
    
    print("\nFeature statistics:")
    feature_names = get_feature_names()
    for i, name in enumerate(feature_names[:stats['num_features']]):
        print(f"  {name:20s}: mean={stats['feature_mean'][i]:.4f}, std={stats['feature_std'][i]:.4f}")
    
    # Sample
    sample = dataset[0]
    print(f"\nSample shape: {sample['features'].shape}")


if __name__ == "__main__":
    # Quick test
    import sys
    
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "processed_data"
    
    for split in ["train", "validation", "test_balanced"]:
        try:
            inspect_dataset(data_dir, split)
        except Exception as e:
            print(f"Could not load {split}: {e}")

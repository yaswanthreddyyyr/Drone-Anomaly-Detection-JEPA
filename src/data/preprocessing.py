"""
JEPA-DRONE Data Preprocessing Module

Handles:
- Loading raw flight logs from CSV
- Feature extraction and derivation
- Chunking into fixed-size windows
- Train/val/test splitting
- Normalization
- Saving processed data
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from tqdm import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class FlightCase:
    """Represents a single flight case with metadata"""
    case_id: int
    case_name: str
    profile: str  # balanced, strong, subtle
    replicate: str  # rep_00, rep_01, etc.
    anomaly_type: str  # injection, normal, deletion_gap, etc.
    data: pd.DataFrame = field(repr=False)
    labels: np.ndarray = field(repr=False)
    metadata: Dict = field(default_factory=dict)
    
    @property
    def num_waypoints(self) -> int:
        return len(self.data)
    
    @property
    def anomaly_rate(self) -> float:
        return self.labels.sum() / len(self.labels) if len(self.labels) > 0 else 0.0
    
    @property
    def is_normal(self) -> bool:
        return self.anomaly_type == "normal"


@dataclass
class Chunk:
    """Represents a single chunk of waypoints"""
    case_id: int
    chunk_idx: int
    features: np.ndarray  # Shape: (chunk_size, num_features)
    labels: np.ndarray    # Shape: (chunk_size,) binary labels
    anomaly_type: str
    profile: str
    replicate: str  # rep_00, rep_01, etc.
    
    @property
    def is_anomalous(self) -> bool:
        """A chunk is anomalous if ANY waypoint is anomalous"""
        return self.labels.sum() > 0
    
    @property
    def anomaly_ratio(self) -> float:
        """Fraction of anomalous waypoints in chunk"""
        return self.labels.sum() / len(self.labels)


class DataPreprocessor:
    """
    Main data preprocessing class for JEPA-DRONE project.
    
    Usage:
        preprocessor = DataPreprocessor(config)
        preprocessor.load_all_cases()
        preprocessor.create_chunks()
        preprocessor.compute_normalization_stats()
        preprocessor.create_splits()
        preprocessor.save_processed_data()
    """
    
    # Feature columns in order
    BASE_FEATURES = ["latitude", "longitude", "altitude", "speed", "heading"]
    
    def __init__(self, config: Dict):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary (from YAML)
        """
        self.config = config
        self.raw_dir = Path(config["data"]["raw_dir"])
        self.processed_dir = Path(config["data"]["processed_dir"])
        
        # Chunking parameters
        self.chunk_size = config["chunking"]["chunk_size"]
        self.stride = config["chunking"]["stride"]
        self.min_chunk_size = config["chunking"]["min_chunk_size"]
        
        # Feature settings
        self.use_derived = config["features"]["use_derived"]
        
        # Storage
        self.cases: List[FlightCase] = []
        self.chunks: List[Chunk] = []
        self.normalization_stats: Dict = {}
        
        # Create output directory
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_all_cases(self, 
                       profiles: Optional[List[str]] = None,
                       replicates: Optional[List[str]] = None) -> None:
        """
        Load all flight cases from raw data directory.
        
        Args:
            profiles: List of profiles to load (e.g., ["balanced", "strong"])
            replicates: List of replicates to load (e.g., ["rep_00", "rep_01"])
        """
        profiles = profiles or ["balanced", "strong", "subtle"]
        replicates = replicates or ["rep_00", "rep_01", "rep_02", "rep_03"]
        
        logger.info(f"Loading cases from: {self.raw_dir}")
        logger.info(f"Profiles: {profiles}, Replicates: {replicates}")
        
        self.cases = []
        
        for profile in profiles:
            for replicate in replicates:
                cases_dir = self.raw_dir / profile / replicate / "cases"
                
                if not cases_dir.exists():
                    logger.warning(f"Directory not found: {cases_dir}")
                    continue
                
                case_dirs = sorted(cases_dir.iterdir())
                
                for case_dir in tqdm(case_dirs, desc=f"{profile}/{replicate}"):
                    if not case_dir.is_dir():
                        continue
                    
                    case = self._load_single_case(case_dir, profile, replicate)
                    if case is not None:
                        self.cases.append(case)
        
        logger.info(f"Loaded {len(self.cases)} flight cases")
        self._print_case_statistics()
    
    def _load_single_case(self, 
                          case_dir: Path, 
                          profile: str, 
                          replicate: str) -> Optional[FlightCase]:
        """Load a single flight case from directory."""
        try:
            # Load metadata
            meta_file = case_dir / "case_meta.json"
            with open(meta_file) as f:
                metadata = json.load(f)
            
            # Load flight data
            csv_file = case_dir / "decoded_flightlog.csv"
            df = pd.read_csv(csv_file)
            
            # Load labels
            labels_file = case_dir / "labels.csv"
            labels_df = pd.read_csv(labels_file)
            labels = labels_df["label"].values.astype(np.int8)
            
            # Extract anomaly type from case name
            case_name = case_dir.name
            anomaly_type = metadata["tamper"]["type"]
            
            return FlightCase(
                case_id=metadata["case_id"],
                case_name=case_name,
                profile=profile,
                replicate=replicate,
                anomaly_type=anomaly_type,
                data=df,
                labels=labels,
                metadata=metadata
            )
            
        except Exception as e:
            logger.error(f"Error loading case {case_dir}: {e}")
            return None
    
    def _print_case_statistics(self) -> None:
        """Print statistics about loaded cases."""
        from collections import Counter
        
        profiles = Counter(c.profile for c in self.cases)
        anomaly_types = Counter(c.anomaly_type for c in self.cases)
        
        logger.info("\n=== Case Statistics ===")
        logger.info(f"By Profile: {dict(profiles)}")
        logger.info(f"By Anomaly Type: {dict(anomaly_types)}")
        
        total_waypoints = sum(c.num_waypoints for c in self.cases)
        logger.info(f"Total Waypoints: {total_waypoints:,}")
    
    def extract_features(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extract and compute all features from raw flight data.
        
        Args:
            df: Raw flight log DataFrame
            
        Returns:
            Feature matrix of shape (N, num_features)
        """
        # Start with base features
        features = df[self.BASE_FEATURES].values.copy()
        
        if self.use_derived:
            derived = self._compute_derived_features(df)
            features = np.hstack([features, derived])
        
        return features.astype(np.float32)
    
    def _compute_derived_features(self, df: pd.DataFrame) -> np.ndarray:
        """Compute derived features from base telemetry.
        
        Features computed:
        - 1st order: delta_lat, delta_lon, delta_alt, acceleration, angular_velocity, distance
        - 2nd order: jerk, angular_acceleration, altitude_jerk
        - Windowed: speed_std_5, heading_std_5, altitude_std_5
        
        Total: 12 derived features (was 6)
        """
        n = len(df)
        
        # Check config for enhanced features
        use_second_order = self.config.get("features", {}).get("use_second_order", False)
        use_windowed_stats = self.config.get("features", {}).get("use_windowed_stats", False)
        window_size = self.config.get("features", {}).get("window_size", 5)
        
        # Base: 6 first-order derivatives
        num_features = 6
        if use_second_order:
            num_features += 3  # jerk, angular_acceleration, altitude_jerk
        if use_windowed_stats:
            num_features += 3  # speed_std, heading_std, altitude_std
        
        derived = np.zeros((n, num_features), dtype=np.float32)
        
        # ========== 1ST ORDER DERIVATIVES ==========
        # Delta latitude (rate of change)
        derived[1:, 0] = np.diff(df["latitude"].values)
        
        # Delta longitude (rate of change)
        derived[1:, 1] = np.diff(df["longitude"].values)
        
        # Delta altitude (climb rate)
        delta_alt = np.zeros(n)
        delta_alt[1:] = np.diff(df["altitude"].values)
        derived[:, 2] = delta_alt
        
        # Acceleration (delta speed)
        acceleration = np.zeros(n)
        acceleration[1:] = np.diff(df["speed"].values)
        derived[:, 3] = acceleration
        
        # Angular velocity (delta heading) - handle wrap-around
        headings = df["heading"].values
        delta_heading = np.zeros(n)
        raw_delta = np.diff(headings)
        # Normalize to [-180, 180]
        delta_heading[1:] = np.mod(raw_delta + 180, 360) - 180
        derived[:, 4] = delta_heading
        
        # Distance traveled (haversine approximation for small distances)
        lat = np.radians(df["latitude"].values)
        lon = np.radians(df["longitude"].values)
        dlat = np.diff(lat)
        dlon = np.diff(lon)
        dist = np.sqrt(dlat**2 + dlon**2) * 111000  # meters
        derived[1:, 5] = dist
        
        feat_idx = 6
        
        # ========== 2ND ORDER DERIVATIVES ==========
        if use_second_order:
            # Jerk (rate of change of acceleration)
            jerk = np.zeros(n)
            jerk[2:] = np.diff(acceleration[1:])
            derived[:, feat_idx] = jerk
            feat_idx += 1
            
            # Angular acceleration (rate of change of angular velocity)
            angular_accel = np.zeros(n)
            angular_accel[2:] = np.diff(delta_heading[1:])
            # Normalize to [-180, 180]
            angular_accel = np.mod(angular_accel + 180, 360) - 180
            derived[:, feat_idx] = angular_accel
            feat_idx += 1
            
            # Altitude jerk (rate of change of climb rate)
            alt_jerk = np.zeros(n)
            alt_jerk[2:] = np.diff(delta_alt[1:])
            derived[:, feat_idx] = alt_jerk
            feat_idx += 1
        
        # ========== WINDOWED STATISTICS ==========
        if use_windowed_stats:
            # Speed variance over window
            speed = df["speed"].values
            speed_std = self._rolling_std(speed, window_size)
            derived[:, feat_idx] = speed_std
            feat_idx += 1
            
            # Heading variance over window (handle circular)
            heading_std = self._rolling_circular_std(headings, window_size)
            derived[:, feat_idx] = heading_std
            feat_idx += 1
            
            # Altitude variance over window
            altitude = df["altitude"].values
            alt_std = self._rolling_std(altitude, window_size)
            derived[:, feat_idx] = alt_std
            feat_idx += 1
        
        return derived
    
    def _rolling_std(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation."""
        n = len(arr)
        result = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            start = max(0, i - window + 1)
            result[i] = np.std(arr[start:i+1])
        
        return result
    
    def _rolling_circular_std(self, arr: np.ndarray, window: int) -> np.ndarray:
        """Compute rolling standard deviation for circular data (heading)."""
        n = len(arr)
        result = np.zeros(n, dtype=np.float32)
        
        for i in range(n):
            start = max(0, i - window + 1)
            window_data = arr[start:i+1]
            
            # Convert to radians for circular statistics
            rad = np.radians(window_data)
            
            # Circular mean
            sin_mean = np.mean(np.sin(rad))
            cos_mean = np.mean(np.cos(rad))
            
            # Circular variance (1 - R, where R is resultant length)
            R = np.sqrt(sin_mean**2 + cos_mean**2)
            circular_var = 1 - R
            
            # Convert to approximate standard deviation in degrees
            result[i] = np.degrees(np.sqrt(-2 * np.log(max(R, 1e-10))))
        
        return result
    
    def create_chunks(self) -> None:
        """
        Create fixed-size chunks from all loaded cases.
        
        Uses sliding window with configurable stride.
        """
        logger.info(f"Creating chunks (size={self.chunk_size}, stride={self.stride})")
        
        self.chunks = []
        
        for case in tqdm(self.cases, desc="Chunking"):
            # Extract features
            features = self.extract_features(case.data)
            labels = case.labels
            
            n = len(features)
            chunk_idx = 0
            
            # Sliding window
            for start in range(0, n - self.min_chunk_size + 1, self.stride):
                end = min(start + self.chunk_size, n)
                
                # Skip if chunk is too small
                if end - start < self.min_chunk_size:
                    continue
                
                chunk_features = features[start:end]
                chunk_labels = labels[start:end]
                
                # Pad if necessary
                if len(chunk_features) < self.chunk_size:
                    pad_size = self.chunk_size - len(chunk_features)
                    chunk_features = np.pad(
                        chunk_features, 
                        ((0, pad_size), (0, 0)), 
                        mode='edge'
                    )
                    chunk_labels = np.pad(
                        chunk_labels,
                        (0, pad_size),
                        mode='edge'
                    )
                
                chunk = Chunk(
                    case_id=case.case_id,
                    chunk_idx=chunk_idx,
                    features=chunk_features,
                    labels=chunk_labels,
                    anomaly_type=case.anomaly_type,
                    profile=case.profile,
                    replicate=case.replicate
                )
                
                self.chunks.append(chunk)
                chunk_idx += 1
        
        logger.info(f"Created {len(self.chunks):,} chunks")
        self._print_chunk_statistics()
    
    def _print_chunk_statistics(self) -> None:
        """Print statistics about created chunks."""
        from collections import Counter
        
        anomalous = sum(1 for c in self.chunks if c.is_anomalous)
        normal = len(self.chunks) - anomalous
        
        logger.info("\n=== Chunk Statistics ===")
        logger.info(f"Total Chunks: {len(self.chunks):,}")
        logger.info(f"Normal Chunks: {normal:,} ({100*normal/len(self.chunks):.1f}%)")
        logger.info(f"Anomalous Chunks: {anomalous:,} ({100*anomalous/len(self.chunks):.1f}%)")
        
        by_type = Counter(c.anomaly_type for c in self.chunks)
        logger.info(f"By Anomaly Type: {dict(by_type)}")
    
    def compute_normalization_stats(self, 
                                    use_chunks: Optional[List[Chunk]] = None) -> Dict:
        """
        Compute normalization statistics from training data.
        
        Args:
            use_chunks: Chunks to compute stats from (default: all chunks)
            
        Returns:
            Dictionary with normalization parameters
        """
        chunks = use_chunks or self.chunks
        
        # Stack all features
        all_features = np.vstack([c.features for c in chunks])
        
        method = self.config["normalization"]["method"]
        
        if method == "robust":
            # Robust scaling using median and IQR
            median = np.median(all_features, axis=0)
            q1 = np.percentile(all_features, 25, axis=0)
            q3 = np.percentile(all_features, 75, axis=0)
            iqr = q3 - q1
            iqr[iqr == 0] = 1.0  # Avoid division by zero
            
            self.normalization_stats = {
                "method": "robust",
                "center": median,
                "scale": iqr
            }
            
        elif method == "standard":
            mean = np.mean(all_features, axis=0)
            std = np.std(all_features, axis=0)
            std[std == 0] = 1.0
            
            self.normalization_stats = {
                "method": "standard",
                "center": mean,
                "scale": std
            }
            
        elif method == "minmax":
            min_val = np.min(all_features, axis=0)
            max_val = np.max(all_features, axis=0)
            range_val = max_val - min_val
            range_val[range_val == 0] = 1.0
            
            self.normalization_stats = {
                "method": "minmax",
                "center": min_val,
                "scale": range_val
            }
        
        # Clip outliers if configured
        if self.config["normalization"]["clip_outliers"]:
            p_low, p_high = self.config["normalization"]["clip_percentile"]
            self.normalization_stats["clip_low"] = np.percentile(all_features, p_low, axis=0)
            self.normalization_stats["clip_high"] = np.percentile(all_features, p_high, axis=0)
        
        logger.info(f"Computed normalization stats using {method} method")
        return self.normalization_stats
    
    def normalize_features(self, features: np.ndarray) -> np.ndarray:
        """
        Apply normalization to features.
        
        Args:
            features: Raw feature array
            
        Returns:
            Normalized features
        """
        if not self.normalization_stats:
            raise ValueError("Normalization stats not computed. Call compute_normalization_stats() first.")
        
        stats = self.normalization_stats
        
        # Clip outliers first if configured
        if "clip_low" in stats:
            features = np.clip(features, stats["clip_low"], stats["clip_high"])
        
        # Apply normalization
        normalized = (features - stats["center"]) / stats["scale"]
        
        return normalized.astype(np.float32)
    
    def create_splits(self) -> Dict[str, List[Chunk]]:
        """
        Create train/validation/test splits according to config.
        
        Returns:
            Dictionary with split names as keys and chunk lists as values
        """
        splits_config = self.config["splits"]
        
        # Training split config
        train_profiles = splits_config["train"]["profiles"]
        train_reps = splits_config["train"]["replicates"]
        
        # Validation split config
        val_profiles = splits_config["validation"]["profiles"]
        val_reps = splits_config["validation"]["replicates"]
        
        # Test split configs
        test_balanced_reps = splits_config["test"]["balanced"]["replicates"]
        test_strong_reps = splits_config["test"]["strong"]["replicates"]
        test_subtle_reps = splits_config["test"]["subtle"]["replicates"]
        
        logger.info("Creating splits based on profile and replicate...")
        
        # Group chunks by split criteria
        splits = {
            "train": [],
            "validation": [],
            "test_balanced": [],
            "test_strong": [],
            "test_subtle": []
        }
        
        for chunk in self.chunks:
            # Training: balanced, rep_00 and rep_01
            if chunk.profile in train_profiles and chunk.replicate in train_reps:
                splits["train"].append(chunk)
            
            # Validation: balanced, rep_02
            elif chunk.profile in val_profiles and chunk.replicate in val_reps:
                splits["validation"].append(chunk)
            
            # Test balanced: balanced, rep_03
            elif chunk.profile == "balanced" and chunk.replicate in test_balanced_reps:
                splits["test_balanced"].append(chunk)
            
            # Test strong: all strong replicates
            elif chunk.profile == "strong" and chunk.replicate in test_strong_reps:
                splits["test_strong"].append(chunk)
            
            # Test subtle: all subtle replicates
            elif chunk.profile == "subtle" and chunk.replicate in test_subtle_reps:
                splits["test_subtle"].append(chunk)
        
        # Log split sizes
        for split_name, split_chunks in splits.items():
            anomalous = sum(1 for c in split_chunks if c.is_anomalous)
            logger.info(f"{split_name}: {len(split_chunks):,} chunks ({anomalous:,} anomalous)")
        
        # Create train_normal split (only non-anomalous chunks for JEPA training)
        splits["train_normal"] = [c for c in splits["train"] if not c.is_anomalous]
        logger.info(f"train_normal: {len(splits['train_normal']):,} chunks (0 anomalous)")
        
        return splits
    
    def save_processed_data(self, 
                            splits: Dict[str, List[Chunk]],
                            normalize: bool = True) -> None:
        """
        Save processed data to disk.
        
        Args:
            splits: Dictionary of split name -> chunks
            normalize: Whether to apply normalization
        """
        logger.info(f"Saving processed data to: {self.processed_dir}")
        
        for split_name, chunks in splits.items():
            if not chunks:
                logger.warning(f"No chunks for split: {split_name}")
                continue
            
            # Stack features and labels
            features = np.stack([c.features for c in chunks])
            labels = np.stack([c.labels for c in chunks])
            
            # Chunk-level labels (is the chunk anomalous?)
            chunk_labels = np.array([1 if c.is_anomalous else 0 for c in chunks])
            
            # Anomaly types
            anomaly_types = np.array([c.anomaly_type for c in chunks])
            
            # Apply normalization
            if normalize and self.normalization_stats:
                # Reshape for normalization
                orig_shape = features.shape
                features_flat = features.reshape(-1, features.shape[-1])
                features_normalized = self.normalize_features(features_flat)
                features = features_normalized.reshape(orig_shape)
            
            # Save as numpy arrays
            split_dir = self.processed_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            np.save(split_dir / "features.npy", features)
            np.save(split_dir / "labels.npy", labels)
            np.save(split_dir / "chunk_labels.npy", chunk_labels)
            np.save(split_dir / "anomaly_types.npy", anomaly_types)
            
            logger.info(f"Saved {split_name}: features {features.shape}, labels {labels.shape}")
        
        # Save normalization stats
        if self.normalization_stats:
            stats_to_save = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.normalization_stats.items()
            }
            with open(self.processed_dir / "normalization_stats.json", "w") as f:
                json.dump(stats_to_save, f, indent=2)
        
        # Save config for reproducibility
        import yaml
        with open(self.processed_dir / "config.yaml", "w") as f:
            yaml.dump(self.config, f, default_flow_style=False)
        
        logger.info("Data processing complete!")
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names in order."""
        names = list(self.BASE_FEATURES)
        
        if self.use_derived:
            derived_names = [
                "delta_lat", "delta_lon", "delta_alt",
                "acceleration", "angular_velocity", "distance"
            ]
            names.extend(derived_names)
        
        return names


def load_config(config_path: str = "configs/config.yaml") -> Dict:
    """Load configuration from YAML file."""
    import yaml
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    return config


def run_preprocessing_pipeline(config_path: str = "configs/config.yaml") -> None:
    """
    Run the complete data preprocessing pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    # Load config
    config = load_config(config_path)
    
    # Initialize preprocessor
    preprocessor = DataPreprocessor(config)
    
    # Step 1: Load all cases
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Loading flight cases")
    logger.info("="*60)
    preprocessor.load_all_cases()
    
    # Step 2: Create chunks
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Creating chunks")
    logger.info("="*60)
    preprocessor.create_chunks()
    
    # Step 3: Create splits
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Creating train/val/test splits")
    logger.info("="*60)
    splits = preprocessor.create_splits()
    
    # Step 4: Compute normalization stats (from training data only)
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Computing normalization statistics")
    logger.info("="*60)
    preprocessor.compute_normalization_stats(use_chunks=splits["train"])
    
    # Step 5: Save processed data
    logger.info("\n" + "="*60)
    logger.info("STEP 5: Saving processed data")
    logger.info("="*60)
    preprocessor.save_processed_data(splits, normalize=True)
    
    logger.info("\n" + "="*60)
    logger.info("PREPROCESSING COMPLETE!")
    logger.info("="*60)
    
    # Print summary
    print("\n📊 Dataset Summary:")
    print(f"   Feature dimensions: {splits['train'][0].features.shape[-1]}")
    print(f"   Chunk size: {config['chunking']['chunk_size']} waypoints")
    print(f"   Training chunks: {len(splits['train']):,}")
    print(f"   Validation chunks: {len(splits['validation']):,}")
    print(f"   Test (balanced): {len(splits['test_balanced']):,}")
    print(f"   Test (strong): {len(splits['test_strong']):,}")
    print(f"   Test (subtle): {len(splits['test_subtle']):,}")
    print(f"\n   Output directory: {config['data']['processed_dir']}")


if __name__ == "__main__":
    run_preprocessing_pipeline()

"""
JEPA-DRONE: Inference Pipeline

Complete inference pipeline for anomaly detection:
1. Load trained JEPA model
2. Load fitted Isolation Forest
3. Process new telemetry data
4. Output anomaly predictions and scores
"""

import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .jepa import JEPA
from .isolation_forest import AnomalyDetector, EmbeddingExtractor


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnomalyPrediction:
    """Single chunk anomaly prediction."""
    
    chunk_id: int
    is_anomaly: bool
    anomaly_score: float  # Lower = more anomalous
    confidence: float  # 0-1 scale
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class FlightPrediction:
    """Aggregated flight-level prediction."""
    
    flight_id: str
    is_anomaly: bool
    anomaly_chunks: int
    total_chunks: int
    anomaly_ratio: float
    mean_score: float
    min_score: float  # Most anomalous chunk
    chunk_predictions: List[AnomalyPrediction]
    
    def to_dict(self) -> Dict:
        result = asdict(self)
        result["chunk_predictions"] = [p.to_dict() for p in self.chunk_predictions]
        return result


class JEPADroneInference:
    """
    Complete inference pipeline for JEPA-DRONE anomaly detection.
    
    Usage:
        # Initialize
        pipeline = JEPADroneInference.from_checkpoint(
            jepa_path="checkpoints/jepa_best.pt",
            detector_path="outputs/anomaly_detector.pkl"
        )
        
        # Single chunk inference
        features = torch.randn(1, 20, 11)  # (batch, seq_len, features)
        result = pipeline.predict(features)
        
        # Batch inference
        results = pipeline.predict_batch(dataloader)
    """
    
    def __init__(
        self,
        jepa_model: JEPA,
        anomaly_detector: AnomalyDetector,
        device: str = "auto",
        pooling: str = "mean"
    ):
        """
        Initialize inference pipeline.
        
        Args:
            jepa_model: Trained JEPA model
            anomaly_detector: Fitted Isolation Forest detector
            device: Device for inference
            pooling: Embedding pooling strategy
        """
        self.jepa_model = jepa_model
        self.anomaly_detector = anomaly_detector
        self.pooling = pooling
        
        # Setup device
        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        
        # Move model to device and set eval mode
        self.jepa_model = self.jepa_model.to(self.device)
        self.jepa_model.eval()
        
        # Initialize embedding extractor
        self.extractor = EmbeddingExtractor(
            self.jepa_model,
            device=str(self.device),
            pooling=pooling
        )
        
        logger.info(f"JEPADroneInference initialized on {self.device}")
    
    @classmethod
    def from_checkpoint(
        cls,
        jepa_path: Union[str, Path],
        detector_path: Union[str, Path],
        config: Optional[Dict] = None,
        device: str = "auto"
    ) -> "JEPADroneInference":
        """
        Load inference pipeline from saved checkpoints.
        
        Args:
            jepa_path: Path to JEPA model checkpoint
            detector_path: Path to Isolation Forest detector
            config: Optional model config (loaded from checkpoint if not provided)
            device: Device for inference
            
        Returns:
            Initialized JEPADroneInference instance
        """
        logger.info(f"Loading JEPA model from {jepa_path}...")
        
        # Load JEPA checkpoint
        checkpoint = torch.load(jepa_path, map_location="cpu")
        
        # Get config from checkpoint or use provided
        if config is None:
            config = checkpoint.get("config", {})
        
        # Create model
        model_config = config.get("model", {})
        masking_config = config.get("masking", {})
        chunking_config = config.get("chunking", {})
        
        jepa_model = JEPA(
            input_dim=11,  # Default: 5 base + 6 derived features
            embed_dim=model_config.get("embedding_dim", 256),
            encoder_hidden=model_config.get("encoder_hidden", [512, 256, 256]),
            predictor_hidden=model_config.get("predictor_hidden", [256, 256]),
            chunk_size=chunking_config.get("chunk_size", 20),
            dropout=model_config.get("dropout", 0.1),
            adaptive_masking=masking_config.get("adaptive", True),
            min_mask_ratio=masking_config.get("min_mask_ratio", 0.20),
            max_mask_ratio=masking_config.get("max_mask_ratio", 0.50),
            fixed_mask_ratio=masking_config.get("fixed_mask_ratio", 0.30)
        )
        
        # Load state dict
        jepa_model.load_state_dict(checkpoint["model_state_dict"])
        
        logger.info(f"Loading Anomaly Detector from {detector_path}...")
        anomaly_detector = AnomalyDetector.load(detector_path)
        
        return cls(jepa_model, anomaly_detector, device=device)
    
    @torch.no_grad()
    def predict(
        self,
        features: torch.Tensor,
        threshold: Optional[float] = None,
        return_embeddings: bool = False
    ) -> Union[List[AnomalyPrediction], Tuple[List[AnomalyPrediction], np.ndarray]]:
        """
        Predict anomaly for input chunks.
        
        Args:
            features: Input tensor (batch, seq_len, num_features)
            threshold: Optional custom threshold
            return_embeddings: Whether to return embeddings
            
        Returns:
            List of AnomalyPrediction objects
            Optionally: tuple of (predictions, embeddings)
        """
        # Extract embeddings
        embeddings = self.extractor.extract(features)
        embeddings_np = embeddings.cpu().numpy()
        
        # Get anomaly scores
        scores = self.anomaly_detector.score_samples(embeddings_np)
        predictions = self.anomaly_detector.predict(embeddings_np, threshold)
        
        # Convert scores to confidence (0-1 scale)
        # Use fit statistics for normalization
        fit_stats = self.anomaly_detector.fit_stats
        score_mean = fit_stats.get("score_mean", 0)
        score_std = fit_stats.get("score_std", 1)
        
        # Confidence: higher score = more normal = higher confidence in normal
        confidences = 1 / (1 + np.exp(-(scores - score_mean) / score_std))
        
        results = []
        for i in range(len(scores)):
            pred = AnomalyPrediction(
                chunk_id=i,
                is_anomaly=bool(predictions[i]),
                anomaly_score=float(scores[i]),
                confidence=float(1 - confidences[i] if predictions[i] else confidences[i])
            )
            results.append(pred)
        
        if return_embeddings:
            return results, embeddings_np
        return results
    
    def predict_batch(
        self,
        dataloader: DataLoader,
        threshold: Optional[float] = None,
        return_labels: bool = True,
        verbose: bool = True
    ) -> Tuple[List[AnomalyPrediction], Optional[np.ndarray]]:
        """
        Predict anomalies for a batch of data.
        
        Args:
            dataloader: PyTorch DataLoader
            threshold: Optional custom threshold
            return_labels: Whether to return ground truth labels
            verbose: Show progress bar
            
        Returns:
            Tuple of (predictions list, labels array or None)
        """
        from tqdm import tqdm
        
        all_predictions = []
        all_labels = []
        chunk_idx = 0
        
        iterator = tqdm(dataloader, desc="Predicting") if verbose else dataloader
        
        for batch in iterator:
            features = batch["features"]
            batch_preds = self.predict(features, threshold)
            
            # Update chunk IDs to be global
            for pred in batch_preds:
                pred.chunk_id = chunk_idx
                chunk_idx += 1
            
            all_predictions.extend(batch_preds)
            
            if return_labels and "chunk_label" in batch:
                all_labels.extend(batch["chunk_label"].numpy())
        
        labels = np.array(all_labels) if all_labels else None
        return all_predictions, labels
    
    def predict_flight(
        self,
        chunks: List[torch.Tensor],
        flight_id: str = "unknown",
        anomaly_threshold_ratio: float = 0.1
    ) -> FlightPrediction:
        """
        Aggregate chunk predictions into flight-level prediction.
        
        A flight is marked anomalous if the ratio of anomalous chunks
        exceeds the threshold.
        
        Args:
            chunks: List of chunk tensors (seq_len, num_features)
            flight_id: Identifier for the flight
            anomaly_threshold_ratio: Ratio of anomalous chunks to flag flight
            
        Returns:
            FlightPrediction with aggregated results
        """
        # Stack chunks into batch
        batch = torch.stack(chunks, dim=0)
        
        # Get predictions
        chunk_preds = self.predict(batch)
        
        # Aggregate
        anomaly_chunks = sum(1 for p in chunk_preds if p.is_anomaly)
        total_chunks = len(chunk_preds)
        anomaly_ratio = anomaly_chunks / total_chunks if total_chunks > 0 else 0
        
        scores = [p.anomaly_score for p in chunk_preds]
        
        return FlightPrediction(
            flight_id=flight_id,
            is_anomaly=anomaly_ratio >= anomaly_threshold_ratio,
            anomaly_chunks=anomaly_chunks,
            total_chunks=total_chunks,
            anomaly_ratio=anomaly_ratio,
            mean_score=float(np.mean(scores)),
            min_score=float(np.min(scores)),
            chunk_predictions=chunk_preds
        )
    
    def get_embedding(
        self,
        features: torch.Tensor,
        return_all_positions: bool = False
    ) -> np.ndarray:
        """
        Extract embeddings without anomaly detection.
        
        Args:
            features: Input tensor (batch, seq_len, num_features)
            return_all_positions: Return per-position embeddings
            
        Returns:
            Embeddings array
        """
        embeddings = self.extractor.extract(features, return_all_positions)
        return embeddings.cpu().numpy()


class GPSSpoofingDetector(JEPADroneInference):
    """
    Specialized detector for GPS spoofing (injection) attacks.
    
    Inherits from JEPADroneInference but adds GPS-specific analysis:
    - Coordinate jump detection
    - Trajectory consistency checks
    - Spoofing confidence scoring
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.gps_feature_indices = [0, 1, 2]  # lat, lon, alt
    
    def detect_coordinate_jumps(
        self,
        features: torch.Tensor,
        jump_threshold: float = 3.0  # Standard deviations
    ) -> Dict:
        """
        Detect sudden coordinate jumps that may indicate GPS spoofing.
        
        Args:
            features: Input tensor (batch, seq_len, num_features)
            jump_threshold: Threshold in std deviations
            
        Returns:
            Dictionary with jump analysis
        """
        features_np = features.cpu().numpy()
        
        results = {
            "has_jumps": [],
            "jump_locations": [],
            "jump_magnitudes": []
        }
        
        for i in range(len(features_np)):
            chunk = features_np[i]
            gps_data = chunk[:, self.gps_feature_indices]
            
            # Compute differences
            diffs = np.diff(gps_data, axis=0)
            diff_magnitudes = np.sqrt((diffs ** 2).sum(axis=1))
            
            # Find jumps
            mean_diff = diff_magnitudes.mean()
            std_diff = diff_magnitudes.std()
            threshold = mean_diff + jump_threshold * std_diff
            
            jump_mask = diff_magnitudes > threshold
            jump_locs = np.where(jump_mask)[0]
            
            results["has_jumps"].append(len(jump_locs) > 0)
            results["jump_locations"].append(jump_locs.tolist())
            results["jump_magnitudes"].append(diff_magnitudes[jump_mask].tolist())
        
        return results
    
    def predict_gps_spoofing(
        self,
        features: torch.Tensor,
        threshold: Optional[float] = None
    ) -> List[Dict]:
        """
        Specialized GPS spoofing prediction.
        
        Combines:
        1. JEPA-based anomaly detection
        2. Coordinate jump analysis
        3. Trajectory consistency
        
        Args:
            features: Input tensor (batch, seq_len, num_features)
            threshold: Optional custom threshold
            
        Returns:
            List of prediction dictionaries with GPS-specific info
        """
        # Get base predictions
        base_preds = self.predict(features, threshold)
        
        # Analyze GPS jumps
        jump_analysis = self.detect_coordinate_jumps(features)
        
        results = []
        for i, pred in enumerate(base_preds):
            result = pred.to_dict()
            result["has_gps_jumps"] = jump_analysis["has_jumps"][i]
            result["jump_locations"] = jump_analysis["jump_locations"][i]
            result["jump_magnitudes"] = jump_analysis["jump_magnitudes"][i]
            
            # Combined spoofing confidence
            if pred.is_anomaly and jump_analysis["has_jumps"][i]:
                result["spoofing_confidence"] = "high"
            elif pred.is_anomaly or jump_analysis["has_jumps"][i]:
                result["spoofing_confidence"] = "medium"
            else:
                result["spoofing_confidence"] = "low"
            
            results.append(result)
        
        return results


def run_inference(
    jepa_path: str,
    detector_path: str,
    data_path: str,
    output_path: str = "outputs/predictions.json",
    split: str = "test_balanced",
    batch_size: int = 256,
    device: str = "auto"
) -> Dict:
    """
    Run inference pipeline on test data.
    
    Args:
        jepa_path: Path to JEPA checkpoint
        detector_path: Path to anomaly detector
        data_path: Path to processed data directory
        output_path: Where to save predictions
        split: Which data split to evaluate
        batch_size: Batch size for inference
        device: Device for inference
        
    Returns:
        Results dictionary with metrics and predictions
    """
    from ..data.dataset import DroneChunkDataset
    
    logger.info("\n" + "="*60)
    logger.info("JEPA-DRONE Inference Pipeline")
    logger.info("="*60)
    
    # Load pipeline
    pipeline = JEPADroneInference.from_checkpoint(
        jepa_path=jepa_path,
        detector_path=detector_path,
        device=device
    )
    
    # Load data
    logger.info(f"\nLoading {split} data from {data_path}...")
    dataset = DroneChunkDataset(data_path, split=split, return_labels=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4
    )
    
    logger.info(f"  - Samples: {len(dataset):,}")
    
    # Run predictions
    logger.info("\nRunning inference...")
    predictions, labels = pipeline.predict_batch(dataloader, return_labels=True)
    
    # Compute metrics
    if labels is not None:
        pred_labels = np.array([p.is_anomaly for p in predictions]).astype(int)
        scores = np.array([p.anomaly_score for p in predictions])
        
        from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
        
        auc_roc = roc_auc_score(labels, -scores)
        f1 = f1_score(labels, pred_labels)
        tn, fp, fn, tp = confusion_matrix(labels, pred_labels, labels=[0, 1]).ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0
        
        metrics = {
            "auc_roc": float(auc_roc),
            "f1_score": float(f1),
            "recall": float(recall),
            "precision": float(precision),
            "false_alarm_rate": float(far),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn)
        }
        
        logger.info(f"\nResults:")
        logger.info(f"  - AUC-ROC: {auc_roc:.4f}")
        logger.info(f"  - Recall: {recall:.4f}")
        logger.info(f"  - Precision: {precision:.4f}")
        logger.info(f"  - F1 Score: {f1:.4f}")
        logger.info(f"  - False Alarm Rate: {far:.4f}")
    else:
        metrics = {}
    
    # Save results
    results = {
        "split": split,
        "n_samples": len(predictions),
        "metrics": metrics,
        "predictions": [p.to_dict() for p in predictions[:1000]]  # Save first 1000
    }
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nSaved predictions to: {output_path}")
    logger.info("="*60 + "\n")
    
    return results

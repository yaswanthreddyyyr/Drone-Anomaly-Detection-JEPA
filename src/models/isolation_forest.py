"""
JEPA-DRONE: Isolation Forest Anomaly Detector

This module implements the anomaly detection layer using Isolation Forest
on top of JEPA-learned embeddings.

Key Features:
- Fits on normal flight embeddings (unsupervised)
- Predicts anomaly scores for new embeddings
- Supports contamination tuning
- Provides per-sample and per-chunk anomaly scores
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    confusion_matrix
)

from .jepa import JEPA


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingExtractor:
    """
    Extracts embeddings from a trained JEPA model.
    
    Uses the context encoder to get 256-D embeddings for each chunk.
    """
    
    def __init__(
        self,
        model: JEPA,
        device: str = "auto",
        pooling: str = "mean"  # mean, max, cls
    ):
        """
        Initialize embedding extractor.
        
        Args:
            model: Trained JEPA model
            device: Device to run inference on
            pooling: How to pool temporal embeddings (mean, max, or cls)
        """
        self.model = model
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
        
        self.model = self.model.to(self.device)
        self.model.eval()
    
    @torch.no_grad()
    def extract(
        self,
        features: torch.Tensor,
        return_all_positions: bool = False
    ) -> torch.Tensor:
        """
        Extract embeddings from input features.
        
        Args:
            features: Input tensor (batch, seq_len, num_features)
            return_all_positions: If True, return embeddings for all positions
                                  If False, return pooled chunk embedding
                                  
        Returns:
            Embeddings tensor:
                - If return_all_positions: (batch, seq_len, embed_dim)
                - If not: (batch, embed_dim)
        """
        features = features.to(self.device)
        
        # Get embeddings from context encoder
        embeddings = self.model.context_encoder(features)  # (batch, seq_len, embed_dim)
        
        if return_all_positions:
            return embeddings
        
        # Pool temporal dimension
        if self.pooling == "mean":
            pooled = embeddings.mean(dim=1)  # (batch, embed_dim)
        elif self.pooling == "max":
            pooled = embeddings.max(dim=1)[0]  # (batch, embed_dim)
        elif self.pooling == "cls":
            # Use first position as CLS token
            pooled = embeddings[:, 0, :]  # (batch, embed_dim)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
        
        return pooled
    
    def extract_from_dataloader(
        self,
        dataloader,
        return_labels: bool = True,
        verbose: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract embeddings from a dataloader.
        
        Args:
            dataloader: PyTorch DataLoader
            return_labels: Whether to collect labels
            verbose: Show progress bar
            
        Returns:
            Tuple of (embeddings, chunk_labels, anomaly_types)
        """
        from tqdm import tqdm
        
        all_embeddings = []
        all_labels = []
        all_types = []
        
        iterator = tqdm(dataloader, desc="Extracting embeddings") if verbose else dataloader
        
        for batch in iterator:
            features = batch["features"]
            embeddings = self.extract(features)
            all_embeddings.append(embeddings.cpu().numpy())
            
            if return_labels and "chunk_label" in batch:
                all_labels.append(batch["chunk_label"].numpy())
                if "anomaly_type" in batch:
                    all_types.extend(batch["anomaly_type"])
        
        embeddings = np.concatenate(all_embeddings, axis=0)
        
        if return_labels and len(all_labels) > 0:
            labels = np.concatenate(all_labels, axis=0)
            types = np.array(all_types) if all_types else None
            return embeddings, labels, types
        
        return embeddings, None, None


class AnomalyDetector:
    """
    Isolation Forest-based anomaly detector for drone telemetry.
    
    Workflow:
    1. Extract embeddings from normal flight data using JEPA encoder
    2. Fit Isolation Forest on normal embeddings
    3. Score new embeddings - lower scores indicate anomalies
    
    Attributes:
        iso_forest: Fitted sklearn IsolationForest model
        threshold: Decision threshold for binary classification
        embedding_dim: Dimensionality of input embeddings
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float = 0.05,
        max_samples: Union[str, int] = "auto",
        max_features: float = 1.0,
        random_state: int = 42,
        threshold: Optional[float] = None
    ):
        """
        Initialize anomaly detector.
        
        Args:
            n_estimators: Number of trees in the forest
            contamination: Expected proportion of anomalies (for threshold)
            max_samples: Number of samples to draw for each tree
            max_features: Number of features to draw for each tree
            random_state: Random seed for reproducibility
            threshold: Optional custom decision threshold
        """
        self.n_estimators = n_estimators
        self.contamination = contamination
        self.max_samples = max_samples
        self.max_features = max_features
        self.random_state = random_state
        self._threshold = threshold
        
        self.iso_forest = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            max_features=max_features,
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
            warm_start=False
        )
        
        self.is_fitted = False
        self.embedding_dim = None
        self.fit_stats = {}
    
    def fit(
        self,
        embeddings: np.ndarray,
        verbose: bool = True
    ) -> "AnomalyDetector":
        """
        Fit Isolation Forest on normal flight embeddings.
        
        Args:
            embeddings: Normal flight embeddings (N, embed_dim)
            verbose: Whether to log progress
            
        Returns:
            Self (for chaining)
        """
        if verbose:
            logger.info(f"Fitting Isolation Forest on {len(embeddings):,} normal samples...")
            logger.info(f"  - n_estimators: {self.n_estimators}")
            logger.info(f"  - contamination: {self.contamination}")
            logger.info(f"  - embedding_dim: {embeddings.shape[1]}")
        
        self.embedding_dim = embeddings.shape[1]
        
        # Fit the model
        self.iso_forest.fit(embeddings)
        self.is_fitted = True
        
        # Compute fit statistics
        scores = self.iso_forest.score_samples(embeddings)
        self.fit_stats = {
            "n_samples": len(embeddings),
            "embedding_dim": self.embedding_dim,
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
            "score_percentiles": {
                "1%": float(np.percentile(scores, 1)),
                "5%": float(np.percentile(scores, 5)),
                "10%": float(np.percentile(scores, 10)),
                "50%": float(np.percentile(scores, 50)),
                "90%": float(np.percentile(scores, 90)),
                "95%": float(np.percentile(scores, 95)),
                "99%": float(np.percentile(scores, 99)),
            }
        }
        
        if verbose:
            logger.info(f"Fit complete. Score distribution:")
            logger.info(f"  - Mean: {self.fit_stats['score_mean']:.4f}")
            logger.info(f"  - Std: {self.fit_stats['score_std']:.4f}")
            logger.info(f"  - 5th percentile: {self.fit_stats['score_percentiles']['5%']:.4f}")
        
        return self
    
    def score_samples(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores for embeddings.
        
        Lower scores indicate more anomalous samples.
        
        Args:
            embeddings: Input embeddings (N, embed_dim)
            
        Returns:
            Anomaly scores (N,) - lower = more anomalous
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted before scoring")
        
        return self.iso_forest.score_samples(embeddings)
    
    def predict(
        self,
        embeddings: np.ndarray,
        threshold: Optional[float] = None
    ) -> np.ndarray:
        """
        Predict binary anomaly labels.
        
        Args:
            embeddings: Input embeddings (N, embed_dim)
            threshold: Decision threshold (default: use sklearn's)
            
        Returns:
            Binary predictions (N,) - 1 = anomaly, 0 = normal
        """
        if not self.is_fitted:
            raise RuntimeError("AnomalyDetector must be fitted before predicting")
        
        if threshold is None:
            # Use sklearn's prediction (based on contamination)
            predictions = self.iso_forest.predict(embeddings)
            # Convert from {-1, 1} to {1, 0}
            return (predictions == -1).astype(int)
        else:
            scores = self.score_samples(embeddings)
            return (scores < threshold).astype(int)
    
    @property
    def threshold(self) -> float:
        """Get the decision threshold."""
        if self._threshold is not None:
            return self._threshold
        # Estimate threshold from fit statistics
        return self.fit_stats.get("score_percentiles", {}).get("5%", -0.5)
    
    @threshold.setter
    def threshold(self, value: float):
        """Set custom decision threshold."""
        self._threshold = value
    
    def tune_threshold(
        self,
        val_embeddings: np.ndarray,
        val_labels: np.ndarray,
        metric: str = "f1",
        verbose: bool = True
    ) -> float:
        """
        Tune decision threshold on validation data.
        
        Args:
            val_embeddings: Validation embeddings (N, embed_dim)
            val_labels: Ground truth labels (N,) - 1 = anomaly
            metric: Metric to optimize ("f1", "recall", "precision")
            verbose: Whether to log progress
            
        Returns:
            Optimal threshold
        """
        scores = self.score_samples(val_embeddings)
        
        # Try different thresholds
        percentiles = np.linspace(1, 20, 50)
        best_threshold = None
        best_score = 0
        
        results = []
        for p in percentiles:
            thresh = np.percentile(scores, p)
            preds = (scores < thresh).astype(int)
            
            if metric == "f1":
                score = f1_score(val_labels, preds, zero_division=0)
            elif metric == "recall":
                tn, fp, fn, tp = confusion_matrix(val_labels, preds, labels=[0, 1]).ravel()
                score = tp / (tp + fn) if (tp + fn) > 0 else 0
            elif metric == "precision":
                tn, fp, fn, tp = confusion_matrix(val_labels, preds, labels=[0, 1]).ravel()
                score = tp / (tp + fp) if (tp + fp) > 0 else 0
            else:
                raise ValueError(f"Unknown metric: {metric}")
            
            results.append((thresh, score, p))
            
            if score > best_score:
                best_score = score
                best_threshold = thresh
        
        self._threshold = best_threshold
        
        if verbose:
            logger.info(f"Threshold tuning complete:")
            logger.info(f"  - Best threshold: {best_threshold:.4f}")
            logger.info(f"  - Best {metric}: {best_score:.4f}")
        
        return best_threshold
    
    def evaluate(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        threshold: Optional[float] = None,
        anomaly_types: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Evaluate anomaly detection performance.
        
        Args:
            embeddings: Test embeddings (N, embed_dim)
            labels: Ground truth labels (N,) - 1 = anomaly
            threshold: Decision threshold (optional)
            anomaly_types: Optional per-sample anomaly type strings
            
        Returns:
            Dictionary with evaluation metrics
        """
        scores = self.score_samples(embeddings)
        predictions = self.predict(embeddings, threshold)
        
        # Core metrics
        auc_roc = roc_auc_score(labels, -scores)  # Negate: lower score = anomaly
        auc_pr = average_precision_score(labels, -scores)
        
        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
        
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        far = fp / (fp + tn) if (fp + tn) > 0 else 0  # False Alarm Rate
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results = {
            "n_samples": len(labels),
            "n_anomalies": int(labels.sum()),
            "n_normal": int(len(labels) - labels.sum()),
            "auc_roc": float(auc_roc),
            "auc_pr": float(auc_pr),
            "recall": float(recall),
            "precision": float(precision),
            "f1_score": float(f1),
            "false_alarm_rate": float(far),
            "specificity": float(specificity),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "threshold_used": float(threshold) if threshold else self.threshold
        }
        
        # Per-anomaly-type breakdown
        if anomaly_types is not None:
            unique_types = np.unique(anomaly_types)
            per_type_results = {}
            
            for atype in unique_types:
                if atype == "normal":
                    continue
                    
                mask = anomaly_types == atype
                if mask.sum() == 0:
                    continue
                
                type_labels = labels[mask]
                type_scores = scores[mask]
                type_preds = predictions[mask]
                
                if len(np.unique(type_labels)) < 2:
                    # Can't compute AUC with single class
                    type_auc = 0.0
                else:
                    type_auc = roc_auc_score(type_labels, -type_scores)
                
                type_recall = (type_preds == type_labels).mean() if type_labels.sum() > 0 else 0
                
                per_type_results[atype] = {
                    "n_samples": int(mask.sum()),
                    "auc_roc": float(type_auc),
                    "recall": float((type_preds[type_labels == 1] == 1).mean()) if (type_labels == 1).sum() > 0 else 0
                }
            
            results["per_type"] = per_type_results
        
        return results
    
    def save(self, path: Union[str, Path]):
        """Save the fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            "iso_forest": self.iso_forest,
            "is_fitted": self.is_fitted,
            "embedding_dim": self.embedding_dim,
            "fit_stats": self.fit_stats,
            "config": {
                "n_estimators": self.n_estimators,
                "contamination": self.contamination,
                "max_samples": self.max_samples,
                "max_features": self.max_features,
                "random_state": self.random_state,
                "threshold": self._threshold
            }
        }
        
        with open(path, "wb") as f:
            pickle.dump(save_dict, f)
        
        logger.info(f"Saved AnomalyDetector to {path}")
    
    @classmethod
    def load(cls, path: Union[str, Path]) -> "AnomalyDetector":
        """Load a fitted model from disk."""
        with open(path, "rb") as f:
            save_dict = pickle.load(f)
        
        config = save_dict["config"]
        detector = cls(
            n_estimators=config["n_estimators"],
            contamination=config["contamination"],
            max_samples=config["max_samples"],
            max_features=config["max_features"],
            random_state=config["random_state"],
            threshold=config.get("threshold")
        )
        
        detector.iso_forest = save_dict["iso_forest"]
        detector.is_fitted = save_dict["is_fitted"]
        detector.embedding_dim = save_dict["embedding_dim"]
        detector.fit_stats = save_dict["fit_stats"]
        
        logger.info(f"Loaded AnomalyDetector from {path}")
        return detector


def fit_anomaly_detector(
    jepa_model: JEPA,
    train_loader,
    val_loader=None,
    config: Optional[Dict] = None,
    device: str = "auto",
    output_dir: str = "outputs"
) -> Tuple[AnomalyDetector, Dict]:
    """
    Fit Isolation Forest anomaly detector on JEPA embeddings.
    
    Complete pipeline:
    1. Extract embeddings from training data
    2. Fit Isolation Forest on normal embeddings
    3. Optionally tune threshold on validation data
    4. Evaluate and return results
    
    Args:
        jepa_model: Trained JEPA model
        train_loader: DataLoader for training data (normal only)
        val_loader: Optional DataLoader for validation (with labels)
        config: Configuration dictionary
        device: Device for embedding extraction
        output_dir: Directory to save results
        
    Returns:
        Tuple of (fitted AnomalyDetector, evaluation results dict)
    """
    config = config or {}
    if_config = config.get("isolation_forest", {})
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\n" + "="*60)
    logger.info("Fitting Isolation Forest Anomaly Detector")
    logger.info("="*60)
    
    # Step 1: Extract embeddings
    logger.info("\nStep 1: Extracting embeddings from JEPA encoder...")
    extractor = EmbeddingExtractor(jepa_model, device=device)
    
    train_embeddings, train_labels, _ = extractor.extract_from_dataloader(
        train_loader, 
        return_labels=True,
        verbose=True
    )
    
    # Filter to normal samples only for fitting
    if train_labels is not None:
        normal_mask = train_labels == 0
        normal_embeddings = train_embeddings[normal_mask]
        logger.info(f"  - Total embeddings: {len(train_embeddings):,}")
        logger.info(f"  - Normal embeddings (for fitting): {len(normal_embeddings):,}")
    else:
        normal_embeddings = train_embeddings
        logger.info(f"  - Embeddings extracted: {len(normal_embeddings):,}")
    
    # Step 2: Fit Isolation Forest
    logger.info("\nStep 2: Fitting Isolation Forest...")
    detector = AnomalyDetector(
        n_estimators=if_config.get("n_estimators", 100),
        contamination=if_config.get("contamination", 0.05),
        max_samples=if_config.get("max_samples", "auto"),
        random_state=if_config.get("random_state", 42)
    )
    
    detector.fit(normal_embeddings, verbose=True)
    
    results = {"fit_stats": detector.fit_stats}
    
    # Step 3: Tune threshold on validation data
    if val_loader is not None:
        logger.info("\nStep 3: Extracting validation embeddings...")
        val_embeddings, val_labels, val_types = extractor.extract_from_dataloader(
            val_loader,
            return_labels=True,
            verbose=True
        )
        
        logger.info(f"  - Validation samples: {len(val_embeddings):,}")
        logger.info(f"  - Anomalies: {val_labels.sum():,}")
        logger.info(f"  - Normal: {(val_labels == 0).sum():,}")
        
        # Tune threshold
        logger.info("\nStep 4: Tuning decision threshold...")
        detector.tune_threshold(val_embeddings, val_labels, metric="f1", verbose=True)
        
        # Evaluate on validation
        logger.info("\nStep 5: Evaluating on validation set...")
        val_results = detector.evaluate(val_embeddings, val_labels, anomaly_types=val_types)
        results["validation"] = val_results
        
        logger.info(f"\nValidation Results:")
        logger.info(f"  - AUC-ROC: {val_results['auc_roc']:.4f}")
        logger.info(f"  - Recall: {val_results['recall']:.4f}")
        logger.info(f"  - Precision: {val_results['precision']:.4f}")
        logger.info(f"  - F1 Score: {val_results['f1_score']:.4f}")
        logger.info(f"  - False Alarm Rate: {val_results['false_alarm_rate']:.4f}")
    
    # Save detector
    detector_path = output_dir / "anomaly_detector.pkl"
    detector.save(detector_path)
    
    # Save results
    results_path = output_dir / "anomaly_detector_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"\nSaved detector to: {detector_path}")
    logger.info(f"Saved results to: {results_path}")
    logger.info("="*60 + "\n")
    
    return detector, results

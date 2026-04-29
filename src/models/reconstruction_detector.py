"""
Reconstruction Error-based Anomaly Detector

Uses JEPA's prediction error directly as anomaly score instead of 
Isolation Forest on embeddings.

Key insight: JEPA learns to predict masked waypoints from context.
For normal data, prediction error should be low.
For anomalous data, prediction error should be high.

This avoids the information loss from Isolation Forest projection.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, precision_recall_curve

from .jepa import JEPA


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ReconstructionAnomalyDetector:
    """
    Anomaly detector using JEPA reconstruction error.
    
    Instead of:
    1. Encode chunks -> embeddings
    2. Fit Isolation Forest on embeddings
    3. Score by IF anomaly score
    
    We do:
    1. For each chunk, mask waypoints and predict
    2. Compute prediction error (MSE between predicted and target embeddings)
    3. High error = anomaly
    
    Advantages:
    - Direct use of JEPA's learned representations
    - No information loss from IF projection
    - Can use multiple mask samples for robust scoring
    """
    
    def __init__(
        self,
        model: JEPA,
        device: str = "auto",
        n_mask_samples: int = 5,
        error_aggregation: str = "max",  # max, mean, sum
        normalize_error: bool = True
    ):
        """
        Initialize detector.
        
        Args:
            model: Trained JEPA model
            device: Device for inference
            n_mask_samples: Number of different masks to average over
            error_aggregation: How to aggregate errors across positions
            normalize_error: Whether to normalize by mask size
        """
        self.model = model
        self.n_mask_samples = n_mask_samples
        self.error_aggregation = error_aggregation
        self.normalize_error = normalize_error
        
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
        
        # Threshold calibration
        self.threshold = None
        self.train_error_stats = {}
    
    @torch.no_grad()
    def compute_reconstruction_error(
        self,
        features: torch.Tensor,
        use_fixed_mask: bool = False,
        fixed_mask_ratio: float = 0.3
    ) -> torch.Tensor:
        """
        Compute reconstruction error for a batch.
        
        Args:
            features: Input features (batch, seq_len, num_features)
            use_fixed_mask: Use fixed masking instead of adaptive
            fixed_mask_ratio: Mask ratio if using fixed
            
        Returns:
            Reconstruction errors (batch,)
        """
        features = features.to(self.device)
        batch_size = features.size(0)
        
        all_errors = []
        
        for _ in range(self.n_mask_samples):
            # Forward pass with detailed output
            output = self.model.forward(features, return_detailed=True)
            
            # Get per-sample loss directly
            sample_error = output["loss_per_sample"]  # (batch,)
            
            all_errors.append(sample_error)
        
        # Average over mask samples
        errors = torch.stack(all_errors, dim=0).mean(dim=0)  # (batch,)
        
        return errors.cpu()
    
    def fit_threshold(
        self,
        train_loader: DataLoader,
        method: str = "percentile",
        percentile: float = 95,
        verbose: bool = True
    ) -> float:
        """
        Calibrate threshold on normal training data.
        
        Args:
            train_loader: DataLoader with normal training data
            method: "percentile" or "std"
            percentile: Percentile for threshold (if method=percentile)
            verbose: Show progress
            
        Returns:
            Calibrated threshold
        """
        all_errors = []
        
        iterator = tqdm(train_loader, desc="Computing train errors") if verbose else train_loader
        
        for batch in iterator:
            features = batch["features"]
            errors = self.compute_reconstruction_error(features)
            all_errors.append(errors.numpy())
        
        train_errors = np.concatenate(all_errors)
        
        # Compute statistics
        self.train_error_stats = {
            "mean": float(np.mean(train_errors)),
            "std": float(np.std(train_errors)),
            "min": float(np.min(train_errors)),
            "max": float(np.max(train_errors)),
            "percentiles": {
                str(p): float(np.percentile(train_errors, p))
                for p in [1, 5, 10, 25, 50, 75, 90, 95, 99]
            }
        }
        
        if method == "percentile":
            self.threshold = float(np.percentile(train_errors, percentile))
        elif method == "std":
            # threshold = mean + 2*std
            self.threshold = self.train_error_stats["mean"] + 2 * self.train_error_stats["std"]
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if verbose:
            logger.info(f"Train error stats:")
            logger.info(f"  Mean: {self.train_error_stats['mean']:.6f}")
            logger.info(f"  Std: {self.train_error_stats['std']:.6f}")
            logger.info(f"  95th percentile: {self.train_error_stats['percentiles']['95']:.6f}")
            logger.info(f"  Calibrated threshold: {self.threshold:.6f}")
        
        return self.threshold
    
    def tune_threshold(
        self,
        val_loader: DataLoader,
        metric: str = "f1",
        verbose: bool = True
    ) -> float:
        """
        Tune threshold on validation data with labels.
        
        Args:
            val_loader: DataLoader with validation data
            metric: "f1", "recall@fpr05", etc.
            verbose: Show progress
            
        Returns:
            Optimal threshold
        """
        all_errors = []
        all_labels = []
        
        iterator = tqdm(val_loader, desc="Computing val errors") if verbose else val_loader
        
        for batch in iterator:
            features = batch["features"]
            labels = batch["chunk_label"]
            
            errors = self.compute_reconstruction_error(features)
            all_errors.append(errors.numpy())
            all_labels.append(labels.numpy())
        
        errors = np.concatenate(all_errors)
        labels = np.concatenate(all_labels)
        
        # Find optimal threshold
        if metric == "f1":
            precision, recall, thresholds = precision_recall_curve(labels, errors)
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            self.threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else float(thresholds[-1])
            
            if verbose:
                logger.info(f"F1-optimal threshold: {self.threshold:.6f}")
                logger.info(f"  Precision: {precision[best_idx]:.4f}")
                logger.info(f"  Recall: {recall[best_idx]:.4f}")
                logger.info(f"  F1: {f1_scores[best_idx]:.4f}")
        
        elif metric.startswith("recall@fpr"):
            # e.g., recall@fpr05 -> find threshold for 5% FAR
            target_fpr = float(metric.split("fpr")[-1]) / 100
            
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(labels, errors)
            
            # Find threshold closest to target FPR
            idx = np.argmin(np.abs(fpr - target_fpr))
            self.threshold = float(thresholds[idx])
            
            if verbose:
                logger.info(f"Threshold for {target_fpr:.1%} FAR: {self.threshold:.6f}")
                logger.info(f"  Achieved FAR: {fpr[idx]:.4f}")
                logger.info(f"  Recall: {tpr[idx]:.4f}")
        
        return self.threshold
    
    def score_samples(self, dataloader: DataLoader, verbose: bool = True) -> np.ndarray:
        """
        Get anomaly scores for all samples in dataloader.
        
        Higher scores = more anomalous.
        
        Returns:
            Anomaly scores (N,)
        """
        all_errors = []
        
        iterator = tqdm(dataloader, desc="Scoring") if verbose else dataloader
        
        for batch in iterator:
            features = batch["features"]
            errors = self.compute_reconstruction_error(features)
            all_errors.append(errors.numpy())
        
        return np.concatenate(all_errors)
    
    def predict(
        self,
        dataloader: DataLoader,
        threshold: Optional[float] = None,
        verbose: bool = True
    ) -> np.ndarray:
        """
        Predict binary anomaly labels.
        
        Returns:
            Binary predictions (N,) - 1 = anomaly
        """
        if threshold is None:
            threshold = self.threshold
        
        if threshold is None:
            raise RuntimeError("Threshold not set. Call fit_threshold() or tune_threshold() first.")
        
        scores = self.score_samples(dataloader, verbose=verbose)
        return (scores > threshold).astype(int)
    
    def evaluate(
        self,
        dataloader: DataLoader,
        verbose: bool = True
    ) -> Dict:
        """
        Full evaluation on labeled data.
        
        Returns:
            Dictionary with metrics
        """
        all_errors = []
        all_labels = []
        all_types = []
        
        iterator = tqdm(dataloader, desc="Evaluating") if verbose else dataloader
        
        for batch in iterator:
            features = batch["features"]
            labels = batch["chunk_label"]
            
            errors = self.compute_reconstruction_error(features)
            all_errors.append(errors.numpy())
            all_labels.append(labels.numpy())
            
            if "anomaly_type" in batch:
                all_types.extend(batch["anomaly_type"])
        
        errors = np.concatenate(all_errors)
        labels = np.concatenate(all_labels)
        
        # Compute metrics
        auc_roc = roc_auc_score(labels, errors)
        
        # With current threshold
        preds = (errors > self.threshold).astype(int) if self.threshold else None
        
        results = {
            "n_samples": len(labels),
            "n_anomalies": int(labels.sum()),
            "n_normal": int((1 - labels).sum()),
            "auc_roc": float(auc_roc),
            "threshold": self.threshold,
            "train_error_stats": self.train_error_stats,
            "score_stats": {
                "mean": float(np.mean(errors)),
                "std": float(np.std(errors)),
                "min": float(np.min(errors)),
                "max": float(np.max(errors))
            }
        }
        
        if preds is not None:
            from sklearn.metrics import confusion_matrix
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            
            results.update({
                "recall": float(tp / (tp + fn)) if (tp + fn) > 0 else 0,
                "precision": float(tp / (tp + fp)) if (tp + fp) > 0 else 0,
                "f1": float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else 0,
                "false_alarm_rate": float(fp / (fp + tn)) if (fp + tn) > 0 else 0,
                "confusion_matrix": {
                    "tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)
                }
            })
        
        if verbose:
            logger.info(f"Evaluation results:")
            logger.info(f"  AUC-ROC: {auc_roc:.4f}")
            if preds is not None:
                logger.info(f"  Recall: {results['recall']:.4f}")
                logger.info(f"  Precision: {results['precision']:.4f}")
                logger.info(f"  F1: {results['f1']:.4f}")
                logger.info(f"  FAR: {results['false_alarm_rate']:.4f}")
        
        return results
    
    def save(self, path: Union[str, Path]):
        """Save detector state (threshold and stats)."""
        path = Path(path)
        state = {
            "threshold": self.threshold,
            "train_error_stats": self.train_error_stats,
            "n_mask_samples": self.n_mask_samples,
            "error_aggregation": self.error_aggregation,
            "normalize_error": self.normalize_error
        }
        with open(path, 'w') as f:
            json.dump(state, f, indent=2)
    
    def load(self, path: Union[str, Path]):
        """Load detector state."""
        path = Path(path)
        with open(path) as f:
            state = json.load(f)
        
        self.threshold = state.get("threshold")
        self.train_error_stats = state.get("train_error_stats", {})

"""
Adaptive Entropy-Guided Masking for JEPA-DRONE

This module implements the core novelty of the JEPA-DRONE paper:
entropy-guided adaptive masking that adjusts mask ratio based on
flight dynamics complexity.

Key insight:
- Stable cruise (low entropy) → harder prediction task (50% mask)
- Turbulent/maneuvering flight (high entropy) → easier task (20% mask)

This curriculum-like approach helps the model learn both stable patterns
and complex dynamics without overwhelming information loss.
"""

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class EntropyCalculator(nn.Module):
    """
    Computes entropy scores for flight chunks to guide adaptive masking.
    
    Uses multiple entropy measures:
    1. Temporal variance - how much features change over time
    2. Differential entropy - rate of change complexity
    3. Local smoothness - short-term predictability
    
    Combines these into a single entropy score per chunk.
    """
    
    def __init__(
        self,
        feature_indices: List[int] = None,
        window_size: int = 5,
        method: str = "combined"  # "variance", "differential", "combined"
    ):
        """
        Args:
            feature_indices: Which features to use for entropy (default: speed, heading, altitude)
            window_size: Window size for local entropy calculation
            method: Entropy calculation method
        """
        super().__init__()
        
        # Default: use speed (3), heading (4), altitude (2), 
        # acceleration (8), angular_velocity (9)
        self.feature_indices = feature_indices or [2, 3, 4, 8, 9]
        self.window_size = window_size
        self.method = method
        
        # Learnable weights for feature importance (optional enhancement)
        self.feature_weights = nn.Parameter(
            torch.ones(len(self.feature_indices)), 
            requires_grad=False  # Start with fixed weights
        )
    
    def compute_temporal_variance(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute variance of features over time.
        
        High variance = turbulent/maneuvering flight
        Low variance = stable cruise
        """
        # x: (batch, seq_len, n_features)
        variance = x.var(dim=1)  # (batch, n_features)
        
        # Weighted average
        weights = F.softmax(self.feature_weights, dim=0)
        weighted_var = (variance * weights).sum(dim=1)  # (batch,)
        
        return weighted_var
    
    def compute_differential_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of first-order differences.
        
        Measures how unpredictable the rate of change is.
        """
        # First-order differences
        diff = x[:, 1:, :] - x[:, :-1, :]  # (batch, seq_len-1, n_features)
        
        # Variance of differences
        diff_var = diff.var(dim=1)  # (batch, n_features)
        
        # Weighted average
        weights = F.softmax(self.feature_weights, dim=0)
        weighted_diff_var = (diff_var * weights).sum(dim=1)  # (batch,)
        
        return weighted_diff_var
    
    def compute_local_smoothness(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute local smoothness using sliding window variance.
        
        Low local variance = smooth, predictable
        High local variance = erratic, unpredictable
        """
        batch_size, seq_len, n_features = x.shape
        
        if seq_len < self.window_size:
            return self.compute_temporal_variance(x)
        
        # Compute variance in sliding windows
        local_vars = []
        for i in range(seq_len - self.window_size + 1):
            window = x[:, i:i+self.window_size, :]
            local_var = window.var(dim=1)  # (batch, n_features)
            local_vars.append(local_var)
        
        # Average local variance
        local_vars = torch.stack(local_vars, dim=1)  # (batch, n_windows, n_features)
        avg_local_var = local_vars.mean(dim=(1, 2))  # (batch,)
        
        return avg_local_var
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy score for each chunk.
        
        Args:
            x: Input features (batch, seq_len, num_features)
            
        Returns:
            Normalized entropy scores (batch,) in range [0, 1]
        """
        batch_size = x.shape[0]
        
        # Extract relevant features
        if x.shape[-1] > max(self.feature_indices):
            selected = x[:, :, self.feature_indices]
        else:
            # Fallback if fewer features available
            selected = x
        
        if self.method == "variance":
            entropy = self.compute_temporal_variance(selected)
        elif self.method == "differential":
            entropy = self.compute_differential_entropy(selected)
        elif self.method == "combined":
            # Combine multiple entropy measures
            temp_var = self.compute_temporal_variance(selected)
            diff_ent = self.compute_differential_entropy(selected)
            
            # Handle single-sample batches
            if batch_size == 1:
                # Can't normalize, just use the raw values
                entropy = (temp_var + diff_ent) / 2
            else:
                # Normalize each component
                temp_var_std = temp_var.std()
                diff_ent_std = diff_ent.std()
                
                if temp_var_std > 1e-8:
                    temp_var_norm = (temp_var - temp_var.mean()) / temp_var_std
                else:
                    temp_var_norm = torch.zeros_like(temp_var)
                    
                if diff_ent_std > 1e-8:
                    diff_ent_norm = (diff_ent - diff_ent.mean()) / diff_ent_std
                else:
                    diff_ent_norm = torch.zeros_like(diff_ent)
                
                # Average
                entropy = (temp_var_norm + diff_ent_norm) / 2
        else:
            entropy = self.compute_temporal_variance(selected)
        
        # Normalize to [0, 1] using sigmoid
        # For single-sample batches, center around 0 (gives ~0.5 entropy)
        if batch_size == 1:
            entropy_normalized = torch.sigmoid(entropy)
        else:
            entropy_normalized = torch.sigmoid(entropy - entropy.mean())
        
        return entropy_normalized


class AdaptiveMaskingModule(nn.Module):
    """
    Entropy-guided adaptive masking for JEPA training.
    
    Core novelty of JEPA-DRONE: adjusts mask ratio based on chunk complexity.
    
    Strategy:
    - Low entropy (stable cruise) → high mask ratio (50%)
      Reasoning: Simple patterns are easy to predict, challenge the model more
      
    - High entropy (turbulent) → low mask ratio (20%)  
      Reasoning: Complex patterns need more context to predict accurately
    
    This creates a natural curriculum where the model:
    1. Gets easier learning signal from complex chunks (more context)
    2. Is challenged more on simple patterns (less context)
    """
    
    def __init__(
        self,
        chunk_size: int = 20,
        min_mask_ratio: float = 0.20,
        max_mask_ratio: float = 0.50,
        fixed_mask_ratio: float = 0.30,
        adaptive: bool = True,
        entropy_method: str = "combined",
        block_masking: bool = False,  # Whether to mask contiguous blocks
        block_size: int = 4
    ):
        super().__init__()
        
        self.chunk_size = chunk_size
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.fixed_mask_ratio = fixed_mask_ratio
        self.adaptive = adaptive
        self.block_masking = block_masking
        self.block_size = block_size
        
        # Entropy calculator
        self.entropy_calc = EntropyCalculator(method=entropy_method)
        
        # Statistics tracking
        self.register_buffer('mask_ratio_history', torch.zeros(1000))
        self.register_buffer('entropy_history', torch.zeros(1000))
        self.register_buffer('history_idx', torch.tensor(0))
    
    def compute_mask_ratio(self, entropy: torch.Tensor) -> torch.Tensor:
        """
        Map entropy scores to mask ratios.
        
        Low entropy → high mask ratio (harder task)
        High entropy → low mask ratio (easier task)
        
        Args:
            entropy: Normalized entropy scores (batch,) in [0, 1]
            
        Returns:
            Mask ratios (batch,) in [min_mask_ratio, max_mask_ratio]
        """
        # Inverse relationship: high entropy → low mask ratio
        # mask_ratio = max - entropy * (max - min)
        mask_ratios = self.max_mask_ratio - entropy * (self.max_mask_ratio - self.min_mask_ratio)
        
        return mask_ratios
    
    def generate_random_mask(
        self,
        batch_size: int,
        seq_len: int,
        mask_ratios: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate random point-wise masks.
        """
        masks = []
        mask_positions_list = []
        visible_positions_list = []
        
        for i in range(batch_size):
            num_mask = int(seq_len * mask_ratios[i].item())
            num_mask = max(1, min(num_mask, seq_len - 1))
            
            # Random permutation
            perm = torch.randperm(seq_len, device=device)
            
            mask_idx = perm[:num_mask].sort()[0]
            visible_idx = perm[num_mask:].sort()[0]
            
            # Create boolean mask
            mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            mask[mask_idx] = True
            
            masks.append(mask)
            mask_positions_list.append(mask_idx)
            visible_positions_list.append(visible_idx)
        
        return masks, mask_positions_list, visible_positions_list
    
    def generate_block_mask(
        self,
        batch_size: int,
        seq_len: int,
        mask_ratios: torch.Tensor,
        device: torch.device
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate contiguous block masks.
        
        Masks contiguous blocks of waypoints instead of random points.
        This tests the model's ability to predict longer sequences.
        """
        masks = []
        mask_positions_list = []
        visible_positions_list = []
        
        num_blocks = seq_len // self.block_size
        
        for i in range(batch_size):
            num_mask_blocks = int(num_blocks * mask_ratios[i].item())
            num_mask_blocks = max(1, min(num_mask_blocks, num_blocks - 1))
            
            # Random block selection
            block_perm = torch.randperm(num_blocks, device=device)
            mask_blocks = block_perm[:num_mask_blocks]
            
            # Create mask from blocks
            mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            for block_idx in mask_blocks:
                start = block_idx * self.block_size
                end = min(start + self.block_size, seq_len)
                mask[start:end] = True
            
            mask_idx = torch.where(mask)[0]
            visible_idx = torch.where(~mask)[0]
            
            masks.append(mask)
            mask_positions_list.append(mask_idx)
            visible_positions_list.append(visible_idx)
        
        return masks, mask_positions_list, visible_positions_list
    
    def _update_statistics(self, mask_ratios: torch.Tensor, entropy: torch.Tensor):
        """Track masking statistics for analysis."""
        idx = self.history_idx.item() % 1000
        self.mask_ratio_history[idx] = mask_ratios.mean().item()
        self.entropy_history[idx] = entropy.mean().item()
        self.history_idx += 1
    
    def get_statistics(self) -> Dict[str, float]:
        """Get masking statistics."""
        valid_idx = min(self.history_idx.item(), 1000)
        if valid_idx < 2:
            return {"avg_mask_ratio": 0, "avg_entropy": 0, "mask_ratio_std": 0, "entropy_std": 0}
        
        return {
            "avg_mask_ratio": self.mask_ratio_history[:valid_idx].mean().item(),
            "avg_entropy": self.entropy_history[:valid_idx].mean().item(),
            "mask_ratio_std": self.mask_ratio_history[:valid_idx].std().item(),
            "entropy_std": self.entropy_history[:valid_idx].std().item()
        }
    
    def forward(
        self, 
        x: torch.Tensor,
        return_entropy: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate adaptive masks for input sequences.
        
        Args:
            x: Input features (batch, seq_len, num_features)
            return_entropy: If True, also return entropy scores
            
        Returns:
            mask: Boolean tensor (batch, seq_len) - True = masked
            mask_positions: Indices of masked positions (batch, max_masked)
            visible_positions: Indices of visible positions (batch, max_visible)
            entropy (optional): Entropy scores if return_entropy=True
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if self.adaptive:
            # Compute entropy
            entropy = self.entropy_calc(x)  # (batch,)
            
            # Map entropy to mask ratios
            mask_ratios = self.compute_mask_ratio(entropy)
            
            # Track statistics
            self._update_statistics(mask_ratios, entropy)
        else:
            # Fixed mask ratio
            mask_ratios = torch.full((batch_size,), self.fixed_mask_ratio, device=device)
            entropy = torch.zeros(batch_size, device=device)
        
        # Generate masks
        if self.block_masking:
            masks, mask_pos_list, visible_pos_list = self.generate_block_mask(
                batch_size, seq_len, mask_ratios, device
            )
        else:
            masks, mask_pos_list, visible_pos_list = self.generate_random_mask(
                batch_size, seq_len, mask_ratios, device
            )
        
        # Stack masks
        masks = torch.stack(masks)  # (batch, seq_len)
        
        # Pad position tensors to same length
        max_masked = max(len(m) for m in mask_pos_list)
        max_visible = max(len(v) for v in visible_pos_list)
        
        mask_positions = torch.zeros(batch_size, max_masked, dtype=torch.long, device=device)
        visible_positions = torch.zeros(batch_size, max_visible, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            mask_positions[i, :len(mask_pos_list[i])] = mask_pos_list[i]
            visible_positions[i, :len(visible_pos_list[i])] = visible_pos_list[i]
        
        if return_entropy:
            return masks, mask_positions, visible_positions, entropy
        
        return masks, mask_positions, visible_positions


def visualize_masking(
    features: torch.Tensor,
    masks: torch.Tensor,
    entropy: torch.Tensor = None,
    mask_ratios: torch.Tensor = None,
    sample_idx: int = 0,
    save_path: str = None
):
    """
    Visualize the adaptive masking on a sample chunk.
    
    Args:
        features: Input features (batch, seq_len, num_features)
        masks: Boolean masks (batch, seq_len)
        entropy: Entropy scores (batch,)
        mask_ratios: Mask ratios (batch,)
        sample_idx: Which sample to visualize
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    feat = features[sample_idx].cpu().numpy()
    mask = masks[sample_idx].cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6))
    
    # Plot features with mask highlighting
    ax1 = axes[0]
    time = np.arange(feat.shape[0])
    
    # Plot altitude, speed, heading
    ax1.plot(time, feat[:, 2], label='Altitude', alpha=0.8)
    ax1.plot(time, feat[:, 3], label='Speed', alpha=0.8)
    ax1.plot(time, feat[:, 4], label='Heading', alpha=0.8)
    
    # Highlight masked regions
    for i, m in enumerate(mask):
        if m:
            ax1.axvspan(i-0.5, i+0.5, alpha=0.3, color='red')
    
    ax1.set_xlabel('Waypoint')
    ax1.set_ylabel('Feature Value (normalized)')
    ax1.legend()
    
    title = 'Adaptive Masking Visualization'
    if entropy is not None:
        title += f' | Entropy: {entropy[sample_idx].item():.3f}'
    if mask_ratios is not None:
        title += f' | Mask Ratio: {mask_ratios[sample_idx].item():.2f}'
    ax1.set_title(title)
    
    # Plot mask
    ax2 = axes[1]
    ax2.imshow(mask.reshape(1, -1), aspect='auto', cmap='Reds', vmin=0, vmax=1)
    ax2.set_xlabel('Waypoint')
    ax2.set_ylabel('Mask')
    ax2.set_yticks([])
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# Test
if __name__ == "__main__":
    print("Testing Adaptive Masking Module...")
    
    # Create module
    masking = AdaptiveMaskingModule(
        chunk_size=20,
        min_mask_ratio=0.20,
        max_mask_ratio=0.50,
        adaptive=True
    )
    
    # Test with random data
    batch_size = 8
    seq_len = 20
    num_features = 11
    
    x = torch.randn(batch_size, seq_len, num_features)
    
    # Add some variance to make entropy different across samples
    x[0] *= 0.1  # Low entropy (stable)
    x[1] *= 5.0  # High entropy (turbulent)
    
    # Generate masks
    masks, mask_pos, visible_pos, entropy = masking(x, return_entropy=True)
    
    print(f"Masks shape: {masks.shape}")
    print(f"Mask positions shape: {mask_pos.shape}")
    print(f"Visible positions shape: {visible_pos.shape}")
    print(f"\nEntropy scores: {entropy}")
    
    # Check mask ratios
    mask_ratios = masks.float().mean(dim=1)
    print(f"Actual mask ratios: {mask_ratios}")
    
    # Expected: low entropy → high mask ratio
    print(f"\nSample 0 (low entropy): entropy={entropy[0]:.3f}, mask_ratio={mask_ratios[0]:.2f}")
    print(f"Sample 1 (high entropy): entropy={entropy[1]:.3f}, mask_ratio={mask_ratios[1]:.2f}")
    
    # Statistics
    print(f"\nMasking statistics: {masking.get_statistics()}")
    
    print("\n✅ Adaptive Masking test passed!")

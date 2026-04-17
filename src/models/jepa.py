"""
JEPA-DRONE: Joint Embedding Predictive Architecture for Drone Anomaly Detection

This module implements the core JEPA model components:
- Context Encoder (fθ): Encodes visible waypoints into latent space
- Target Encoder (fθ̄): EMA-updated encoder for target embeddings
- Predictor (gφ): Predicts masked waypoint embeddings from context
- Masking Module: Fixed and entropy-guided adaptive masking

Based on: Assran et al. "Self-supervised learning from images with a 
joint-embedding predictive architecture" CVPR 2023

Adapted for GPS telemetry time series.
"""

import math
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from .adaptive_masking import AdaptiveMaskingModule


class PositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding for temporal sequences.
    
    Adds position information to waypoint embeddings so the model
    knows the temporal order of waypoints in a chunk.
    """
    
    def __init__(self, d_model: int, max_len: int = 100, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        # Register as buffer (not a parameter, but saved in state_dict)
        self.register_buffer('pe', pe.unsqueeze(0))  # Shape: (1, max_len, d_model)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, d_model)
            
        Returns:
            Tensor with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class WaypointEmbedding(nn.Module):
    """
    Projects raw waypoint features into embedding space.
    
    Takes 11-dimensional telemetry features and projects them
    to the model's embedding dimension.
    """
    
    def __init__(self, input_dim: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        
        self.projection = nn.Sequential(
            nn.Linear(input_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Raw features of shape (batch, seq_len, input_dim)
            
        Returns:
            Embeddings of shape (batch, seq_len, embed_dim)
        """
        return self.projection(x)


class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron encoder for waypoint sequences.
    
    This is the context encoder (fθ) that processes each waypoint
    independently through a shared MLP, then applies temporal
    aggregation via self-attention or pooling.
    """
    
    def __init__(
        self,
        input_dim: int,
        embed_dim: int = 256,
        hidden_dims: List[int] = [512, 256, 256],
        dropout: float = 0.1,
        use_attention: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.use_attention = use_attention
        
        # Waypoint embedding
        self.waypoint_embed = WaypointEmbedding(input_dim, hidden_dims[0], dropout)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dims[0], max_len=100, dropout=dropout)
        
        # MLP layers
        layers = []
        in_dim = hidden_dims[0]
        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        self.mlp = nn.Sequential(*layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dims[-1], embed_dim)
        self.output_norm = nn.LayerNorm(embed_dim)
        
        # Optional: Self-attention for temporal context
        if use_attention:
            self.self_attn = nn.MultiheadAttention(
                embed_dim=hidden_dims[-1],
                num_heads=4,
                dropout=dropout,
                batch_first=True
            )
            self.attn_norm = nn.LayerNorm(hidden_dims[-1])
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode waypoint sequences.
        
        Args:
            x: Input features of shape (batch, seq_len, input_dim)
            mask: Optional boolean mask of shape (batch, seq_len)
                  True = masked (excluded from attention), False = visible
                  
        Returns:
            Embeddings of shape (batch, seq_len, embed_dim)
        """
        # Embed waypoints
        h = self.waypoint_embed(x)  # (batch, seq_len, hidden_dims[0])
        
        # Add positional encoding
        h = self.pos_encoding(h)
        
        # Apply MLP
        h = self.mlp(h)  # (batch, seq_len, hidden_dims[-1])
        
        # Optional: Self-attention for temporal context
        if self.use_attention:
            # Create attention mask if needed
            attn_mask = None
            if mask is not None:
                # Convert boolean mask to attention mask (float)
                # True positions should be -inf (masked out)
                attn_mask = mask.float().masked_fill(mask, float('-inf'))
            
            # Self-attention
            attn_out, _ = self.self_attn(h, h, h, key_padding_mask=mask)
            h = self.attn_norm(h + attn_out)
        
        # Project to output dimension
        out = self.output_proj(h)
        out = self.output_norm(out)
        
        return out


class Predictor(nn.Module):
    """
    Predictor network (gφ) that predicts target embeddings from context.
    
    Takes context embeddings (from visible waypoints) and position indices
    of masked waypoints, and predicts what the target embeddings should be.
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        hidden_dims: List[int] = [256, 256],
        num_positions: int = 20,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_positions = num_positions
        
        # Learnable position embeddings for mask positions
        self.mask_token = nn.Parameter(torch.randn(1, 1, embed_dim) * 0.02)
        self.pos_embed = nn.Embedding(num_positions, embed_dim)
        
        # Context aggregation (attention over visible positions)
        self.context_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=4,
            dropout=dropout,
            batch_first=True
        )
        self.context_norm = nn.LayerNorm(embed_dim)
        
        # Prediction MLP
        layers = []
        in_dim = embed_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            ])
            in_dim = hidden_dim
        
        layers.append(nn.Linear(hidden_dims[-1], embed_dim))
        self.predictor_mlp = nn.Sequential(*layers)
    
    def forward(
        self,
        context_embeds: torch.Tensor,
        mask_positions: torch.Tensor,
        visible_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Predict embeddings for masked positions.
        
        Args:
            context_embeds: Embeddings from context encoder
                           Shape: (batch, seq_len, embed_dim)
            mask_positions: Indices of masked positions
                           Shape: (batch, num_masked)
            visible_mask: Boolean mask where True = visible (not masked)
                         Shape: (batch, seq_len)
                         
        Returns:
            Predicted embeddings for masked positions
            Shape: (batch, num_masked, embed_dim)
        """
        batch_size, seq_len, embed_dim = context_embeds.shape
        num_masked = mask_positions.shape[1]
        
        # Create mask token queries with position embeddings
        # Shape: (batch, num_masked, embed_dim)
        mask_tokens = self.mask_token.expand(batch_size, num_masked, -1)
        pos_embeds = self.pos_embed(mask_positions)  # (batch, num_masked, embed_dim)
        queries = mask_tokens + pos_embeds
        
        # Use attention to gather context from visible positions
        # Key padding mask: True = exclude from attention
        key_padding_mask = ~visible_mask  # Invert: True = masked position (exclude)
        
        # Cross-attention: queries attend to context
        attn_out, _ = self.context_attn(
            query=queries,
            key=context_embeds,
            value=context_embeds,
            key_padding_mask=key_padding_mask
        )
        
        attended = self.context_norm(queries + attn_out)
        
        # Predict target embeddings
        predictions = self.predictor_mlp(attended)
        
        return predictions


class MaskingModule(nn.Module):
    """
    Masking module for JEPA training.
    
    Supports two masking strategies:
    1. Fixed masking: Random mask with fixed ratio (e.g., 30%)
    2. Adaptive masking: Entropy-guided masking (20-50%)
       - Low-entropy (stable) chunks get harder masks (50%)
       - High-entropy (turbulent) chunks get lighter masks (20%)
    """
    
    def __init__(
        self,
        chunk_size: int = 20,
        min_mask_ratio: float = 0.20,
        max_mask_ratio: float = 0.50,
        fixed_mask_ratio: float = 0.30,
        adaptive: bool = False,
        entropy_features: List[int] = [3, 4, 2]  # speed, heading, altitude indices
    ):
        super().__init__()
        
        self.chunk_size = chunk_size
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.fixed_mask_ratio = fixed_mask_ratio
        self.adaptive = adaptive
        self.entropy_features = entropy_features
    
    def compute_entropy(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute entropy of selected features to determine masking ratio.
        
        Higher entropy = more variable/turbulent flight
        Lower entropy = stable cruise
        
        Args:
            x: Input features (batch, seq_len, num_features)
            
        Returns:
            Entropy scores per batch (batch,)
        """
        # Extract entropy-relevant features (speed, heading, altitude changes)
        selected = x[:, :, self.entropy_features]  # (batch, seq_len, n_selected)
        
        # Compute variance as proxy for entropy
        # Higher variance = higher entropy
        variance = selected.var(dim=1).mean(dim=1)  # (batch,)
        
        # Normalize to [0, 1] using sigmoid
        # This maps the variance to a probability-like value
        entropy = torch.sigmoid(variance - variance.mean())
        
        return entropy
    
    def forward(
        self, 
        x: torch.Tensor,
        deterministic: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate masks for the input sequence.
        
        Args:
            x: Input features (batch, seq_len, num_features)
            deterministic: If True, use fixed seed for reproducibility
            
        Returns:
            mask: Boolean tensor (batch, seq_len) - True = masked
            mask_positions: Indices of masked positions (batch, num_masked)
            visible_positions: Indices of visible positions (batch, num_visible)
        """
        batch_size, seq_len, _ = x.shape
        device = x.device
        
        if self.adaptive:
            # Compute entropy for adaptive masking
            entropy = self.compute_entropy(x)  # (batch,)
            
            # Map entropy to mask ratio
            # High entropy (turbulent) -> lower mask ratio (easier)
            # Low entropy (stable) -> higher mask ratio (harder)
            mask_ratios = self.max_mask_ratio - entropy * (self.max_mask_ratio - self.min_mask_ratio)
        else:
            # Fixed mask ratio for all samples
            mask_ratios = torch.full((batch_size,), self.fixed_mask_ratio, device=device)
        
        # Generate random masks
        masks = []
        mask_positions_list = []
        visible_positions_list = []
        
        for i in range(batch_size):
            num_mask = int(seq_len * mask_ratios[i].item())
            num_mask = max(1, min(num_mask, seq_len - 1))  # At least 1 masked, at least 1 visible
            
            # Random permutation
            if deterministic:
                torch.manual_seed(42 + i)
            perm = torch.randperm(seq_len, device=device)
            
            mask_idx = perm[:num_mask]
            visible_idx = perm[num_mask:]
            
            # Create boolean mask
            mask = torch.zeros(seq_len, dtype=torch.bool, device=device)
            mask[mask_idx] = True
            
            masks.append(mask)
            mask_positions_list.append(mask_idx.sort()[0])
            visible_positions_list.append(visible_idx.sort()[0])
        
        # Stack masks
        masks = torch.stack(masks)  # (batch, seq_len)
        
        # Pad position tensors to same length
        max_masked = max(len(m) for m in mask_positions_list)
        max_visible = max(len(v) for v in visible_positions_list)
        
        mask_positions = torch.zeros(batch_size, max_masked, dtype=torch.long, device=device)
        visible_positions = torch.zeros(batch_size, max_visible, dtype=torch.long, device=device)
        
        for i in range(batch_size):
            mask_positions[i, :len(mask_positions_list[i])] = mask_positions_list[i]
            visible_positions[i, :len(visible_positions_list[i])] = visible_positions_list[i]
        
        return masks, mask_positions, visible_positions


class JEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture for Drone Anomaly Detection.
    
    The model learns to predict representations of masked waypoints from
    visible context, without requiring any labels.
    
    Architecture:
        1. Context Encoder (fθ): Encodes visible waypoints
        2. Target Encoder (fθ̄): EMA of context encoder, encodes all waypoints
        3. Predictor (gφ): Predicts target embeddings from context
        
    Training:
        - Mask random waypoints
        - Encode visible waypoints with context encoder
        - Encode all waypoints with target encoder (stop gradient)
        - Predict masked waypoint embeddings from context
        - Loss = MSE between predictions and target embeddings
        
    Inference:
        - Extract embeddings from context encoder
        - Use for anomaly detection (Isolation Forest)
    """
    
    def __init__(
        self,
        input_dim: int = 11,
        embed_dim: int = 256,
        encoder_hidden: List[int] = [512, 256, 256],
        predictor_hidden: List[int] = [256, 256],
        chunk_size: int = 20,
        dropout: float = 0.1,
        ema_decay: float = 0.996,
        # Masking parameters
        adaptive_masking: bool = False,
        min_mask_ratio: float = 0.20,
        max_mask_ratio: float = 0.50,
        fixed_mask_ratio: float = 0.30
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.ema_decay = ema_decay
        
        # Context encoder (fθ)
        self.context_encoder = MLPEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dims=encoder_hidden,
            dropout=dropout,
            use_attention=True
        )
        
        # Target encoder (fθ̄) - EMA of context encoder
        self.target_encoder = MLPEncoder(
            input_dim=input_dim,
            embed_dim=embed_dim,
            hidden_dims=encoder_hidden,
            dropout=dropout,
            use_attention=True
        )
        
        # Initialize target encoder with context encoder weights
        self._init_target_encoder()
        
        # Freeze target encoder (updated only via EMA)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        
        # Predictor (gφ)
        self.predictor = Predictor(
            embed_dim=embed_dim,
            hidden_dims=predictor_hidden,
            num_positions=chunk_size,
            dropout=dropout
        )
        
        # Masking module (enhanced adaptive version)
        self.masking = AdaptiveMaskingModule(
            chunk_size=chunk_size,
            min_mask_ratio=min_mask_ratio,
            max_mask_ratio=max_mask_ratio,
            fixed_mask_ratio=fixed_mask_ratio,
            adaptive=adaptive_masking,
            entropy_method="combined"
        )
        
        # Store config for statistics
        self.adaptive_masking = adaptive_masking
    
    def _init_target_encoder(self):
        """Initialize target encoder with context encoder weights."""
        for param_q, param_k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_k.data.copy_(param_q.data)
    
    @torch.no_grad()
    def update_target_encoder(self):
        """Update target encoder using exponential moving average."""
        for param_q, param_k in zip(
            self.context_encoder.parameters(),
            self.target_encoder.parameters()
        ):
            param_k.data = param_k.data * self.ema_decay + param_q.data * (1 - self.ema_decay)
    
    def forward(
        self, 
        x: torch.Tensor,
        return_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass for training.
        
        Args:
            x: Input features (batch, seq_len, input_dim)
            return_embeddings: If True, also return target embeddings
            
        Returns:
            loss: JEPA prediction loss
            embeddings (optional): Target embeddings for the full sequence
        """
        batch_size, seq_len, _ = x.shape
        
        # Generate masks
        mask, mask_positions, visible_positions = self.masking(x)
        
        # Encode with context encoder (only visible positions matter for prediction)
        context_embeds = self.context_encoder(x, mask=mask)  # (batch, seq_len, embed_dim)
        
        # Encode with target encoder (no gradient)
        with torch.no_grad():
            target_embeds = self.target_encoder(x)  # (batch, seq_len, embed_dim)
        
        # Get target embeddings at masked positions
        target_at_mask = self._gather_at_positions(target_embeds, mask_positions)
        
        # Predict embeddings for masked positions
        visible_mask = ~mask  # True = visible
        predictions = self.predictor(context_embeds, mask_positions, visible_mask)
        
        # Compute loss: MSE between predictions and targets
        # Only compare at valid masked positions
        loss = F.mse_loss(predictions, target_at_mask.detach())
        
        if return_embeddings:
            return loss, target_embeds
        
        return loss
    
    def _gather_at_positions(
        self, 
        embeds: torch.Tensor, 
        positions: torch.Tensor
    ) -> torch.Tensor:
        """
        Gather embeddings at specified positions.
        
        Args:
            embeds: (batch, seq_len, embed_dim)
            positions: (batch, num_positions)
            
        Returns:
            (batch, num_positions, embed_dim)
        """
        batch_size, num_positions = positions.shape
        embed_dim = embeds.shape[-1]
        
        # Expand positions for gathering
        positions = positions.unsqueeze(-1).expand(-1, -1, embed_dim)
        
        return torch.gather(embeds, dim=1, index=positions)
    
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input for inference (anomaly detection).
        
        Uses the context encoder to produce embeddings.
        
        Args:
            x: Input features (batch, seq_len, input_dim)
            
        Returns:
            Embeddings (batch, seq_len, embed_dim)
        """
        self.eval()
        return self.context_encoder(x)
    
    @torch.no_grad()
    def get_chunk_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get a single embedding vector per chunk.
        
        Pools waypoint embeddings into a single chunk representation.
        
        Args:
            x: Input features (batch, seq_len, input_dim)
            
        Returns:
            Chunk embeddings (batch, embed_dim)
        """
        embeds = self.encode(x)  # (batch, seq_len, embed_dim)
        
        # Mean pooling across sequence
        chunk_embed = embeds.mean(dim=1)  # (batch, embed_dim)
        
        return chunk_embed
    
    def get_masking_statistics(self) -> Dict:
        """Get adaptive masking statistics."""
        return self.masking.get_statistics()
    
    def set_masking_mode(self, adaptive: bool):
        """Switch between adaptive and fixed masking."""
        self.masking.adaptive = adaptive
        self.adaptive_masking = adaptive

def create_jepa_model(config: Dict) -> JEPA:
    """
    Create a JEPA model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Initialized JEPA model
    """
    model_config = config["model"]
    masking_config = config["masking"]
    chunk_size = config["chunking"]["chunk_size"]
    
    # Count input features
    num_base = len(config["features"]["base_features"])
    num_derived = len(config["features"]["derived_features"]) if config["features"]["use_derived"] else 0
    input_dim = num_base + num_derived
    
    model = JEPA(
        input_dim=input_dim,
        embed_dim=model_config["embedding_dim"],
        encoder_hidden=model_config["encoder_hidden"],
        predictor_hidden=model_config["predictor_hidden"],
        chunk_size=chunk_size,
        dropout=model_config["dropout"],
        adaptive_masking=masking_config["adaptive"],
        min_mask_ratio=masking_config["min_mask_ratio"],
        max_mask_ratio=masking_config["max_mask_ratio"],
        fixed_mask_ratio=masking_config["fixed_mask_ratio"]
    )
    
    return model


# Test the model
if __name__ == "__main__":
    # Quick test
    print("Testing JEPA model...")
    
    # Create model
    model = JEPA(
        input_dim=11,
        embed_dim=256,
        encoder_hidden=[512, 256, 256],
        predictor_hidden=[256, 256],
        chunk_size=20,
        dropout=0.1,
        adaptive_masking=False,
        fixed_mask_ratio=0.30
    )
    
    # Test forward pass
    batch_size = 4
    seq_len = 20
    input_dim = 11
    
    x = torch.randn(batch_size, seq_len, input_dim)
    
    # Training forward
    loss = model(x)
    print(f"Loss: {loss.item():.4f}")
    
    # Get embeddings
    embeddings = model.encode(x)
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Get chunk embeddings
    chunk_embeds = model.get_chunk_embedding(x)
    print(f"Chunk embeddings shape: {chunk_embeds.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✅ JEPA model test passed!")

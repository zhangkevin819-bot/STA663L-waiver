"""
Modern PyTorch architectures with attention mechanisms and efficient training.
Implements transformer blocks, residual connections, and layer normalization.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from torch import Tensor


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embeddings (RoPE) for improved position encoding."""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)
        self.max_seq_len = max_seq_len
        self._seq_len_cached = None
        self._cos_cached = None
        self._sin_cached = None
    
    def _update_cache(self, seq_len: int, device: torch.device):
        """Update cached sin/cos values if needed."""
        if seq_len != self._seq_len_cached:
            self._seq_len_cached = seq_len
            t = torch.arange(seq_len, device=device).type_as(self.inv_freq)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            emb = torch.cat((freqs, freqs), dim=-1)
            self._cos_cached = emb.cos()
            self._sin_cached = emb.sin()
    
    def forward(self, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Apply rotary embeddings to queries and keys."""
        seq_len = q.shape[1]
        self._update_cache(seq_len, q.device)
        
        # Apply rotation
        q_rot = self._apply_rotation(q, self._cos_cached, self._sin_cached)
        k_rot = self._apply_rotation(k, self._cos_cached, self._sin_cached)
        
        return q_rot, k_rot
    
    @staticmethod
    def _apply_rotation(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        """Apply rotation matrix."""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([
            x1 * cos - x2 * sin,
            x2 * cos + x1 * sin
        ], dim=-1)


class MultiHeadAttention(nn.Module):
    """Multi-head attention with optional flash attention and rotary embeddings."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_rope: bool = True,
        use_flash: bool = False
    ):
        super().__init__()
        assert dim % num_heads == 0
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash
        
        # Projections
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        self.rope = RotaryPositionalEmbedding(self.head_dim) if use_rope else None
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: Input tensor of shape (batch, seq_len, dim)
            mask: Optional attention mask
        Returns:
            Output tensor of shape (batch, seq_len, dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv,
            'b s (three h d) -> three b h s d',
            three=3,
            h=self.num_heads
        )
        
        # Apply rotary embeddings
        if self.rope is not None:
            q, k = self.rope(
                rearrange(q, 'b h s d -> b s (h d)'),
                rearrange(k, 'b h s d -> b s (h d)')
            )
            q = rearrange(q, 'b s (h d) -> b h s d', h=self.num_heads)
            k = rearrange(k, 'b s (h d) -> b h s d', h=self.num_heads)
        
        # Compute attention
        if self.use_flash and hasattr(F, 'scaled_dot_product_attention'):
            # Use PyTorch's flash attention if available
            out = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=mask,
                dropout_p=self.dropout.p if self.training else 0.0
            )
        else:
            # Standard attention
            scores = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale
            
            if mask is not None:
                scores = scores.masked_fill(mask == 0, float('-inf'))
            
            attn = F.softmax(scores, dim=-1)
            attn = self.dropout(attn)
            
            out = torch.einsum('bhij,bhjd->bhid', attn, v)
        
        # Reshape and project output
        out = rearrange(out, 'b h s d -> b s (h d)')
        out = self.out_proj(out)
        
        return out


class FeedForward(nn.Module):
    """Position-wise feed-forward network with GLU activation."""
    
    def __init__(
        self,
        dim: int,
        mult: int = 4,
        dropout: float = 0.1,
        activation: str = "gelu"
    ):
        super().__init__()
        hidden_dim = dim * mult
        
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim * 2),
            nn.GLU(dim=-1) if activation == "glu" else self._get_activation(activation),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim)
        )
    
    @staticmethod
    def _get_activation(name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),
            "mish": nn.Mish()
        }
        return activations.get(name, nn.GELU())
    
    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Transformer block with pre-normalization and residual connections."""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        ff_mult: int = 4,
        use_rope: bool = True
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            use_rope=use_rope
        )
        
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim=dim, mult=ff_mult, dropout=dropout)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """Forward pass with pre-norm and residuals."""
        # Attention block
        x = x + self.dropout(self.attn(self.norm1(x), mask))
        
        # Feed-forward block
        x = x + self.dropout(self.ff(self.norm2(x)))
        
        return x


class TransformerClassifier(nn.Module):
    """Complete transformer model for classification tasks."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        num_layers: int = 6,
        num_heads: int = 8,
        num_classes: int = 2,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        use_rope: bool = True
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Input embedding
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Positional encoding (if not using RoPE)
        if not use_rope:
            self.pos_encoding = nn.Parameter(
                torch.randn(1, max_seq_len, hidden_dim)
            )
        else:
            self.pos_encoding = None
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                use_rope=use_rope
            )
            for _ in range(num_layers)
        ])
        
        # Output head
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
    
    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, input_dim) or (batch, input_dim)
            mask: Optional attention mask
        Returns:
            Logits (batch, num_classes)
        """
        # Handle both sequential and flat inputs
        if x.dim() == 2:
            x = x.unsqueeze(1)  # Add sequence dimension
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        if self.pos_encoding is not None:
            x = x + self.pos_encoding[:, :x.size(1)]
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Global average pooling
        x = self.norm(x)
        x = x.mean(dim=1)
        
        # Classification head
        return self.head(x)
    
    def get_attention_weights(self, x: Tensor) -> list[Tensor]:
        """Extract attention weights from all layers for visualization."""
        weights = []
        
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        x = self.input_proj(x)
        if self.pos_encoding is not None:
            x = x + self.pos_encoding[:, :x.size(1)]
        
        for block in self.blocks:
            # This is simplified - actual implementation would need
            # to modify attention module to return weights
            weights.append(None)
            x = block(x)
        
        return weights

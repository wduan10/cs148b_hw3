"""Rotary Position Embeddings — §6.

You implement: RoPE1D, RoPE2D.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def _apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply one round of rotary embedding to x.

    Args:
        x:   (B, num_heads, T, d)  — d must be even
        cos: (T, d // 2)
        sin: (T, d // 2)

    The 2D rotation formula for each pair (x_{2i}, x_{2i+1}) at position m:
        x_{2i}'   =  x_{2i}   * cos(m·θ_i) − x_{2i+1} * sin(m·θ_i)
        x_{2i+1}' =  x_{2i}   * sin(m·θ_i) + x_{2i+1} * cos(m·θ_i)
    """
    # Reshape for broadcasting: (1, 1, T, d//2)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)

    x_even = x[..., 0::2]  # (B, H, T, d//2)
    x_odd  = x[..., 1::2]  # (B, H, T, d//2)

    x_even_rot = x_even * cos - x_odd * sin
    x_odd_rot  = x_even * sin + x_odd * cos

    # Interleave even/odd back into (B, H, T, d)
    return torch.stack([x_even_rot, x_odd_rot], dim=-1).flatten(-2)


class RoPE1D(nn.Module):
    """1D Rotary Position Embedding.

    For a vector x at position m, RoPE groups dimensions into d/2 pairs and
    rotates each pair (x_{2i}, x_{2i+1}) by angle m * theta_i, where
        theta_i = base ** (-2i / head_dim).

    Apply RoPE to queries and keys (not values) inside attention, before
    computing q @ k^T.

    Args:
        head_dim:    Dimensionality of each attention head. Must be even.
        max_seq_len: Maximum sequence length to precompute angles for.
        base:        Base of the geometric progression (typically 10_000).

    Forward:
        x:         (B, num_heads, T, head_dim)
        positions: (T,) integer tensor of token positions.
        returns:   (B, num_heads, T, head_dim) with RoPE applied.
    """

    def __init__(self, head_dim: int, max_seq_len: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 2 == 0, "head_dim must be even for RoPE"
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.base = base

        # inv_freq[i] = base^(−2i / head_dim),  shape (head_dim // 2,)
        inv_freq = base ** (-torch.arange(0, head_dim, 2).float() / head_dim)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv_freq)          # (max_seq_len, head_dim // 2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(self, x: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        cos = self.cos_cached[positions]   # (T, head_dim // 2)
        sin = self.sin_cached[positions]
        return _apply_rope(x, cos, sin)


class RoPE2D(nn.Module):
    """2D Rotary Position Embedding for image patches.

    Splits head_dim in half. The first half rotates by the patch's x-coordinate
    using 1D RoPE; the second half rotates by the patch's y-coordinate. After
    rotation, dot products depend on the 2D *relative* offset between patches.

    Args:
        head_dim:  Must be divisible by 4 (since each half is split into
                   real/imaginary pairs).
        grid_size: Maximum grid side (patches per row).
        base:      Base of the geometric progression.

    Forward:
        x:        (B, num_heads, T, head_dim)
        x_coords: (T,) integer tensor of x positions on the grid.
        y_coords: (T,) integer tensor of y positions on the grid.
        returns:  (B, num_heads, T, head_dim) with 2D RoPE applied.
    """

    def __init__(self, head_dim: int, grid_size: int, base: float = 10_000.0) -> None:
        super().__init__()
        assert head_dim % 4 == 0, "head_dim must be divisible by 4 for 2D RoPE"
        self.head_dim = head_dim
        self.grid_size = grid_size
        self.base = base

        # Each half (x-axis / y-axis) has head_dim//2 dims, so head_dim//4 pairs.
        half_dim = head_dim // 2
        inv_freq = base ** (-torch.arange(0, half_dim, 2).float() / half_dim)
        t = torch.arange(grid_size).float()
        freqs = torch.outer(t, inv_freq)          # (grid_size, head_dim // 4)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        x_coords: torch.Tensor,
        y_coords: torch.Tensor,
    ) -> torch.Tensor:
        half = self.head_dim // 2

        cos_x = self.cos_cached[x_coords]   # (T, head_dim // 4)
        sin_x = self.sin_cached[x_coords]
        cos_y = self.cos_cached[y_coords]
        sin_y = self.sin_cached[y_coords]

        # First half of head_dim rotated by x-coordinate,
        # second half rotated by y-coordinate.
        x1_rot = _apply_rope(x[..., :half], cos_x, sin_x)
        x2_rot = _apply_rope(x[..., half:], cos_y, sin_y)
        return torch.cat([x1_rot, x2_rot], dim=-1)

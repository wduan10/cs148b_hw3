"""Rotary Position Embeddings — §6.

You implement: RoPE1D, RoPE2D.
Bonus: MRoPE.
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


class MRoPE(nn.Module):
    """Multimodal Rotary Position Embedding (M-RoPE).

    Extends 2D RoPE to handle mixed image–text sequences by assigning every
    token a *triple* of position indices (temporal, row, col):

    • Text tokens at sequential position t:
          pos = (t, t, t)   — all three axes carry the same index.
    • Image tokens at 2D grid cell (r, c) with modal position m:
          pos = (m, r, c)   — temporal is fixed per image block; row and col
                               carry the 2D patch coordinates.

    head_dim is split into three **equal** segments of size seg = head_dim // 3.
    Segment 0 is rotated by the temporal index, segment 1 by the row index, and
    segment 2 by the column index.  Because each segment uses independent
    frequencies, the dot-product q·k decomposes as a sum of three independent
    inner products, each depending only on the *relative* offset along one axis.

    For text–text pairs:   attention ~ f(Δt)
    For image–image pairs: attention ~ f(Δt=0, Δrow, Δcol) = g(Δrow, Δcol)
    For text–image pairs:  attention ~ f(Δt, Δrow=Δt, Δcol=Δt)

    Requirements:
        head_dim % 6 == 0   (three segments, each must be even for rotation)

    Typical compatible configs (d_model, num_heads → head_dim):
        (384, 8) → 48    (288, 6) → 48    (192, 8) → 24

    Args:
        head_dim:      Per-head embedding dimension. Must satisfy head_dim % 6 == 0.
        max_positions: Maximum position index to precompute (covers all three axes).
        base:          Base for geometric frequency progression.

    Forward:
        x:            (B, num_heads, T, head_dim)
        pos_temporal: (T,) integer tensor — modal / sequence position.
        pos_row:      (T,) integer tensor — row / height position.
        pos_col:      (T,) integer tensor — column / width position.
        returns:      (B, num_heads, T, head_dim) with M-RoPE applied.
    """

    def __init__(
        self,
        head_dim: int,
        max_positions: int = 1024,
        base: float = 10_000.0,
    ) -> None:
        super().__init__()
        assert head_dim % 6 == 0, (
            f"head_dim={head_dim} must be divisible by 6 for M-RoPE "
            f"(3 equal even-dimensional segments). "
            f"Compatible (d_model, num_heads) examples: (384,8)→48, (288,6)→48."
        )
        self.head_dim = head_dim
        self.seg_dim  = head_dim // 3   # size of each of the 3 segments

        # All three axes share the same frequency table (same positional scale).
        seg   = self.seg_dim
        inv_freq = base ** (-torch.arange(0, seg, 2).float() / seg)
        t         = torch.arange(max_positions).float()
        freqs     = torch.outer(t, inv_freq)        # (max_positions, seg // 2)
        self.register_buffer("cos_cached", freqs.cos(), persistent=False)
        self.register_buffer("sin_cached", freqs.sin(), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        pos_temporal: torch.Tensor,
        pos_row: torch.Tensor,
        pos_col: torch.Tensor,
    ) -> torch.Tensor:
        s = self.seg_dim

        cos_t = self.cos_cached[pos_temporal]   # (T, seg // 2)
        sin_t = self.sin_cached[pos_temporal]
        cos_r = self.cos_cached[pos_row]
        sin_r = self.sin_cached[pos_row]
        cos_c = self.cos_cached[pos_col]
        sin_c = self.sin_cached[pos_col]

        x0 = _apply_rope(x[..., :s],      cos_t, sin_t)   # temporal segment
        x1 = _apply_rope(x[..., s:2 * s], cos_r, sin_r)   # row segment
        x2 = _apply_rope(x[..., 2 * s:],  cos_c, sin_c)   # column segment
        return torch.cat([x0, x1, x2], dim=-1)


def build_mrope_positions(
    n_text_before: int,
    grid_h: int,
    grid_w: int,
    n_text_after: int,
    image_temporal_pos: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build (temporal, row, col) position triples for an image–text sequence.

    Layout:
        [text_before (n_text_before tokens)]
        [image patches (grid_h * grid_w tokens)]
        [text_after (n_text_after tokens)]

    Assignment rules:
        • text_before[i]  → (i,           i,      i     )
        • image[r, c]     → (img_t,        r,      c     )  img_t = n_text_before
        • text_after[j]   → (img_t + 1 + j, img_t + 1 + j, img_t + 1 + j)

    The temporal index of text tokens after the image skips over the image's
    slot so that text-to-text relative distances are unaffected by image size.

    Args:
        n_text_before: Number of text tokens before the image.
        grid_h, grid_w: Patch grid dimensions.
        n_text_after:  Number of text tokens after the image.
        image_temporal_pos: Override the temporal index assigned to image tokens
            (default: n_text_before).

    Returns:
        Three (T,) long tensors: pos_temporal, pos_row, pos_col.
    """
    img_t = n_text_before if image_temporal_pos is None else image_temporal_pos

    pos_t, pos_r, pos_c = [], [], []

    # text before image
    for i in range(n_text_before):
        pos_t.append(i); pos_r.append(i); pos_c.append(i)

    # image patches (raster order)
    for r in range(grid_h):
        for c in range(grid_w):
            pos_t.append(img_t); pos_r.append(r); pos_c.append(c)

    # text after image
    for j in range(n_text_after):
        t = img_t + 1 + j
        pos_t.append(t); pos_r.append(t); pos_c.append(t)

    return (
        torch.tensor(pos_t, dtype=torch.long),
        torch.tensor(pos_r, dtype=torch.long),
        torch.tensor(pos_c, dtype=torch.long),
    )

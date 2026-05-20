"""Vision Transformer — §2.

You implement: PatchEmbeddings, ViT.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from basics.model import Block, Head


class PatchEmbeddings(nn.Module):
    """Split an image into non-overlapping patches and project each to d_model.

    Implemented with a strided Conv2d whose kernel size and stride both equal
    `patch_size`.

    Args:
        img_size:   Input image side length (assumed square). Must be divisible
                    by patch_size.
        patch_size: Side length of each patch in pixels.
        d_model:    Output embedding dimension per patch.

    Forward:
        x: (B, 3, img_size, img_size) float tensor.
        returns: (B, num_patches, d_model) where num_patches = (img_size // patch_size) ** 2.
    """

    def __init__(self, img_size: int, patch_size: int, d_model: int) -> None:
        super().__init__()
        assert img_size % patch_size == 0, "img_size must be divisible by patch_size"
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(
            in_channels=3,
            out_channels=d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) -> conv -> (B, d_model, H/P, W/P)
        # flatten spatial grid -> (B, d_model, num_patches)
        # transpose -> (B, num_patches, d_model)
        x = self.proj(x)          # (B, d_model, grid, grid)
        x = x.flatten(2)          # (B, d_model, num_patches)
        x = x.transpose(1, 2)     # (B, num_patches, d_model)
        return x


class RoPEHead(nn.Module):
    """Drop-in replacement for basics.model.Head that applies 1D RoPE to q, k.

    Wraps an existing Head and re-implements forward, routing q/k through RoPE
    before computing attention. Values are unaffected (RoPE is not applied to v).

    Positions: CLS token → 0, patch i → i+1.
    """

    def __init__(self, base_head: Head, rope: nn.Module) -> None:
        super().__init__()
        # Delegate all projections and the dropout to the wrapped head.
        self.q_proj   = base_head.q_proj
        self.k_proj   = base_head.k_proj
        self.v_proj   = base_head.v_proj
        self.dropout  = base_head.dropout
        self.head_dim = base_head.head_dim
        self.is_decoder = base_head.is_decoder
        if self.is_decoder:
            self.register_buffer("tril", base_head.tril, persistent=False)
        self.rope = rope

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x)   # (B, T, head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        positions = torch.arange(T, device=x.device)
        # RoPE expects (B, num_heads, T, head_dim); we have one head so num_heads=1.
        q = self.rope(q.unsqueeze(1), positions).squeeze(1)
        k = self.rope(k.unsqueeze(1), positions).squeeze(1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.is_decoder:
            attn = attn.masked_fill(~self.tril[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return attn @ v


class RoPE2DHead(nn.Module):
    """Drop-in replacement for basics.model.Head that applies 2D RoPE to q, k.

    Patch positions are their 2D grid coordinates (row, col).  The CLS token is
    assigned (row=0, col=0).  Coordinates are derived on-the-fly from the actual
    sequence length, so this head works transparently at any image resolution
    (length extrapolation requires only that RoPE2D was built with a large enough
    grid_size).
    """

    def __init__(self, base_head: Head, rope2d: nn.Module) -> None:
        super().__init__()
        self.q_proj    = base_head.q_proj
        self.k_proj    = base_head.k_proj
        self.v_proj    = base_head.v_proj
        self.dropout   = base_head.dropout
        self.head_dim  = base_head.head_dim
        self.is_decoder = base_head.is_decoder
        if self.is_decoder:
            self.register_buffer("tril", base_head.tril, persistent=False)
        self.rope2d = rope2d

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q = self.q_proj(x)   # (B, T, head_dim)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Build (row, col) coordinates for every token.
        # Sequence layout: [CLS, patch_0, patch_1, ..., patch_{N-1}]
        # Patch i occupies grid cell (i // grid, i % grid).
        n_patches = T - 1
        grid = int(math.isqrt(n_patches))
        patch_idx = torch.arange(n_patches, device=x.device)
        row_coords = torch.cat([
            torch.zeros(1, dtype=torch.long, device=x.device),
            patch_idx // grid,
        ])  # (T,)
        col_coords = torch.cat([
            torch.zeros(1, dtype=torch.long, device=x.device),
            patch_idx % grid,
        ])  # (T,)

        # RoPE2D expects (B, num_heads, T, head_dim); here num_heads=1.
        q = self.rope2d(q.unsqueeze(1), row_coords, col_coords).squeeze(1)
        k = self.rope2d(k.unsqueeze(1), row_coords, col_coords).squeeze(1)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if self.is_decoder:
            attn = attn.masked_fill(~self.tril[:T, :T], float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        return attn @ v


class ViT(nn.Module):
    """Vision Transformer.

    Pipeline:
      1. Patchify with `PatchEmbeddings`.
      2. Prepend a learnable [CLS] token.
      3a. (pe="learned") Add a learnable positional embedding (1, N+1, d_model).
      3b. (pe="rope")    No additive PE; apply 1D RoPE to q, k inside every head
                         (positions: CLS=0, patch_i=i+1).
      3c. (pe="rope2d")  No additive PE; apply 2D RoPE to q, k using patch
                         (row, col) grid coordinates (CLS maps to (0, 0)).
      4. Pass the sequence through `num_blocks` Transformer Blocks.
      5. Apply a final LayerNorm.
      6. Return the [CLS] slice — shape (B, d_model) — or all tokens.

    Args:
        img_size, patch_size, d_model, num_heads, num_blocks, dropout
        pe: "learned" (default) | "rope" | "rope2d"
    """

    def __init__(
        self,
        img_size: int,
        patch_size: int,
        d_model: int,
        num_heads: int,
        num_blocks: int,
        dropout: float = 0.1,
        pe: str = "learned",
    ) -> None:
        super().__init__()
        assert pe in ("learned", "rope", "rope2d"), f"Unknown pe mode: {pe!r}"
        self.patch_embed = PatchEmbeddings(img_size, patch_size, d_model)
        self.num_patches = self.patch_embed.num_patches
        self.d_model = d_model
        self.pe_mode = pe

        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        if pe == "learned":
            self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, d_model))
        # For pe="rope" / "rope2d", no pos_embed — positions live inside heads.

        block_size = self.num_patches + 1
        self.blocks = nn.ModuleList([
            Block(d_model, num_heads, block_size, is_decoder=False, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.norm = nn.LayerNorm(d_model)

        head_dim = d_model // num_heads
        if pe == "rope":
            self._install_rope(head_dim)
        elif pe == "rope2d":
            assert head_dim % 4 == 0, (
                f"head_dim={head_dim} must be divisible by 4 for 2D RoPE"
            )
            self._install_rope2d(head_dim)

    def _install_rope(self, head_dim: int) -> None:
        """Replace every Head in every Block with a RoPEHead."""
        from basics.rope import RoPE1D
        # Use a large max_seq_len so RoPE works for extrapolation too.
        rope = RoPE1D(head_dim, max_seq_len=1024)
        for block in self.blocks:
            heads = block.attn.heads
            for i in range(len(heads)):
                heads[i] = RoPEHead(heads[i], rope)

    def _install_rope2d(self, head_dim: int) -> None:
        """Replace every Head in every Block with a RoPE2DHead."""
        from basics.rope import RoPE2D
        train_grid = int(math.isqrt(self.num_patches))
        # Build with extra headroom so extrapolation to larger grids works.
        rope2d = RoPE2D(head_dim, grid_size=max(train_grid * 2, 32))
        for block in self.blocks:
            heads = block.attn.heads
            for i in range(len(heads)):
                heads[i] = RoPE2DHead(heads[i], rope2d)

    def interpolate_pos_embed(self, new_num_patches: int) -> None:
        """Bilinearly interpolate the learned patch positional embeddings to a
        new grid size, updating pos_embed in-place.

        Used for the length-extrapolation evaluation (e.g. 64→144 patches).
        The CLS embedding (index 0) is kept unchanged; only the N patch
        embeddings are interpolated.

        Args:
            new_num_patches: Target number of patches (must be a perfect square).
        """
        assert self.pe_mode == "learned", "interpolate_pos_embed only applies to pe='learned'"
        old_n = self.num_patches
        new_n = new_num_patches
        if old_n == new_n:
            return

        old_grid = int(math.isqrt(old_n))
        new_grid = int(math.isqrt(new_n))
        assert old_grid * old_grid == old_n, "old num_patches must be a perfect square"
        assert new_grid * new_grid == new_n, "new num_patches must be a perfect square"

        cls_pe    = self.pos_embed[:, :1, :]            # (1, 1, d_model)
        patch_pe  = self.pos_embed[:, 1:, :]            # (1, old_n, d_model)

        # Reshape to spatial grid and bilinearly upsample.
        d = self.d_model
        patch_pe  = patch_pe.reshape(1, old_grid, old_grid, d).permute(0, 3, 1, 2)
        patch_pe  = F.interpolate(patch_pe, size=(new_grid, new_grid),
                                  mode="bilinear", align_corners=False)
        patch_pe  = patch_pe.permute(0, 2, 3, 1).reshape(1, new_n, d)

        self.pos_embed = nn.Parameter(torch.cat([cls_pe, patch_pe], dim=1))

    def forward(self, x: torch.Tensor, return_all_tokens: bool = False) -> torch.Tensor:
        B = x.shape[0]
        # 1. patchify
        x = self.patch_embed(x)                               # (B, N, d_model)
        # 2. prepend CLS token
        cls = self.cls_token.expand(B, -1, -1)                # (B, 1, d_model)
        x = torch.cat([cls, x], dim=1)                        # (B, N+1, d_model)
        # 3. positional encoding
        if self.pe_mode == "learned":
            x = x + self.pos_embed                            # (B, N+1, d_model)
        # for "rope": no additive PE; RoPE is applied inside each head
        # 4. transformer blocks
        for block in self.blocks:
            x = block(x)
        # 5. final layer norm
        x = self.norm(x)                                      # (B, N+1, d_model)
        # 6. return CLS or all tokens
        if return_all_tokens:
            return x
        return x[:, 0, :]                                     # (B, d_model)

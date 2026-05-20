"""Vision-Language Model — §5.

You implement: VisionLanguageModel.

Three injection strategies to support:
  - "cls":          Single visual token (the ViT's CLS embedding) prepended.
  - "all_patches":  All N+1 visual tokens (CLS + patches) prepended.
  - "interleaved":  A special <image> token in the prompt is replaced by the
                    sequence of patch embeddings at runtime.

Two attention masking strategies to support (Problem `masking`):
  - "causal":         Fully causal across the whole sequence.
  - "image_bidir":    Bidirectional within the image block, causal everywhere
                      else. Use vlm.masking.build_image_bidir_mask().
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn

InjectionMode = Literal["cls", "all_patches", "interleaved"]
MaskMode = Literal["causal", "image_bidir"]


class VisionLanguageModel(nn.Module):
    """ViT image encoder + projector + pretrained causal LM decoder."""

    def __init__(
        self,
        vit: nn.Module,
        projector: nn.Module,
        decoder: nn.Module,
        tokenizer,
        image_token_id: int | None = None,
    ) -> None:
        super().__init__()
        self.vit = vit
        self.projector = projector
        self.decoder = decoder
        self.tokenizer = tokenizer
        self.image_token_id = image_token_id

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _encode_images(
        self, images: torch.Tensor, injection: InjectionMode
    ) -> torch.Tensor:
        """Run ViT and projector; always returns (B, N_vis, d_decoder)."""
        if injection == "cls":
            vis_feats = self.vit(images)                        # (B, d_model)
        else:
            vis_feats = self.vit(images, return_all_tokens=True)  # (B, N+1, d_model)
        return self.projector(vis_feats)                        # (B, N_vis, d_dec)

    def _embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.decoder.model.embed_tokens(input_ids)

    def _stitch_prepend(
        self,
        vis_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Concatenate [visual | text] for cls / all_patches modes."""
        B, N_vis = vis_embeds.shape[:2]
        inputs_embeds = torch.cat([vis_embeds, text_embeds], dim=1)
        vis_mask = torch.ones(B, N_vis, device=attention_mask.device,
                              dtype=attention_mask.dtype)
        full_mask = torch.cat([vis_mask, attention_mask], dim=1)
        if labels is not None:
            vis_lbl = torch.full((B, N_vis), -100,
                                 device=labels.device, dtype=labels.dtype)
            full_labels = torch.cat([vis_lbl, labels], dim=1)
        else:
            full_labels = None
        return inputs_embeds, full_mask, full_labels

    def _stitch_interleaved(
        self,
        vis_embeds: torch.Tensor,
        text_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Replace the <image> token with the visual patch sequence."""
        B, N_vis, d_dec = vis_embeds.shape
        device = vis_embeds.device

        new_embeds_list: list[torch.Tensor] = []
        new_mask_list:   list[torch.Tensor] = []
        new_lbl_list:    list[torch.Tensor] = []

        for b in range(B):
            img_pos = (input_ids[b] == self.image_token_id).nonzero(as_tuple=False)
            if len(img_pos) == 0:
                # No placeholder: simply prepend
                emb = torch.cat([vis_embeds[b], text_embeds[b]], dim=0)
                msk = torch.cat([
                    torch.ones(N_vis, device=device, dtype=attention_mask.dtype),
                    attention_mask[b],
                ], dim=0)
                lbl = (torch.cat([
                    torch.full((N_vis,), -100, device=device, dtype=labels.dtype),
                    labels[b],
                ], dim=0) if labels is not None else None)
            else:
                p = img_pos[0, 0].item()
                emb = torch.cat([text_embeds[b, :p], vis_embeds[b], text_embeds[b, p + 1:]], dim=0)
                msk = torch.cat([
                    attention_mask[b, :p],
                    torch.ones(N_vis, device=device, dtype=attention_mask.dtype),
                    attention_mask[b, p + 1:],
                ], dim=0)
                if labels is not None:
                    lbl = torch.cat([
                        labels[b, :p],
                        torch.full((N_vis,), -100, device=device, dtype=labels.dtype),
                        labels[b, p + 1:],
                    ], dim=0)
                else:
                    lbl = None

            new_embeds_list.append(emb)
            new_mask_list.append(msk)
            if lbl is not None:
                new_lbl_list.append(lbl)

        max_len = max(e.shape[0] for e in new_embeds_list)
        inputs_embeds = torch.zeros(B, max_len, d_dec, device=device, dtype=vis_embeds.dtype)
        full_mask     = torch.zeros(B, max_len, device=device, dtype=attention_mask.dtype)
        full_labels   = (torch.full((B, max_len), -100, device=device, dtype=labels.dtype)
                         if labels is not None else None)

        for b, (emb, msk) in enumerate(zip(new_embeds_list, new_mask_list)):
            L = emb.shape[0]
            inputs_embeds[b, :L] = emb
            full_mask[b, :L]     = msk
            if full_labels is not None:
                full_labels[b, :L] = new_lbl_list[b]

        return inputs_embeds, full_mask, full_labels

    # ── Forward ───────────────────────────────────────────────────────────────

    def forward(
        self,
        images: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        injection: InjectionMode = "cls",
        mask_mode: MaskMode = "causal",
    ) -> dict:
        # 1 & 2. Visual features → projected embeddings (B, N_vis, d_dec)
        vis_embeds = self._encode_images(images, injection)
        N_vis = vis_embeds.shape[1]

        # 3. Text token embeddings
        text_embeds = self._embed_tokens(input_ids)  # (B, T, d_dec)
        T = text_embeds.shape[1]

        # 4. Stitch
        if injection in ("cls", "all_patches"):
            inputs_embeds, full_mask, full_labels = self._stitch_prepend(
                vis_embeds, text_embeds, attention_mask, labels
            )
        else:  # interleaved
            inputs_embeds, full_mask, full_labels = self._stitch_interleaved(
                vis_embeds, text_embeds, input_ids, attention_mask, labels
            )

        # 5. Attention mask
        T_total = inputs_embeds.shape[1]
        if mask_mode == "image_bidir":
            from vlm.masking import build_image_bidir_mask
            n_text = T_total - N_vis
            dtype  = inputs_embeds.dtype
            attn_4d = build_image_bidir_mask(
                N_vis, n_text, device=inputs_embeds.device, dtype=dtype
            )  # (1, 1, T_total, T_total)
            # Incorporate padding: columns for pad tokens → -inf
            pad_cols = (full_mask == 0).unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T_total)
            attn_4d  = attn_4d.masked_fill(pad_cols, torch.finfo(dtype).min)
            decoder_mask = attn_4d.expand(images.shape[0], -1, -1, -1)
        else:
            decoder_mask = full_mask  # 2-D; decoder applies causal mask on top

        # 6. Decoder forward
        out = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=decoder_mask,
            labels=full_labels,
        )

        result: dict = {"logits": out.logits}
        if out.loss is not None:
            result["loss"] = out.loss
        return result

    # ── Generation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate(
        self,
        images: torch.Tensor,
        prompts: list[str],
        injection: InjectionMode = "cls",
        max_new_tokens: int = 32,
        **gen_kwargs,
    ) -> list[str]:
        """Generate text answers conditioned on images + prompt strings."""
        B = images.shape[0]
        device = images.device

        vis_embeds = self._encode_images(images, injection)  # (B, N_vis, d_dec)
        N_vis = vis_embeds.shape[1]

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(device)
        text_embeds = self._embed_tokens(enc.input_ids)  # (B, T, d_dec)

        if injection in ("cls", "all_patches"):
            vis_mask = torch.ones(B, N_vis, device=device,
                                  dtype=enc.attention_mask.dtype)
            inputs_embeds = torch.cat([vis_embeds, text_embeds], dim=1)
            attn_mask     = torch.cat([vis_mask, enc.attention_mask], dim=1)
        else:  # interleaved
            inputs_embeds, attn_mask, _ = self._stitch_interleaved(
                vis_embeds, text_embeds, enc.input_ids, enc.attention_mask, None
            )

        output_ids = self.decoder.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            max_new_tokens=max_new_tokens,
            **gen_kwargs,
        )
        return self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)

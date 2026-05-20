"""§3 — CLIP-style pretraining on EuroSAT.

You implement the training loop. This script provides the CLI scaffolding,
config loading, and logging hooks.

Usage:
    uv run python scripts/pretrain_clip.py --config configs/clip_eurosat.yaml
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--wandb", action="store_true", help="Log to W&B")
    p.add_argument(
        "--pe",
        choices=["learned", "rope"],
        default="learned",
        help="Positional encoding: 'learned' (additive) or 'rope' (1D RoPE on q/k)",
    )
    p.add_argument(
        "--extrapolate-img-size",
        type=int,
        default=None,
        help="If set, run length-extrapolation eval at this image size after training.",
    )
    return p.parse_args()


def make_lr_lambda(warmup_steps: int, total_steps: int):
    """Linear warmup then cosine decay to 0."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = Path(f"runs/clip_eurosat_{args.pe}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── 1. Data ──────────────────────────────────────────────────────────────
    from vlm.data import EUROSAT_CLASSES, build_eurosat_loaders

    train_cfg = cfg["train"]
    train_dl, val_dl, test_dl = build_eurosat_loaders(
        img_size=cfg["vit"]["img_size"],
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )

    # ── 2. Models ─────────────────────────────────────────────────────────────
    from basics.text_encoder import FrozenTextEncoder
    from basics.vit import ViT

    vit_cfg = cfg["vit"]
    vit = ViT(
        img_size=vit_cfg["img_size"],
        patch_size=vit_cfg["patch_size"],
        d_model=vit_cfg["d_model"],
        num_heads=vit_cfg["num_heads"],
        num_blocks=vit_cfg["num_blocks"],
        dropout=vit_cfg["dropout"],
        pe=args.pe,
    ).to(device)

    text_encoder = FrozenTextEncoder(cfg["text_encoder"]["model_name"])
    text_encoder = text_encoder.to(device)

    # ── 3. Projection heads + logit scale ─────────────────────────────────────
    from vlm.clip import ProjectionHeads, clip_loss, init_logit_scale

    d_proj = cfg["projection"]["d_proj"]
    proj_heads = ProjectionHeads(
        d_image=vit_cfg["d_model"],
        d_text=text_encoder.embedding_dim,
        d_proj=d_proj,
    ).to(device)

    logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07), device=device))

    # ── 4. Optimizer + scheduler ──────────────────────────────────────────────
    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        list(vit.parameters()) + list(proj_heads.parameters()) + [logit_scale],
        lr=optim_cfg["lr"],
        weight_decay=optim_cfg["weight_decay"],
        betas=tuple(optim_cfg["betas"]),
    )

    num_epochs = train_cfg["num_epochs"]
    steps_per_epoch = len(train_dl)
    total_steps = num_epochs * steps_per_epoch
    warmup_steps = optim_cfg["warmup_steps"]

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(warmup_steps, total_steps)
    )

    # ── W&B (optional) ────────────────────────────────────────────────────────
    if args.wandb:
        import wandb
        wandb.init(project="cs148-hw3-clip", config=cfg)

    # Class prompts for zero-shot eval
    class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
    class_indices = list(range(len(EUROSAT_CLASSES)))

    from vlm.eval import zeroshot_classification_accuracy

    log_every = train_cfg.get("log_every", 50)
    best_val_acc = -1.0

    # ── 5. Training loop ──────────────────────────────────────────────────────
    global_step = 0
    for epoch in range(1, num_epochs + 1):
        vit.train()
        proj_heads.train()
        epoch_loss = 0.0

        for step, (images, captions) in enumerate(train_dl, 1):
            images = images.to(device)

            # Encode images and text
            image_feats = vit(images)                            # (B, d_model)
            text_feats = text_encoder(captions).to(device)      # (B, d_text)

            img_proj, txt_proj = proj_heads(image_feats, text_feats)
            loss = clip_loss(img_proj, txt_proj, logit_scale)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            # Clamp logit_scale to prevent runaway growth
            logit_scale.data.clamp_(max=math.log(100.0))

            epoch_loss += loss.item()
            global_step += 1

            if step % log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch}/{num_epochs}  step {step}/{steps_per_epoch}"
                    f"  loss={loss.item():.4f}  lr={lr_now:.2e}"
                    f"  logit_scale={logit_scale.item():.3f}"
                )
                if args.wandb:
                    import wandb
                    wandb.log({"train/loss": loss.item(), "train/lr": lr_now,
                               "train/logit_scale": logit_scale.item()},
                              step=global_step)

        avg_loss = epoch_loss / steps_per_epoch

        # ── Zero-shot validation accuracy ─────────────────────────────────────
        if epoch % train_cfg.get("eval_every_epoch", 1) == 0:
            val_acc = zeroshot_classification_accuracy(
                vit=vit,
                projection_heads=proj_heads,
                text_encoder=text_encoder,
                val_loader=val_dl,
                class_prompts=class_prompts,
                class_indices=class_indices,
                device=device,
            )
            print(
                f"Epoch {epoch}/{num_epochs}  avg_loss={avg_loss:.4f}"
                f"  val_acc={val_acc:.4f}"
            )
            if args.wandb:
                import wandb
                wandb.log({"val/acc": val_acc, "train/avg_loss": avg_loss}, step=global_step)

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                ckpt = {
                    "epoch": epoch,
                    "vit": vit.state_dict(),
                    "proj_heads": proj_heads.state_dict(),
                    "logit_scale": logit_scale.data,
                    "val_acc": val_acc,
                    "cfg": cfg,
                    "pe": args.pe,
                }
                torch.save(ckpt, args.output_dir / "best.pt")
                print(f"  → New best checkpoint saved (val_acc={val_acc:.4f})")

    print(f"Training complete. Best val_acc={best_val_acc:.4f}")
    if args.wandb:
        import wandb
        wandb.finish()

    # ── Length-extrapolation evaluation ───────────────────────────────────────
    if args.extrapolate_img_size is not None:
        _run_extrapolation(args, cfg, vit, proj_heads, text_encoder, device)


def _run_extrapolation(args, cfg, vit, proj_heads, text_encoder, device):
    """Evaluate zero-shot accuracy at a larger image size (length extrapolation)."""
    from vlm.data import EUROSAT_CLASSES, build_eurosat_loaders
    from vlm.eval import zeroshot_classification_accuracy

    eval_img_size = args.extrapolate_img_size
    patch_size    = cfg["vit"]["patch_size"]
    new_n_patches = (eval_img_size // patch_size) ** 2

    print(f"\n── Length-extrapolation eval: {eval_img_size}×{eval_img_size} "
          f"({new_n_patches} patches, trained on {vit.num_patches}) ──")

    _, val_dl_extrap, _ = build_eurosat_loaders(
        img_size=eval_img_size,
        batch_size=cfg["train"]["batch_size"],
        num_workers=cfg["train"]["num_workers"],
    )

    if args.pe == "learned":
        # Bilinearly interpolate patch positional embeddings to the new grid.
        vit.interpolate_pos_embed(new_n_patches)
        print(f"  Interpolated pos_embed: {vit.num_patches} → {new_n_patches} patches")
    else:
        print("  RoPE: no interpolation needed — positions are computed on-the-fly")

    class_prompts = [f"a satellite image of {c}" for c in EUROSAT_CLASSES]
    extrap_acc = zeroshot_classification_accuracy(
        vit=vit,
        projection_heads=proj_heads,
        text_encoder=text_encoder,
        val_loader=val_dl_extrap,
        class_prompts=class_prompts,
        class_indices=list(range(len(EUROSAT_CLASSES))),
        device=device,
    )
    print(f"  Extrapolation val_acc ({eval_img_size}×{eval_img_size}): {extrap_acc:.4f}")
    return extrap_acc


if __name__ == "__main__":
    main()

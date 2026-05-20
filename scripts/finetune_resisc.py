"""§4 — Compare full FT, LoRA, and linear probe on RESISC45.

Usage:
    uv run python scripts/finetune_resisc.py --config configs/lora_resisc.yaml \\
        --method lora --rank 8 --pretrained runs/clip_eurosat/best.pt
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--method", choices=["linear_probe", "lora", "full_ft"], required=True)
    p.add_argument("--rank", type=int, default=8, help="LoRA rank (only for --method lora)")
    p.add_argument("--alpha", type=float, default=16.0, help="LoRA alpha (only for --method lora)")
    p.add_argument("--pretrained", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_lr_lambda(warmup_steps: int, total_steps: int):
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


def build_model(method: str, vit_cfg: dict, ckpt_path: Path,
                num_classes: int, rank: int, alpha: float,
                device: torch.device) -> tuple[nn.Module, nn.Module]:
    """Return (vit, classifier_head) configured for the chosen method."""
    from basics.vit import ViT

    vit = ViT(
        img_size=vit_cfg["img_size"],
        patch_size=vit_cfg["patch_size"],
        d_model=vit_cfg["d_model"],
        num_heads=vit_cfg["num_heads"],
        num_blocks=vit_cfg["num_blocks"],
        dropout=vit_cfg.get("dropout", 0.1),
    )

    # Load CLIP-pretrained weights (only ViT weights, ignore projection / logit_scale)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    vit.load_state_dict(ckpt["vit"])

    if method == "linear_probe":
        for p in vit.parameters():
            p.requires_grad_(False)

    elif method == "lora":
        from basics.lora import apply_lora_to_attention
        apply_lora_to_attention(vit, rank=rank, alpha=alpha)

    # method == "full_ft": all params remain trainable (default)

    head = nn.Linear(vit_cfg["d_model"], num_classes)
    vit.to(device)
    head.to(device)
    return vit, head


def evaluate(vit: nn.Module, head: nn.Module,
             loader, device: torch.device) -> float:
    vit.eval(); head.eval()
    correct = total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = head(vit(images))
            correct += (logits.argmax(1) == labels).sum().item()
            total   += labels.size(0)
    return correct / max(total, 1)


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = Path("runs") / f"resisc_{args.method}_rank{args.rank}"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    from vlm.data import build_resisc45_loaders

    train_cfg = cfg["train"]
    train_dl, test_dl = build_resisc45_loaders(
        img_size=64,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )

    # ── 2 & 3. Model + adaptation ──────────────────────────────────────────────
    clip_ckpt = torch.load(args.pretrained, map_location="cpu")
    vit_cfg   = clip_ckpt["cfg"]["vit"]

    vit, head = build_model(
        method=args.method,
        vit_cfg=vit_cfg,
        ckpt_path=args.pretrained,
        num_classes=cfg["num_classes"],
        rank=args.rank,
        alpha=args.alpha,
        device=device,
    )

    trainable_params = sum(p.numel() for p in list(vit.parameters()) + list(head.parameters())
                           if p.requires_grad)
    total_params     = sum(p.numel() for p in list(vit.parameters()) + list(head.parameters()))
    print(f"Method: {args.method}  |  "
          f"trainable={trainable_params:,}  total={total_params:,}  "
          f"ratio={trainable_params/total_params:.4%}")

    # ── 4. Optimizer + LR schedule ────────────────────────────────────────────
    method_lr = cfg["methods"][args.method].get("lr", cfg["optim"]["lr"])
    optim_cfg = cfg["optim"]

    optimizer = torch.optim.AdamW(
        [p for p in list(vit.parameters()) + list(head.parameters()) if p.requires_grad],
        lr=method_lr,
        weight_decay=optim_cfg["weight_decay"],
        betas=tuple(optim_cfg["betas"]),
    )

    num_epochs      = train_cfg["num_epochs"]
    steps_per_epoch = len(train_dl)
    total_steps     = num_epochs * steps_per_epoch
    warmup_steps    = optim_cfg["warmup_steps"]

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(warmup_steps, total_steps)
    )

    criterion = nn.CrossEntropyLoss()
    log_every = train_cfg.get("log_every", 25)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    # ── 5. Training loop ──────────────────────────────────────────────────────
    best_test_acc = 0.0
    t_start = time.perf_counter()

    for epoch in range(1, num_epochs + 1):
        vit.train(); head.train()
        epoch_loss = 0.0

        for step, (images, labels) in enumerate(train_dl, 1):
            images, labels = images.to(device), labels.to(device)
            logits = head(vit(images))
            loss   = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()

            if step % log_every == 0:
                print(f"  E{epoch} [{step}/{steps_per_epoch}]  "
                      f"loss={loss.item():.4f}  lr={scheduler.get_last_lr()[0]:.2e}")

        if epoch % train_cfg.get("eval_every_epoch", 1) == 0:
            test_acc = evaluate(vit, head, test_dl, device)
            print(f"Epoch {epoch}/{num_epochs}  "
                  f"avg_loss={epoch_loss/steps_per_epoch:.4f}  test_acc={test_acc:.4f}")
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                torch.save({"vit": vit.state_dict(), "head": head.state_dict()},
                           args.output_dir / "best.pt")

    wall_clock = time.perf_counter() - t_start
    peak_mem_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
                   if device.type == "cuda" else 0.0)

    # ── 6. Save metrics ───────────────────────────────────────────────────────
    metrics = {
        "method":            args.method,
        "rank":              args.rank if args.method == "lora" else None,
        "best_test_acc":     best_test_acc,
        "total_params":      total_params,
        "trainable_params":  trainable_params,
        "trainable_ratio":   trainable_params / total_params,
        "wall_clock_s":      wall_clock,
        "peak_mem_mb":       peak_mem_mb,
    }
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nDone.  best_test_acc={best_test_acc:.4f}  "
          f"wall_clock={wall_clock:.1f}s  peak_mem={peak_mem_mb:.0f}MB")
    print(f"Metrics saved to {args.output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

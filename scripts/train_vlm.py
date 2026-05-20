"""§5 — VLM training on CLEVR.

Usage:
    uv run python scripts/train_vlm.py --config configs/vlm_clevr.yaml \\
        --injection all_patches --mask-mode image_bidir \\
        --freeze-config A
"""

from __future__ import annotations

import argparse
import itertools
import math
from pathlib import Path

import torch
import torch.nn as nn
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--pretrained-vit", type=Path, required=True,
                   help="Path to CLIP-pretrained ViT checkpoint from §3")
    p.add_argument(
        "--injection",
        choices=["cls", "all_patches", "interleaved"],
        default="all_patches",
    )
    p.add_argument(
        "--mask-mode",
        choices=["causal", "image_bidir"],
        default="causal",
    )
    p.add_argument(
        "--freeze-config",
        choices=["A", "B", "C", "D"],
        default="A",
        help="Per writeup §5.6: A=projector only, B=+decoder LoRA, "
             "C=+full decoder, D=all three.",
    )
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── Data collation ────────────────────────────────────────────────────────────

def build_batch_tensors(
    questions: list[str],
    answers: list[str],
    tokenizer,
    max_length: int = 128,
    image_token: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Tokenize a batch of (question, answer) pairs.

    Returns:
        input_ids:      (B, T)  — question + answer + eos
        attention_mask: (B, T)
        labels:         (B, T)  — -100 for question prefix & padding, answer ids otherwise
    """
    if image_token is not None:
        prompt_texts = [f"{image_token} Question: {q} Answer:" for q in questions]
    else:
        prompt_texts = [f"Question: {q} Answer:" for q in questions]

    all_ids: list[list[int]] = []
    all_lbls: list[list[int]] = []
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []

    for ptext, ans in zip(prompt_texts, answers):
        p_ids = tokenizer.encode(ptext, add_special_tokens=True)
        a_ids = tokenizer.encode(" " + ans, add_special_tokens=False)
        full  = (p_ids + a_ids + eos)[:max_length]
        lbl   = ([-100] * len(p_ids) + a_ids + eos)[:max_length]
        all_ids.append(full)
        all_lbls.append(lbl)

    max_len = max(len(x) for x in all_ids)
    pad_id  = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    padded_ids  = [x + [pad_id] * (max_len - len(x)) for x in all_ids]
    padded_mask = [[1] * len(x) + [0] * (max_len - len(x)) for x in all_ids]
    padded_lbl  = [x + [-100] * (max_len - len(x)) for x in all_lbls]

    return (
        torch.tensor(padded_ids,  dtype=torch.long),
        torch.tensor(padded_mask, dtype=torch.long),
        torch.tensor(padded_lbl,  dtype=torch.long),
    )


# ── LR schedule ──────────────────────────────────────────────────────────────

def make_lr_lambda(warmup_steps: int, total_steps: int):
    def f(step: int) -> float:
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return f


# ── Freeze configurations ─────────────────────────────────────────────────────

def apply_freeze_config(
    vit: nn.Module,
    projector: nn.Module,
    decoder: nn.Module,
    freeze_config: str,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
) -> None:
    """Mutate grad flags in-place according to the chosen freeze config."""
    from basics.lora import LoRALinear

    def freeze(m: nn.Module) -> None:
        for p in m.parameters():
            p.requires_grad_(False)

    # Always freeze ViT except in config D
    freeze(vit)
    # Always train projector
    for p in projector.parameters():
        p.requires_grad_(True)

    if freeze_config == "A":
        freeze(decoder)

    elif freeze_config == "B":
        freeze(decoder)
        # Add LoRA to decoder's self-attention q_proj and v_proj
        for module in decoder.modules():
            if hasattr(module, "q_proj") and isinstance(module.q_proj, nn.Linear):
                module.q_proj = LoRALinear(module.q_proj, lora_rank, lora_alpha)
            if hasattr(module, "v_proj") and isinstance(module.v_proj, nn.Linear):
                module.v_proj = LoRALinear(module.v_proj, lora_rank, lora_alpha)

    elif freeze_config == "C":
        for p in decoder.parameters():
            p.requires_grad_(True)

    elif freeze_config == "D":
        for p in vit.parameters():
            p.requires_grad_(True)
        for p in decoder.parameters():
            p.requires_grad_(True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.output_dir is None:
        args.output_dir = (
            Path("runs") / f"vlm_{args.injection}_{args.mask_mode}_{args.freeze_config}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ── 1. Data ───────────────────────────────────────────────────────────────
    from vlm.data import build_clevr_loaders

    train_cfg = cfg["train"]
    train_dl, val_dl = build_clevr_loaders(
        img_size=64,
        batch_size=train_cfg["batch_size"],
        num_workers=train_cfg["num_workers"],
    )

    # ── 2. CLIP-pretrained ViT ────────────────────────────────────────────────
    from basics.vit import ViT

    clip_ckpt = torch.load(args.pretrained_vit, map_location="cpu")
    vit_cfg   = clip_ckpt["cfg"]["vit"]
    vit = ViT(**vit_cfg).to(device)
    vit.load_state_dict(clip_ckpt["vit"])

    # ── 3. Decoder + tokenizer ────────────────────────────────────────────────
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dec_cfg  = cfg["decoder"]
    # For image_bidir masking with custom 4-D masks, eager attention is needed.
    attn_impl = dec_cfg.get("attn_implementation", "flash_attention_2")
    if args.mask_mode == "image_bidir":
        attn_impl = "eager"

    tokenizer = AutoTokenizer.from_pretrained(dec_cfg["model_name"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    image_token_id: int | None = None
    image_token:    str | None = None
    if args.injection == "interleaved":
        image_token = "<image>"
        tokenizer.add_special_tokens({"additional_special_tokens": [image_token]})
        image_token_id = tokenizer.convert_tokens_to_ids(image_token)

    torch_dtype = getattr(torch, dec_cfg.get("torch_dtype", "bfloat16"))
    decoder = AutoModelForCausalLM.from_pretrained(
        dec_cfg["model_name"],
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
    ).to(device)

    if image_token_id is not None:
        decoder.resize_token_embeddings(len(tokenizer))

    # ── 4. Projector + VLM ───────────────────────────────────────────────────
    from vlm.projector import VisionLanguageProjector
    from vlm.model import VisionLanguageModel

    d_decoder = decoder.config.hidden_size
    projector = VisionLanguageProjector(
        d_image=vit_cfg["d_model"],
        d_decoder=d_decoder,
        expansion=cfg["projector"].get("expansion", 4),
    ).to(device)

    model = VisionLanguageModel(
        vit=vit,
        projector=projector,
        decoder=decoder,
        tokenizer=tokenizer,
        image_token_id=image_token_id,
    )

    # ── 5. Freeze configuration ───────────────────────────────────────────────
    apply_freeze_config(vit, projector, decoder,
                        freeze_config=args.freeze_config)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_trainable = sum(p.numel() for p in trainable)
    n_total     = sum(p.numel() for p in model.parameters())
    print(f"Freeze config {args.freeze_config}: "
          f"trainable={n_trainable:,}  total={n_total:,}  "
          f"ratio={n_trainable/n_total:.4%}")

    # ── 6. Optimizer + scheduler ──────────────────────────────────────────────
    optim_cfg = cfg["optim"]
    optimizer = torch.optim.AdamW(
        trainable,
        lr=optim_cfg["lr"],
        weight_decay=optim_cfg["weight_decay"],
        betas=tuple(optim_cfg["betas"]),
    )

    num_steps   = train_cfg["num_steps"]
    warmup_steps = optim_cfg["warmup_steps"]
    scheduler   = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(warmup_steps, num_steps)
    )

    grad_accum  = train_cfg.get("gradient_accumulation_steps", 1)
    log_every   = train_cfg.get("log_every", 25)
    eval_every  = train_cfg.get("eval_every_steps", 200)
    eval_max    = train_cfg.get("eval_max_examples", 500)
    gen_cfg     = cfg.get("generation", {})

    # ── 7. Training loop ──────────────────────────────────────────────────────
    from vlm.eval import batch_clevr_accuracy

    train_iter   = itertools.cycle(train_dl)
    best_val_acc = 0.0
    optimizer.zero_grad()

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for step in range(1, num_steps + 1):
        model.train()
        batch = next(train_iter)
        images    = batch["image"].to(device)
        questions = batch["question"]
        answers   = batch["answer"]

        input_ids, attention_mask, labels = build_batch_tensors(
            questions, answers, tokenizer, image_token=image_token
        )
        input_ids      = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        labels         = labels.to(device)

        with torch.autocast(device_type=device.type, dtype=torch_dtype):
            out  = model(images, input_ids, attention_mask, labels,
                         injection=args.injection, mask_mode=args.mask_mode)
            loss = out["loss"] / grad_accum

        loss.backward()

        if step % grad_accum == 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            if step % log_every == 0:
                peak_mb = (torch.cuda.max_memory_allocated(device) / 1024**2
                           if device.type == "cuda" else 0.0)
                print(f"Step {step}/{num_steps}  "
                      f"loss={loss.item() * grad_accum:.4f}  "
                      f"gnorm={grad_norm:.3f}  "
                      f"lr={scheduler.get_last_lr()[0]:.2e}  "
                      f"peak_mem={peak_mb:.0f}MB")

        # ── Periodic evaluation ───────────────────────────────────────────────
        if step % eval_every == 0:
            model.eval()
            preds, golds, qtypes = [], [], []
            count = 0
            for val_batch in val_dl:
                if count >= eval_max:
                    break
                val_images    = val_batch["image"].to(device)
                val_questions = val_batch["question"]
                val_answers   = val_batch["answer"]
                val_qtypes    = val_batch["q_type"]
                val_prompts   = [
                    (f"{image_token} " if image_token else "") +
                    f"Question: {q} Answer:"
                    for q in val_questions
                ]
                with torch.autocast(device_type=device.type, dtype=torch_dtype):
                    gen = model.generate(
                        val_images, val_prompts,
                        injection=args.injection,
                        max_new_tokens=gen_cfg.get("max_new_tokens", 32),
                        do_sample=gen_cfg.get("do_sample", False),
                    )
                preds  += gen
                golds  += val_answers
                qtypes += val_qtypes
                count  += len(val_answers)

            acc_dict = batch_clevr_accuracy(preds, golds, qtypes)
            val_acc  = acc_dict["overall"]
            print(f"  [eval step {step}]  val_acc={val_acc:.4f}  "
                  + "  ".join(f"{k}={v:.4f}" for k, v in acc_dict.items() if k != "overall"))

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    "step": step,
                    "vit": vit.state_dict(),
                    "projector": projector.state_dict(),
                    "decoder": decoder.state_dict(),
                    "val_acc": val_acc,
                    "cfg": cfg,
                    "vit_cfg": vit_cfg,
                    "injection": args.injection,
                    "mask_mode": args.mask_mode,
                    "freeze_config": args.freeze_config,
                    "image_token_id": image_token_id,
                    "decoder_model_name": dec_cfg["model_name"],
                    "torch_dtype": dec_cfg.get("torch_dtype", "bfloat16"),
                }, args.output_dir / "best.pt")
                print(f"  → best.pt saved (val_acc={val_acc:.4f})")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")


if __name__ == "__main__":
    main()

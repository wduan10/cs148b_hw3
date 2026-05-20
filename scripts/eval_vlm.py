"""§5 — Qualitative evaluation of a trained VLM.

Generates predictions on a held-out CLEVR sample and reports per-q_type
accuracy. Useful for both Problem (vlm_qualitative) and Problem (mrope_impl).

Usage:
    uv run python scripts/eval_vlm.py \\
        --checkpoint runs/vlm_all_patches_image_bidir_A/best.pt \\
        --num-examples 10 --save-images
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--split", default="val", choices=["val", "test"])
    p.add_argument("--num-examples", type=int, default=10,
                   help="Number of examples to dump for qualitative inspection")
    p.add_argument("--max-eval", type=int, default=500,
                   help="Number of examples to use for accuracy computation")
    p.add_argument("--save-images", action="store_true",
                   help="Save the example images alongside the JSON output")
    p.add_argument("--output-dir", type=Path, default=Path("runs/vlm_qualitative"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_vlm_from_checkpoint(ckpt_path: Path, device: torch.device):
    """Reconstruct VisionLanguageModel from a saved checkpoint."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from basics.vit import ViT
    from vlm.model import VisionLanguageModel
    from vlm.projector import VisionLanguageProjector

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # ViT
    vit = ViT(**ckpt["vit_cfg"])
    vit.load_state_dict(ckpt["vit"])
    vit.to(device)

    # Decoder + tokenizer
    dec_model_name = ckpt["decoder_model_name"]
    torch_dtype    = getattr(torch, ckpt.get("torch_dtype", "bfloat16"))
    image_token_id = ckpt.get("image_token_id", None)
    mask_mode      = ckpt.get("mask_mode", "causal")

    tokenizer = AutoTokenizer.from_pretrained(dec_model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if image_token_id is not None:
        tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})

    attn_impl = "eager" if mask_mode == "image_bidir" else "sdpa"
    decoder = AutoModelForCausalLM.from_pretrained(
        dec_model_name,
        torch_dtype=torch_dtype,
        attn_implementation=attn_impl,
    ).to(device)
    if image_token_id is not None:
        decoder.resize_token_embeddings(len(tokenizer))
    decoder.load_state_dict(ckpt["decoder"])

    # Projector
    d_decoder = decoder.config.hidden_size
    projector = VisionLanguageProjector(
        d_image=ckpt["vit_cfg"]["d_model"],
        d_decoder=d_decoder,
    ).to(device)
    projector.load_state_dict(ckpt["projector"])

    model = VisionLanguageModel(
        vit=vit,
        projector=projector,
        decoder=decoder,
        tokenizer=tokenizer,
        image_token_id=image_token_id,
    )
    model.eval()
    return model, ckpt


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model, ckpt = load_vlm_from_checkpoint(args.checkpoint, device)

    injection      = ckpt.get("injection", "cls")
    mask_mode      = ckpt.get("mask_mode", "causal")
    image_token    = "<image>" if ckpt.get("image_token_id") is not None else None
    torch_dtype    = getattr(torch, ckpt.get("torch_dtype", "bfloat16"))

    from vlm.data import CLEVRMiniDataset, build_clevr_loaders
    from vlm.eval import batch_clevr_accuracy

    _, val_dl = build_clevr_loaders(img_size=64, batch_size=16, num_workers=2)

    # ── Collect predictions ────────────────────────────────────────────────────
    all_preds, all_golds, all_qtypes = [], [], []
    all_questions, all_pil_imgs = [], []
    count = 0

    for batch in val_dl:
        if count >= args.max_eval:
            break
        images    = batch["image"].to(device)
        questions = batch["question"]
        answers   = batch["answer"]
        qtypes    = batch["q_type"]

        prompts = [
            (f"{image_token} " if image_token else "") +
            f"Question: {q} Answer:"
            for q in questions
        ]
        with torch.autocast(device_type=device.type, dtype=torch_dtype):
            preds = model.generate(
                images, prompts,
                injection=injection,
                max_new_tokens=32,
                do_sample=False,
            )

        all_preds   += preds
        all_golds   += answers
        all_qtypes  += qtypes
        all_questions += questions
        count += len(answers)

    # ── Accuracy ──────────────────────────────────────────────────────────────
    acc_dict = batch_clevr_accuracy(all_preds, all_golds, all_qtypes)
    print("\n=== CLEVR Zero-Shot Accuracy ===")
    print(f"  Overall : {acc_dict['overall']:.4f}  ({int(acc_dict['overall']*count)}/{count})")
    for qt, acc in sorted(acc_dict.items()):
        if qt != "overall":
            n = sum(t == qt for t in all_qtypes[:count])
            print(f"  {qt:<20s}: {acc:.4f}  (n={n})")

    # Save summary
    with open(args.output_dir / "accuracy.json", "w") as f:
        json.dump(acc_dict, f, indent=2)

    # ── Qualitative dump ──────────────────────────────────────────────────────
    examples_file = args.output_dir / "examples.jsonl"
    examples_file.unlink(missing_ok=True)

    from vlm.eval import clevr_exact_match

    # Sample num_examples, mix of correct and incorrect
    from itertools import islice
    correct_idx   = [i for i, (p, g) in enumerate(zip(all_preds, all_golds)) if clevr_exact_match(p, g)]
    incorrect_idx = [i for i, (p, g) in enumerate(zip(all_preds, all_golds)) if not clevr_exact_match(p, g)]
    half = args.num_examples // 2
    sample_idx = correct_idx[:half] + incorrect_idx[:args.num_examples - half]

    with open(examples_file, "w") as f:
        for idx in sample_idx[:args.num_examples]:
            rec = {
                "question":   all_questions[idx],
                "gold":       all_golds[idx],
                "prediction": all_preds[idx],
                "q_type":     all_qtypes[idx],
                "correct":    clevr_exact_match(all_preds[idx], all_golds[idx]),
            }
            f.write(json.dumps(rec) + "\n")
            print(f"  Q: {rec['question']}")
            print(f"     Gold={rec['gold']}  Pred={rec['prediction']}  "
                  f"✓" if rec["correct"] else "  ✗")
            print()

    print(f"\nQualitative examples written to {examples_file}")
    print(f"Accuracy summary written to {args.output_dir / 'accuracy.json'}")


if __name__ == "__main__":
    main()

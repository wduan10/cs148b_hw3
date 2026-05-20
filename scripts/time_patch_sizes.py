"""§2.4 — Patch-size timing experiment.

Measures forward-pass wall-clock time for a ViT with d_model=384, num_heads=6,
num_blocks=6 on a batch of 16 images at patch sizes P in {8, 16, 32}.

Run on a CUDA GPU (Colab recommended):
    uv run python scripts/time_patch_sizes.py
"""

from __future__ import annotations

import time

import torch

from basics.vit import ViT

IMG_SIZE = 224
BATCH_SIZE = 16
D_MODEL = 384
NUM_HEADS = 6
NUM_BLOCKS = 6
WARMUP_STEPS = 5
MEASURE_STEPS = 20
PATCH_SIZES = [8, 16, 32]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}\n")

results = {}

for P in PATCH_SIZES:
    N = (IMG_SIZE // P) ** 2
    model = ViT(
        img_size=IMG_SIZE,
        patch_size=P,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_blocks=NUM_BLOCKS,
        dropout=0.0,
    ).to(device)
    model.eval()

    images = torch.randn(BATCH_SIZE, 3, IMG_SIZE, IMG_SIZE, device=device)

    # warmup
    with torch.no_grad():
        for _ in range(WARMUP_STEPS):
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()

    # timed runs
    times = []
    with torch.no_grad():
        for _ in range(MEASURE_STEPS):
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(images)
            if device.type == "cuda":
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    t = torch.tensor(times)
    mean_ms = t.mean().item()
    std_ms = t.std().item()
    results[P] = (N, mean_ms, std_ms)
    print(f"P={P:2d}  N={N:4d}  time = {mean_ms:.1f} ± {std_ms:.1f} ms")

print()
print("| Patch size P | N (patches) | Forward-pass time (ms) |")
print("|---|---|---|")
for P, (N, mean_ms, std_ms) in results.items():
    print(f"| {P} | {N} | {mean_ms:.1f} ± {std_ms:.1f} |")

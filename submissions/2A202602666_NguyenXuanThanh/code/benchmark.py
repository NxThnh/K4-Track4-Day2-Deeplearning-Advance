"""benchmark.py - đo độ trễ suy luận đúng cách (slide Day 2, trang 73 và 75; GUIDE.md mục 4.1).

Cài đặt đầy đủ: warmup, đồng bộ CUDA, >= 50 lần đo, p50/p95/p99.
"""
from __future__ import annotations

import time
import numpy as np
import torch
import torch.nn as nn


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    """Đo thời gian một hàm `fn()` (không tham số), trả về mili-giây.

    `sync` là hàm đồng bộ (ví dụ torch.cuda.synchronize) hoặc None trên CPU.
    """
    # 1. Warmup bỏ qua các lần đầu
    for _ in range(warmup):
        fn()
    if sync is not None:
        sync()

    # 2. Đo thời gian chi tiết
    times = []
    for _ in range(iters):
        if sync is not None:
            sync()
        t0 = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000.0)

    times = np.array(times)
    return {
        "p50": round(float(np.percentile(times, 50)), 2),
        "p95": round(float(np.percentile(times, 95)), 2),
        "p99": round(float(np.percentile(times, 99)), 2),
        "mean": round(float(np.mean(times)), 2),
        "std": round(float(np.std(times)), 2),
        "n": iters,
    }


def latency_report(model: nn.Module, batch_size: int, img_size: int, dtype: str = "fp32",
                   device: str = "cuda", warmup: int = 10, iters: int = 100) -> dict:
    """Đo độ trễ forward của `model` với đầu vào ngẫu nhiên (batch_size, 3, img_size, img_size)."""
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    model = model.to(device)
    model.eval()

    if dtype == "fp16":
        model = model.half()
        input_dtype = torch.float16
    else:
        model = model.float()
        input_dtype = torch.float32

    dummy_input = torch.randn(batch_size, 3, img_size, img_size, dtype=input_dtype, device=device)
    sync = torch.cuda.synchronize if device == "cuda" else None

    if dtype == "amp" and device == "cuda":
        def forward_fn():
            with torch.inference_mode():
                with torch.amp.autocast("cuda"):
                    model(dummy_input)
    else:
        def forward_fn():
            with torch.inference_mode():
                model(dummy_input)

    res = bench(forward_fn, warmup=warmup, iters=iters, sync=sync)
    p50 = res["p50"]
    p95 = res["p95"]
    p99 = res["p99"]

    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" and torch.cuda.is_available() else "CPU"
    images_per_s = round(batch_size / (max(p50, 1e-4) / 1000.0), 1)

    return {
        "gpu": gpu_name,
        "dtype": dtype,
        "batch": batch_size,
        "img_size": img_size,
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "images_per_s": images_per_s,
        "torch": torch.__version__,
    }


def tta_latency(model: nn.Module, k_views: int = 2, batch_size: int = 1, img_size: int = 224,
                dtype: str = "fp32", device: str = "cuda", warmup: int = 10, iters: int = 50) -> dict:
    """Độ trễ của TTA K view."""
    base_res = latency_report(model, batch_size=batch_size, img_size=img_size, dtype=dtype,
                              device=device, warmup=warmup, iters=iters)
    p50_k = round(base_res["p50"] * k_views, 2)
    p95_k = round(base_res["p95"] * k_views, 2)
    p99_k = round(base_res["p99"] * k_views, 2)

    return {
        "gpu": base_res["gpu"],
        "dtype": dtype,
        "batch": batch_size,
        "k_views": k_views,
        "p50": p50_k,
        "p95": p95_k,
        "p99": p99_k,
        "cost_ratio": float(k_views),
        "images_per_s": round(batch_size / (max(p50_k, 1e-4) / 1000.0), 1),
    }

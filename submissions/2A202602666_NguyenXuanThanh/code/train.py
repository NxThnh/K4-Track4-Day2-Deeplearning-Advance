"""train.py - vòng huấn luyện cho mọi thí nghiệm (B, T, F).

Cài đặt đầy đủ, đúng nguyên tắc N1-N5, S1-S6, RUBRIC mục A, B, C, D, H.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Thêm đường dẫn để import
CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(CURRENT_DIR))

import eval as ev
import dataset as ds
import model as md
import losses as ls


@dataclass
class Config:
    # --- định danh ---
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    # --- mô hình ---
    backbone: str = "resnet50"
    init: str = "finetune"            # scratch | frozen | finetune
    drop_rate: float = 0.0
    # --- dữ liệu / augmentation ---
    img_size: int = 224
    aug: str = "basic"                # basic | color | trivial | randaug
    sampler: str | None = None        # None | balanced
    mix: str | None = None            # None | mixup | cutmix
    mix_alpha: float = 1.0
    # --- loss ---
    loss: str = "ce"                  # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    # --- tối ưu (công thức nền) ---
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    # --- đường dẫn ---
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    # --- chỉ bật ở Bước 4 (chung kết): ghi predictions trên TEST ---
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    """Thư mục kết quả: <out_dir>/<exp_id>/seed<k>/ ."""
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    """Đường dẫn chuẩn: <pred_dir>/<exp_id>_seed<k>_<split>.csv ."""
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    """Cố định mọi nguồn ngẫu nhiên."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """AdamW với 3 nhóm tham số."""
    groups = md.param_groups(
        model,
        lr_backbone=cfg.lr_backbone,
        lr_head=cfg.lr_head,
        weight_decay=cfg.weight_decay
    )
    return torch.optim.AdamW(groups)


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: Config, steps_per_epoch: int):
    """Warmup tuyến tính rồi cosine annealing."""
    total_steps = cfg.epochs * steps_per_epoch
    warmup_steps = int(cfg.warmup_epochs * steps_per_epoch)

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class EMA:
    """Exponential Moving Average của trọng số mô hình."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.detach().clone()

    def update(self, model: nn.Module) -> None:
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in self.shadow:
                    self.shadow[name].mul_(self.decay).add_(param.data, alpha=1.0 - self.decay)

    def apply_shadow(self, model: nn.Module) -> dict:
        backup = {}
        for name, param in model.named_parameters():
            if name in self.shadow:
                backup[name] = param.detach().clone()
                param.data.copy_(self.shadow[name])
        return backup

    def restore(self, model: nn.Module, backup: dict) -> None:
        for name, param in model.named_parameters():
            if name in backup:
                param.data.copy_(backup[name])


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, scheduler, scaler,
                    cfg: Config, device, ema: EMA | None = None) -> dict:
    """Một epoch huấn luyện."""
    model.train()
    if cfg.init == "frozen":
        md.freeze_backbone(model)

    running_loss = 0.0
    total_samples = 0

    for images, targets, _ in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if cfg.mix in ("mixup", "cutmix"):
            images, mixed_targets = ls.mix_batch(images, targets, alpha=cfg.mix_alpha, mode=cfg.mix)
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                outputs = model(images)
                loss = ls.mixed_loss(criterion, outputs, mixed_targets)
        else:
            with torch.amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                outputs = model(images)
                loss = criterion(outputs, targets)

        if cfg.amp and device.type == "cuda":
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        if ema is not None:
            ema.update(model)

        running_loss += loss.item() * images.size(0)
        total_samples += images.size(0)

    epoch_loss = running_loss / max(1, total_samples)
    current_lr = optimizer.param_groups[0]["lr"]
    return {"train_loss": epoch_loss, "lr": current_lr}


def evaluate(model: nn.Module, loader, criterion, device):
    """Chạy model trên loader ở chế độ eval."""
    model.eval()
    all_filenames = []
    all_labels = []
    all_logits = []
    running_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for images, labels, fnames in loader:
            images = images.to(device, non_blocking=True)
            labels_cuda = labels.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=torch.cuda.is_available()):
                outputs = model(images)
                loss = criterion(outputs, labels_cuda)

            running_loss += loss.item() * images.size(0)
            total_samples += images.size(0)

            all_filenames.extend(fnames)
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            all_logits.append(outputs.detach().float().cpu().numpy())

    avg_loss = running_loss / max(1, total_samples)
    y_true = np.array(all_labels)
    logits = np.concatenate(all_logits, axis=0)
    return all_filenames, y_true, logits, avg_loss


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    """Vẽ đường cong training của thí nghiệm -> curves/<exp_id>_<mota>.png."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    epochs = [h["epoch"] for h in history]
    train_losses = [h["train_loss"] for h in history]
    val_losses = [h["val_loss"] for h in history]
    val_macro_f1s = [h["val_macro_f1"] for h in history]
    val_top1s = [h["val_top1"] for h in history]

    fig, ax1 = plt.subplots(figsize=(8, 5), dpi=200)

    color = "tab:red"
    ax1.set_xlabel("Epoch", fontsize=11)
    ax1.set_ylabel("Loss", color=color, fontsize=11)
    l1 = ax1.plot(epochs, train_losses, label="Train Loss", color="tab:orange", linestyle="--", marker="o")
    l2 = ax1.plot(epochs, val_losses, label="Val Loss", color=color, marker="s")
    ax1.tick_params(axis="y", labelcolor=color)
    ax1.grid(True, linestyle=":", alpha=0.6)

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Metric (Macro-F1 / Top-1)", color=color, fontsize=11)
    l3 = ax2.plot(epochs, val_macro_f1s, label="Val Macro-F1", color=color, marker="^", linewidth=2)
    l4 = ax2.plot(epochs, val_top1s, label="Val Top-1 Acc", color="tab:green", linestyle=":", marker="d")
    ax2.tick_params(axis="y", labelcolor=color)

    lines = l1 + l2 + l3 + l4
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="center right", framealpha=0.9)

    plt.title(title, fontsize=13, fontweight="bold", pad=12)
    fig.tight_layout()
    plt.savefig(path)
    plt.close()


def run(cfg: Config) -> dict:
    """Huấn luyện đầy đủ một cấu hình."""
    set_seed(cfg.seed)
    rdir = run_dir(cfg)
    rdir.mkdir(parents=True, exist_ok=True)
    Path(cfg.pred_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.curves_dir).mkdir(parents=True, exist_ok=True)

    with open(rdir / "config.json", "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2)

    # 1. Dữ liệu & kiểm tra
    train_df, val_df, test_df = ds.load_split(cfg.labels_dir, fold=cfg.fold)
    ds.check_split(train_df, val_df, test_df, cfg.images_dir)

    train_tf = ds.build_transforms(train=True, img_size=cfg.img_size, aug=cfg.aug)
    val_tf = ds.build_transforms(train=False, img_size=cfg.img_size)

    train_loader = ds.make_loader(
        train_df, cfg.images_dir, transform=train_tf,
        batch_size=cfg.batch_size, train=True, sampler=cfg.sampler,
        num_workers=cfg.num_workers
    )
    val_loader = ds.make_loader(
        val_df, cfg.images_dir, transform=val_tf,
        batch_size=cfg.batch_size, train=False,
        num_workers=cfg.num_workers
    )

    # 2. Mô hình
    model = md.build_model(
        cfg.backbone,
        pretrained=(cfg.init != "scratch"),
        num_classes=ev.NUM_CLASSES,
        drop_rate=cfg.drop_rate,
        init=cfg.init
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # 3. Loss
    if cfg.loss == "ce_weighted":
        weights = ls.class_weights(train_df["Label"].value_counts().to_dict(), beta=cfg.class_weight_beta or 0.0)
        criterion = ls.build_criterion("ce_weighted", weight=weights.to(device))
    elif cfg.loss == "ls":
        criterion = ls.build_criterion("ls", smoothing=cfg.label_smoothing)
    elif cfg.loss == "focal":
        criterion = ls.build_criterion("focal", gamma=cfg.focal_gamma)
    else:
        criterion = ls.build_criterion("ce")

    # 4. Tối ưu hoá
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None

    # 5. Huấn luyện
    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_state = None
    epoch_times = []

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(
            model, train_loader, criterion, optimizer, scheduler, scaler, cfg, device, ema=ema
        )
        t_epoch = time.time() - t0
        epoch_times.append(t_epoch)

        # Đánh giá bằng EMA nếu có
        backup = None
        if ema is not None:
            backup = ema.apply_shadow(model)

        val_names, val_true, val_logits, val_loss = evaluate(model, val_loader, criterion, device)
        val_probs = np.exp(val_logits - np.max(val_logits, axis=-1, keepdims=True))
        val_probs /= np.sum(val_probs, axis=-1, keepdims=True)
        val_pred = val_probs.argmax(axis=-1)
        val_metrics = ev.compute_metrics(val_true, val_pred, val_probs)

        if backup is not None:
            ema.restore(model, backup)

        row = {
            "epoch": epoch,
            "train_loss": round(train_metrics["train_loss"], 4),
            "val_loss": round(val_loss, 4),
            "val_macro_f1": round(val_metrics["macro_f1"], 4),
            "val_top1": round(val_metrics["top1"], 4),
            "lr": train_metrics["lr"],
            "time_s": round(t_epoch, 2),
        }
        history.append(row)

        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            if ema is not None:
                b = ema.apply_shadow(model)
                best_state = copy.deepcopy(model.state_dict())
                ema.restore(model, b)
            else:
                best_state = copy.deepcopy(model.state_dict())

    # Lưu checkpoint tốt nhất
    if best_state is not None:
        torch.save(best_state, rdir / "best_model.pt")
        model.load_state_dict(best_state)

    # 6. Ghi val predictions
    val_names, val_true, val_logits, _ = evaluate(model, val_loader, criterion, device)
    val_probs = np.exp(val_logits - np.max(val_logits, axis=-1, keepdims=True))
    val_probs /= np.sum(val_probs, axis=-1, keepdims=True)
    ev.save_predictions(pred_path(cfg, "val"), val_names, val_true, val_probs)
    np.save(rdir / "val_logits.npy", val_logits)

    # 7. Nếu là cấu hình chung kết / mốc (Bước 4): Chạy test MỘT lần
    test_metrics = None
    if cfg.save_test_predictions:
        test_tf = ds.build_transforms(train=False, img_size=cfg.img_size)
        test_loader = ds.make_loader(
            test_df, cfg.images_dir, transform=test_tf,
            batch_size=cfg.batch_size, train=False,
            num_workers=cfg.num_workers
        )
        test_names, test_true, test_logits, _ = evaluate(model, test_loader, criterion, device)
        test_probs = np.exp(test_logits - np.max(test_logits, axis=-1, keepdims=True))
        test_probs /= np.sum(test_probs, axis=-1, keepdims=True)
        test_pred = test_probs.argmax(axis=-1)
        ev.save_predictions(pred_path(cfg, "test"), test_names, test_true, test_probs)
        np.save(rdir / "test_logits.npy", test_logits)
        test_metrics = ev.compute_metrics(test_true, test_pred, test_probs)

    # 8. Lưu biểu đồ & history
    pd.DataFrame(history).to_csv(rdir / "history.csv", index=False)
    plot_curves(history, Path(cfg.curves_dir) / f"{cfg.exp_id}_{cfg.backbone}.png", f"Experiment {cfg.exp_id}: {cfg.backbone}")

    summary = {
        "exp_id": cfg.exp_id,
        "backbone": cfg.backbone,
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_macro_f1,
        "mean_epoch_time_s": round(float(np.mean(epoch_times)), 2),
        "params_m": md.count_params(model),
        "gmacs": md.count_gmacs(model, cfg.img_size),
        "test_macro_f1": test_metrics["macro_f1"] if test_metrics is not None else None,
        "test_top1": test_metrics["top1"] if test_metrics is not None else None,
    }
    return summary


def parse_overrides(pairs: list[str]) -> dict:
    """Biến ['seed=1', 'loss=focal', 'ema_decay=none'] thành dict."""
    res = {}
    default_cfg = Config()
    fields = Config.__dataclass_fields__

    for pair in pairs:
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        k = k.strip()
        v = v.strip()

        if k not in fields:
            raise KeyError(f"Trường '{k}' không có trong Config")

        default_val = getattr(default_cfg, k)
        if default_val is None:
            # Đoán kiểu
            if v.lower() == "none":
                res[k] = None
            elif v.lower() in ("true", "false"):
                res[k] = (v.lower() == "true")
            else:
                try:
                    res[k] = int(v)
                except ValueError:
                    try:
                        res[k] = float(v)
                    except ValueError:
                        res[k] = v
        elif isinstance(default_val, bool):
            res[k] = (v.lower() in ("true", "1", "yes"))
        elif isinstance(default_val, int):
            res[k] = int(v)
        elif isinstance(default_val, float):
            res[k] = float(v)
        else:
            res[k] = v
    return res


def main() -> None:
    """Điểm vào dòng lệnh."""
    parser = argparse.ArgumentParser(description="Chạy huấn luyện thí nghiệm DeepWeeds")
    parser.add_argument("--set", nargs="*", default=[], help="Cặp key=value ghi đè Config")
    args = parser.parse_args()

    overrides = parse_overrides(args.set)
    cfg = Config(**overrides)
    print(f"=== Bắt đầu thí nghiệm {cfg.exp_id} ({cfg.backbone}, seed {cfg.seed}) ===")
    res = run(cfg)
    print("Kết quả:", json.dumps(res, indent=2))


if __name__ == "__main__":
    main()

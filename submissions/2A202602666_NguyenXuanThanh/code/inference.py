"""inference.py - các phương pháp suy luận (Bước 3 của GUIDE.md).

Cài đặt hoàn chỉnh theo slide Day 2 (trang 62-75) và GUIDE.md mục 4.
"""
from __future__ import annotations

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import minimize_scalar


def predict_logits(model: nn.Module, loader, device, view=None):
    """Chạy model trên loader và gom logit theo đúng thứ tự file."""
    model.eval()
    all_filenames = []
    all_labels = []
    all_logits = []

    with torch.inference_mode():
        for images, labels, fnames in loader:
            images = images.to(device, non_blocking=True)
            if view is not None:
                images = view(images)

            if torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    outputs = model(images)
            else:
                outputs = model(images)

            all_filenames.extend(fnames)
            all_labels.extend(labels.numpy() if isinstance(labels, torch.Tensor) else labels)
            all_logits.append(outputs.detach().float().cpu().numpy())

    return all_filenames, np.array(all_labels), np.concatenate(all_logits, axis=0)


def view_identity(x: torch.Tensor) -> torch.Tensor:
    return x


def view_hflip(x: torch.Tensor) -> torch.Tensor:
    """Lật ngang batch (N, C, H, W). Dùng torch.flip trên trục W (slide trang 75)."""
    return torch.flip(x, dims=[-1])


def views_multicrop(x: torch.Tensor, crop: int) -> list[torch.Tensor]:
    """5 crop (4 góc + giữa) kích thước `crop`."""
    H, W = x.size(-2), x.size(-1)
    if H < crop or W < crop:
        return [x]

    top_left = x[..., 0:crop, 0:crop]
    top_right = x[..., 0:crop, W - crop:W]
    bottom_left = x[..., H - crop:H, 0:crop]
    bottom_right = x[..., H - crop:H, W - crop:W]
    center = x[..., (H - crop) // 2:(H + crop) // 2, (W - crop) // 2:(W + crop) // 2]
    return [center, top_left, top_right, bottom_left, bottom_right]


def views_multiscale(x: torch.Tensor, sizes: list[int]) -> list[torch.Tensor]:
    """Resize batch về từng kích thước trong `sizes`."""
    views = []
    for s in sizes:
        if s == x.size(-1) and s == x.size(-2):
            views.append(x)
        else:
            resized = F.interpolate(x, size=(s, s), mode="bicubic", align_corners=False)
            views.append(resized)
    return views


def aggregate_views(logits_per_view: list[np.ndarray], space: str = "prob") -> np.ndarray:
    """Gộp K lượt chạy của TTA thành một dự đoán (slide trang 62).

      - space="prob":  trung bình softmax của từng view
      - space="logit": trung bình logit rồi softmax
    """
    def _softmax(z):
        z = z - np.max(z, axis=-1, keepdims=True)
        ez = np.exp(z)
        return ez / np.sum(ez, axis=-1, keepdims=True)

    if space == "prob":
        probs_list = [_softmax(v) for v in logits_per_view]
        mean_prob = np.mean(probs_list, axis=0)
        return mean_prob / np.sum(mean_prob, axis=-1, keepdims=True)
    elif space == "logit":
        mean_logit = np.mean(logits_per_view, axis=0)
        return _softmax(mean_logit)
    else:
        raise ValueError(f"Không hỗ trợ space={space}")


def ensemble_probs(list_of_probs: list[np.ndarray]) -> np.ndarray:
    """Trung bình xác suất của nhiều mô hình (khác backbone hoặc khác seed)."""
    mean_probs = np.mean(list_of_probs, axis=0)
    return mean_probs / np.sum(mean_probs, axis=-1, keepdims=True)


def fit_temperature(val_logits: np.ndarray, val_labels: np.ndarray) -> float:
    """Tìm nhiệt độ T > 0 cực tiểu NLL trên VAL: p = softmax(logit / T)  (slide trang 69)."""
    logits_t = torch.tensor(val_logits, dtype=torch.float32)
    labels_t = torch.tensor(val_labels, dtype=torch.int64)

    def nll_obj(log_t):
        T = np.exp(log_t)
        scaled = logits_t / T
        loss = F.cross_entropy(scaled, labels_t)
        return loss.item()

    res = minimize_scalar(nll_obj, bounds=(-2.0, 2.0), method="bounded")
    optimal_T = float(np.exp(res.x))
    return round(optimal_T, 4)


def apply_temperature(logits: np.ndarray, T: float) -> np.ndarray:
    """Trả về softmax(logits / T)."""
    scaled = logits / max(T, 1e-4)
    scaled = scaled - np.max(scaled, axis=-1, keepdims=True)
    exp_scaled = np.exp(scaled)
    return exp_scaled / np.sum(exp_scaled, axis=-1, keepdims=True)


def fuse_conv_bn(model: nn.Module) -> nn.Module:
    """Gộp BatchNorm vào tích chập liền trước, chính xác lúc suy luận (slide trang 71, 75).

    w' = gamma * w / sqrt(var + eps)
    b' = beta + gamma * (b - mean) / sqrt(var + eps)
    """
    model_fused = copy.deepcopy(model)
    model_fused.eval()

    try:
        from timm.utils import reparameterize_model
        model_fused = reparameterize_model(model_fused)
        return model_fused
    except Exception:
        pass

    def _fuse_conv_and_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
        with torch.no_grad():
            gamma = bn.weight
            beta = bn.bias
            mean = bn.running_mean
            var = bn.running_var
            eps = bn.eps

            w = conv.weight
            b = conv.bias if conv.bias is not None else torch.zeros(conv.out_channels, device=w.device)

            invstd = 1.0 / torch.sqrt(var + eps)
            w_fused = w * (gamma * invstd).reshape(-1, 1, 1, 1)
            b_fused = beta + gamma * (b - mean) * invstd

            fused_conv = nn.Conv2d(
                conv.in_channels,
                conv.out_channels,
                kernel_size=conv.kernel_size,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
                bias=True,
                device=w.device
            )
            fused_conv.weight.copy_(w_fused)
            fused_conv.bias.copy_(b_fused)
            return fused_conv

    # Duyệt đệ quy thay thế cặp Conv2d + BatchNorm2d
    for name, module in list(model_fused.named_children()):
        if isinstance(module, nn.Sequential):
            i = 0
            while i < len(module) - 1:
                if isinstance(module[i], nn.Conv2d) and isinstance(module[i + 1], nn.BatchNorm2d):
                    module[i] = _fuse_conv_and_bn(module[i], module[i + 1])
                    module[i + 1] = nn.Identity()
                    i += 2
                else:
                    i += 1
        else:
            fuse_conv_bn(module)

    return model_fused

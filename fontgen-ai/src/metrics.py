#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
評価指標 - 画像品質の測定
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict


def calculate_mse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """
    Mean Squared Error

    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)

    Returns:
        MSE value
    """
    mse = F.mse_loss(pred, target, reduction='mean')
    return mse.item()


def calculate_psnr(pred: torch.Tensor, target: torch.Tensor, max_val: float = 1.0) -> float:
    """
    Peak Signal-to-Noise Ratio

    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)
        max_val: Maximum pixel value (default: 1.0)

    Returns:
        PSNR in dB
    """
    mse = F.mse_loss(pred, target, reduction='mean')

    if mse < 1e-10:
        return 100.0  # Perfect reconstruction

    psnr = 20 * torch.log10(torch.tensor(max_val) / torch.sqrt(mse))
    return psnr.item()


def calculate_ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int = 11,
    size_average: bool = True
) -> float:
    """
    Structural Similarity Index (SSIM)

    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)
        window_size: Window size for SSIM
        size_average: Average over batch

    Returns:
        SSIM value (0-1, higher is better)
    """
    # Constants
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    # Create Gaussian window
    channel = pred.size(1)
    window = _create_window(window_size, channel).to(pred.device)

    # Calculate SSIM
    mu1 = F.conv2d(pred, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(target, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(pred * pred, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(target * target, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(pred * target, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean().item()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def _create_window(window_size: int, channel: int) -> torch.Tensor:
    """Create Gaussian window for SSIM"""
    def _gaussian(window_size, sigma):
        gauss = torch.Tensor([
            np.exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()

    _1D_window = _gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window


def calculate_all_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    """
    Calculate all metrics

    Args:
        pred: Predicted image (B, C, H, W)
        target: Target image (B, C, H, W)

    Returns:
        Dict of metrics
    """
    return {
        'mse': calculate_mse(pred, target),
        'psnr': calculate_psnr(pred, target),
        'ssim': calculate_ssim(pred, target),
    }


if __name__ == "__main__":
    # Test
    batch_size = 4
    pred = torch.rand(batch_size, 1, 128, 128)
    target = torch.rand(batch_size, 1, 128, 128)

    metrics = calculate_all_metrics(pred, target)
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")

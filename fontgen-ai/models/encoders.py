#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
エンコーダー - 骨格とスタイルの抽出
"""

import torch
import torch.nn as nn
from typing import Tuple


class SkeletonEncoder(nn.Module):
    """
    骨格エンコーダー - 文字の構造的特徴を抽出

    入力: (B, 1, 128, 128)
    出力: z_content (B, z_content_dim)
    """

    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: list = [64, 128, 256, 512],
        z_dim: int = 128,
        image_size: int = 128,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.z_dim = z_dim

        # Convolutional layers
        layers = []
        in_ch = input_channels

        for out_ch in conv_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)

        # 出力サイズを計算
        # 128 -> 64 -> 32 -> 16 -> 8
        final_size = image_size // (2 ** len(conv_channels))
        final_dim = conv_channels[-1] * final_size * final_size

        # Fully connected layers (mean and log_var for VAE)
        self.fc_mean = nn.Linear(final_dim, z_dim)
        self.fc_logvar = nn.Linear(final_dim, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input image (B, 1, H, W)

        Returns:
            mean: Mean of latent distribution (B, z_dim)
            logvar: Log variance of latent distribution (B, z_dim)
        """
        # Convolutional layers
        h = self.conv_layers(x)

        # Flatten
        h = h.view(h.size(0), -1)

        # Mean and log variance
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)

        return mean, logvar


class StyleEncoder(nn.Module):
    """
    スタイルエンコーダー - 文字のスタイル特徴を抽出

    入力: (B, 1, 128, 128)
    出力: z_style (B, z_style_dim)
    """

    def __init__(
        self,
        input_channels: int = 1,
        conv_channels: list = [64, 128, 256],
        z_dim: int = 64,
        image_size: int = 128,
    ):
        super().__init__()

        self.input_channels = input_channels
        self.z_dim = z_dim

        # Convolutional layers
        layers = []
        in_ch = input_channels

        for out_ch in conv_channels:
            layers.extend([
                nn.Conv2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            ])
            in_ch = out_ch

        self.conv_layers = nn.Sequential(*layers)

        # Adaptive pooling (より柔軟なスタイル抽出)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))

        # Fully connected layers
        final_dim = conv_channels[-1] * 4 * 4
        self.fc_mean = nn.Linear(final_dim, z_dim)
        self.fc_logvar = nn.Linear(final_dim, z_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input image (B, 1, H, W)

        Returns:
            mean: Mean of style distribution (B, z_dim)
            logvar: Log variance of style distribution (B, z_dim)
        """
        # Convolutional layers
        h = self.conv_layers(x)

        # Adaptive pooling
        h = self.adaptive_pool(h)

        # Flatten
        h = h.view(h.size(0), -1)

        # Mean and log variance
        mean = self.fc_mean(h)
        logvar = self.fc_logvar(h)

        return mean, logvar


def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """
    Reparameterization trick for VAE

    z = mean + std * epsilon
    where epsilon ~ N(0, 1)

    Args:
        mean: Mean (B, z_dim)
        logvar: Log variance (B, z_dim)

    Returns:
        z: Sampled latent variable (B, z_dim)
    """
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


def test_encoders():
    """テスト関数"""
    batch_size = 4
    image_size = 128

    # ダミー入力
    x = torch.randn(batch_size, 1, image_size, image_size)

    # 骨格エンコーダー
    skeleton_encoder = SkeletonEncoder(
        input_channels=1,
        conv_channels=[64, 128, 256, 512],
        z_dim=128,
        image_size=image_size,
    )

    mean_content, logvar_content = skeleton_encoder(x)
    z_content = reparameterize(mean_content, logvar_content)

    print("Skeleton Encoder:")
    print(f"  Input shape: {x.shape}")
    print(f"  Mean shape: {mean_content.shape}")
    print(f"  LogVar shape: {logvar_content.shape}")
    print(f"  z_content shape: {z_content.shape}")

    # スタイルエンコーダー
    style_encoder = StyleEncoder(
        input_channels=1,
        conv_channels=[64, 128, 256],
        z_dim=64,
        image_size=image_size,
    )

    mean_style, logvar_style = style_encoder(x)
    z_style = reparameterize(mean_style, logvar_style)

    print("\nStyle Encoder:")
    print(f"  Input shape: {x.shape}")
    print(f"  Mean shape: {mean_style.shape}")
    print(f"  LogVar shape: {logvar_style.shape}")
    print(f"  z_style shape: {z_style.shape}")

    # パラメータ数
    total_params_skeleton = sum(p.numel() for p in skeleton_encoder.parameters())
    total_params_style = sum(p.numel() for p in style_encoder.parameters())

    print(f"\nTotal parameters:")
    print(f"  Skeleton Encoder: {total_params_skeleton:,}")
    print(f"  Style Encoder: {total_params_style:,}")
    print(f"  Total: {total_params_skeleton + total_params_style:,}")


if __name__ == "__main__":
    test_encoders()

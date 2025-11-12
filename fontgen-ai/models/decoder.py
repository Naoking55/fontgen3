#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
デコーダー - 潜在変数から画像を生成
"""

import torch
import torch.nn as nn


class FontDecoder(nn.Module):
    """
    フォントデコーダー - 骨格とスタイルから文字画像を生成

    入力: z_content (B, z_content_dim) + z_style (B, z_style_dim)
    出力: reconstructed image (B, 1, 128, 128)
    """

    def __init__(
        self,
        z_content_dim: int = 128,
        z_style_dim: int = 64,
        conv_channels: list = [512, 256, 128, 64],
        output_channels: int = 1,
        image_size: int = 128,
    ):
        super().__init__()

        self.z_content_dim = z_content_dim
        self.z_style_dim = z_style_dim
        self.z_dim = z_content_dim + z_style_dim
        self.output_channels = output_channels

        # 初期サイズを計算 (逆畳み込みの回数から)
        # 128 <- 64 <- 32 <- 16 <- 8
        self.initial_size = image_size // (2 ** len(conv_channels))
        self.initial_channels = conv_channels[0]

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(self.z_dim, self.initial_channels * self.initial_size ** 2),
            nn.ReLU(inplace=True),
        )

        # Deconvolutional layers (Transposed Convolution)
        layers = []
        in_ch = self.initial_channels

        for idx, out_ch in enumerate(conv_channels[1:] + [output_channels]):
            is_last = (idx == len(conv_channels) - 1)

            layers.append(
                nn.ConvTranspose2d(
                    in_ch, out_ch,
                    kernel_size=4, stride=2, padding=1
                )
            )

            if not is_last:
                layers.append(nn.BatchNorm2d(out_ch))
                layers.append(nn.ReLU(inplace=True))
            else:
                # 最終層はSigmoid（[0, 1]範囲に）
                layers.append(nn.Sigmoid())

            in_ch = out_ch

        self.deconv_layers = nn.Sequential(*layers)

    def forward(
        self,
        z_content: torch.Tensor,
        z_style: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass

        Args:
            z_content: Content latent variable (B, z_content_dim)
            z_style: Style latent variable (B, z_style_dim)

        Returns:
            reconstructed: Reconstructed image (B, 1, H, W)
        """
        # Concatenate content and style
        z = torch.cat([z_content, z_style], dim=1)  # (B, z_content_dim + z_style_dim)

        # Fully connected
        h = self.fc(z)  # (B, initial_channels * initial_size^2)

        # Reshape to 4D tensor
        h = h.view(
            -1,
            self.initial_channels,
            self.initial_size,
            self.initial_size
        )  # (B, C, H, W)

        # Deconvolutional layers
        reconstructed = self.deconv_layers(h)

        return reconstructed


def test_decoder():
    """テスト関数"""
    batch_size = 4
    z_content_dim = 128
    z_style_dim = 64
    image_size = 128

    # ダミー潜在変数
    z_content = torch.randn(batch_size, z_content_dim)
    z_style = torch.randn(batch_size, z_style_dim)

    # デコーダー
    decoder = FontDecoder(
        z_content_dim=z_content_dim,
        z_style_dim=z_style_dim,
        conv_channels=[512, 256, 128, 64],
        output_channels=1,
        image_size=image_size,
    )

    # 生成
    reconstructed = decoder(z_content, z_style)

    print("Font Decoder:")
    print(f"  z_content shape: {z_content.shape}")
    print(f"  z_style shape: {z_style.shape}")
    print(f"  Reconstructed shape: {reconstructed.shape}")
    print(f"  Output range: [{reconstructed.min():.3f}, {reconstructed.max():.3f}]")

    # パラメータ数
    total_params = sum(p.numel() for p in decoder.parameters())
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    test_decoder()

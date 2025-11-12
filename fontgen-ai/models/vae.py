#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAE - Variational Autoencoder for Font Generation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from .encoders import SkeletonEncoder, StyleEncoder, reparameterize
from .decoder import FontDecoder


class FontVAE(nn.Module):
    """
    フォント生成VAE

    骨格(content)とスタイル(style)を分離学習
    """

    def __init__(
        self,
        image_size: int = 128,
        z_content_dim: int = 128,
        z_style_dim: int = 64,
        input_channels: int = 1,
        output_channels: int = 1,
    ):
        super().__init__()

        self.image_size = image_size
        self.z_content_dim = z_content_dim
        self.z_style_dim = z_style_dim

        # Encoders
        self.skeleton_encoder = SkeletonEncoder(
            input_channels=input_channels,
            conv_channels=[64, 128, 256, 512],
            z_dim=z_content_dim,
            image_size=image_size,
        )

        self.style_encoder = StyleEncoder(
            input_channels=input_channels,
            conv_channels=[64, 128, 256],
            z_dim=z_style_dim,
            image_size=image_size,
        )

        # Decoder
        self.decoder = FontDecoder(
            z_content_dim=z_content_dim,
            z_style_dim=z_style_dim,
            conv_channels=[512, 256, 128, 64],
            output_channels=output_channels,
            image_size=image_size,
        )

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Encode image to latent variables

        Args:
            x: Input image (B, 1, H, W)

        Returns:
            mean_content, logvar_content, mean_style, logvar_style
        """
        mean_content, logvar_content = self.skeleton_encoder(x)
        mean_style, logvar_style = self.style_encoder(x)

        return mean_content, logvar_content, mean_style, logvar_style

    def decode(
        self,
        z_content: torch.Tensor,
        z_style: torch.Tensor
    ) -> torch.Tensor:
        """
        Decode latent variables to image

        Args:
            z_content: Content latent (B, z_content_dim)
            z_style: Style latent (B, z_style_dim)

        Returns:
            reconstructed: Reconstructed image (B, 1, H, W)
        """
        return self.decoder(z_content, z_style)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass

        Args:
            x: Input image (B, 1, H, W)

        Returns:
            Dict containing:
                - reconstructed: (B, 1, H, W)
                - mean_content: (B, z_content_dim)
                - logvar_content: (B, z_content_dim)
                - mean_style: (B, z_style_dim)
                - logvar_style: (B, z_style_dim)
                - z_content: (B, z_content_dim)
                - z_style: (B, z_style_dim)
        """
        # Encode
        mean_content, logvar_content, mean_style, logvar_style = self.encode(x)

        # Reparameterize
        z_content = reparameterize(mean_content, logvar_content)
        z_style = reparameterize(mean_style, logvar_style)

        # Decode
        reconstructed = self.decode(z_content, z_style)

        return {
            "reconstructed": reconstructed,
            "mean_content": mean_content,
            "logvar_content": logvar_content,
            "mean_style": mean_style,
            "logvar_style": logvar_style,
            "z_content": z_content,
            "z_style": z_style,
        }


class VAELoss(nn.Module):
    """
    VAE損失関数

    Total Loss = Reconstruction + KL Divergence + Style Consistency + Skeleton Consistency
    """

    def __init__(
        self,
        recon_weight: float = 1.0,
        kl_weight: float = 0.001,
        style_weight: float = 0.5,
        skeleton_weight: float = 0.5,
    ):
        super().__init__()

        self.recon_weight = recon_weight
        self.kl_weight = kl_weight
        self.style_weight = style_weight
        self.skeleton_weight = skeleton_weight

    def reconstruction_loss(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor
    ) -> torch.Tensor:
        """
        再構成損失 (MSE + BCE)

        Args:
            reconstructed: (B, 1, H, W)
            original: (B, 1, H, W)

        Returns:
            loss: scalar
        """
        # MSE loss
        mse_loss = F.mse_loss(reconstructed, original, reduction="mean")

        # BCE loss (binary cross entropy)
        bce_loss = F.binary_cross_entropy(
            reconstructed, original, reduction="mean"
        )

        # 両方の平均
        return (mse_loss + bce_loss) / 2.0

    def kl_divergence(
        self,
        mean: torch.Tensor,
        logvar: torch.Tensor
    ) -> torch.Tensor:
        """
        KL Divergence: KL(q(z|x) || p(z))
        where p(z) = N(0, I)

        Args:
            mean: (B, z_dim)
            logvar: (B, z_dim)

        Returns:
            loss: scalar
        """
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return kl_loss / mean.size(0)  # バッチ平均

    def style_consistency_loss(
        self,
        z_style: torch.Tensor,
        font_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        スタイル一貫性損失
        同じフォントの文字は同じスタイルを持つべき

        Args:
            z_style: (B, z_style_dim)
            font_ids: (B,)

        Returns:
            loss: scalar
        """
        # 同じフォントIDのペアを見つける
        unique_fonts = torch.unique(font_ids)
        loss = 0.0
        count = 0

        for font_id in unique_fonts:
            mask = (font_ids == font_id)
            same_font_styles = z_style[mask]

            if same_font_styles.size(0) > 1:
                # 同じフォント内のスタイルの分散を最小化
                mean_style = same_font_styles.mean(dim=0, keepdim=True)
                loss += F.mse_loss(same_font_styles, mean_style.expand_as(same_font_styles))
                count += 1

        return loss / max(count, 1)

    def skeleton_consistency_loss(
        self,
        z_content: torch.Tensor,
        char_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        骨格一貫性損失
        同じ文字の異なるフォントは同じ骨格を持つべき

        Args:
            z_content: (B, z_content_dim)
            char_ids: (B,)

        Returns:
            loss: scalar
        """
        # 同じ文字IDのペアを見つける
        unique_chars = torch.unique(char_ids)
        loss = 0.0
        count = 0

        for char_id in unique_chars:
            mask = (char_ids == char_id)
            same_char_contents = z_content[mask]

            if same_char_contents.size(0) > 1:
                # 同じ文字内の骨格の分散を最小化
                mean_content = same_char_contents.mean(dim=0, keepdim=True)
                loss += F.mse_loss(same_char_contents, mean_content.expand_as(same_char_contents))
                count += 1

        return loss / max(count, 1)

    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        original: torch.Tensor,
        font_ids: torch.Tensor = None,
        char_ids: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate total loss

        Args:
            outputs: Model outputs (from FontVAE.forward)
            original: Original images (B, 1, H, W)
            font_ids: Font IDs (B,) [optional]
            char_ids: Character IDs (B,) [optional]

        Returns:
            Dict containing all losses
        """
        reconstructed = outputs["reconstructed"]
        mean_content = outputs["mean_content"]
        logvar_content = outputs["logvar_content"]
        mean_style = outputs["mean_style"]
        logvar_style = outputs["logvar_style"]
        z_content = outputs["z_content"]
        z_style = outputs["z_style"]

        # 再構成損失
        recon_loss = self.reconstruction_loss(reconstructed, original)

        # KL Divergence
        kl_content = self.kl_divergence(mean_content, logvar_content)
        kl_style = self.kl_divergence(mean_style, logvar_style)
        kl_loss = (kl_content + kl_style) / 2.0

        # Style consistency (同じフォント)
        style_loss = torch.tensor(0.0, device=reconstructed.device)
        if font_ids is not None:
            style_loss = self.style_consistency_loss(z_style, font_ids)

        # Skeleton consistency (同じ文字)
        skeleton_loss = torch.tensor(0.0, device=reconstructed.device)
        if char_ids is not None:
            skeleton_loss = self.skeleton_consistency_loss(z_content, char_ids)

        # Total loss
        total_loss = (
            self.recon_weight * recon_loss +
            self.kl_weight * kl_loss +
            self.style_weight * style_loss +
            self.skeleton_weight * skeleton_loss
        )

        return {
            "total_loss": total_loss,
            "recon_loss": recon_loss,
            "kl_loss": kl_loss,
            "kl_content": kl_content,
            "kl_style": kl_style,
            "style_loss": style_loss,
            "skeleton_loss": skeleton_loss,
        }


def test_vae():
    """テスト関数"""
    batch_size = 8
    image_size = 128

    # ダミーデータ
    x = torch.rand(batch_size, 1, image_size, image_size)
    font_ids = torch.randint(0, 3, (batch_size,))  # 3つのフォント
    char_ids = torch.randint(0, 10, (batch_size,))  # 10文字

    # モデル
    model = FontVAE(
        image_size=image_size,
        z_content_dim=128,
        z_style_dim=64,
    )

    # 順伝播
    outputs = model(x)

    print("FontVAE Forward:")
    for key, value in outputs.items():
        print(f"  {key}: {value.shape}")

    # 損失計算
    criterion = VAELoss(
        recon_weight=1.0,
        kl_weight=0.001,
        style_weight=0.5,
        skeleton_weight=0.5,
    )

    losses = criterion(outputs, x, font_ids, char_ids)

    print("\nLosses:")
    for key, value in losses.items():
        print(f"  {key}: {value.item():.4f}")

    # パラメータ数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params:,}")


if __name__ == "__main__":
    test_vae()

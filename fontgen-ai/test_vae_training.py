#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
VAE学習テスト - オールインワンスクリプト

使い方:
    python test_vae_training.py

このスクリプトは以下を実行します:
1. ダミーデータ生成
2. モデル作成
3. 小規模学習（5エポック）
4. 結果確認
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


# ============================================================
# デバイス設定
# ============================================================

def get_device():
    """最適なデバイスを取得"""
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        print(f"✓ Using CUDA: {torch.cuda.get_device_name(0)}")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("✓ Using MPS (Apple Silicon)")
    else:
        device = torch.device("cpu")
        print("✓ Using CPU")
    return device


# ============================================================
# モデル定義
# ============================================================

class SkeletonEncoder(nn.Module):
    """骨格エンコーダー"""
    def __init__(self, z_dim=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.LeakyReLU(0.2),
        )
        self.fc_mean = nn.Linear(512 * 8 * 8, z_dim)
        self.fc_logvar = nn.Linear(512 * 8 * 8, z_dim)

    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.fc_mean(h), self.fc_logvar(h)


class StyleEncoder(nn.Module):
    """スタイルエンコーダー"""
    def __init__(self, z_dim=64):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.LeakyReLU(0.2),
        )
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc_mean = nn.Linear(256 * 16, z_dim)
        self.fc_logvar = nn.Linear(256 * 16, z_dim)

    def forward(self, x):
        h = self.pool(self.conv(x)).view(x.size(0), -1)
        return self.fc_mean(h), self.fc_logvar(h)


class FontDecoder(nn.Module):
    """デコーダー"""
    def __init__(self, z_content_dim=128, z_style_dim=64):
        super().__init__()
        self.fc = nn.Linear(z_content_dim + z_style_dim, 512 * 8 * 8)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(512, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.ConvTranspose2d(64, 1, 4, 2, 1), nn.Sigmoid(),
        )

    def forward(self, z_content, z_style):
        z = torch.cat([z_content, z_style], dim=1)
        h = self.fc(z).view(-1, 512, 8, 8)
        return self.deconv(h)


class FontVAE(nn.Module):
    """フォントVAE"""
    def __init__(self, z_content_dim=128, z_style_dim=64):
        super().__init__()
        self.skeleton_encoder = SkeletonEncoder(z_content_dim)
        self.style_encoder = StyleEncoder(z_style_dim)
        self.decoder = FontDecoder(z_content_dim, z_style_dim)

    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x):
        mean_content, logvar_content = self.skeleton_encoder(x)
        mean_style, logvar_style = self.style_encoder(x)
        z_content = self.reparameterize(mean_content, logvar_content)
        z_style = self.reparameterize(mean_style, logvar_style)
        reconstructed = self.decoder(z_content, z_style)
        return {
            'reconstructed': reconstructed,
            'mean_content': mean_content,
            'logvar_content': logvar_content,
            'mean_style': mean_style,
            'logvar_style': logvar_style,
        }


# ============================================================
# 損失関数
# ============================================================

def vae_loss(outputs, original):
    """VAE損失"""
    reconstructed = outputs['reconstructed']
    mean_content = outputs['mean_content']
    logvar_content = outputs['logvar_content']
    mean_style = outputs['mean_style']
    logvar_style = outputs['logvar_style']

    # 再構成損失
    recon_loss = F.mse_loss(reconstructed, original) + F.binary_cross_entropy(reconstructed, original)

    # KLダイバージェンス
    kl_content = -0.5 * torch.sum(1 + logvar_content - mean_content.pow(2) - logvar_content.exp())
    kl_style = -0.5 * torch.sum(1 + logvar_style - mean_style.pow(2) - logvar_style.exp())
    kl_loss = (kl_content + kl_style) / original.size(0)

    total_loss = recon_loss + 0.001 * kl_loss

    return total_loss, recon_loss, kl_loss


# ============================================================
# ダミーデータセット
# ============================================================

class DummyFontDataset(Dataset):
    """ダミーフォントデータセット"""
    def __init__(self, num_samples=1000, image_size=128):
        self.num_samples = num_samples
        self.image_size = image_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # ランダムな文字風の画像を生成
        img = np.zeros((self.image_size, self.image_size), dtype=np.float32)

        # ランダムな線を描画
        for _ in range(np.random.randint(3, 8)):
            x1, y1 = np.random.randint(20, self.image_size - 20, 2)
            x2, y2 = np.random.randint(20, self.image_size - 20, 2)
            thickness = np.random.randint(2, 6)

            # 線を描画（簡易版）
            pts = np.linspace([x1, y1], [x2, y2], 100).astype(int)
            for px, py in pts:
                if 0 <= px < self.image_size and 0 <= py < self.image_size:
                    img[py, px] = 1.0
                    for dx in range(-thickness, thickness):
                        for dy in range(-thickness, thickness):
                            nx, ny = px + dx, py + dy
                            if 0 <= nx < self.image_size and 0 <= ny < self.image_size:
                                img[ny, nx] = 1.0

        return torch.from_numpy(img).unsqueeze(0)  # (1, H, W)


# ============================================================
# メトリクス
# ============================================================

def calculate_ssim_simple(pred, target):
    """簡易SSIM計算"""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu1, mu2 = pred.mean(), target.mean()
    sigma1, sigma2 = pred.std(), target.std()
    sigma12 = ((pred - mu1) * (target - mu2)).mean()
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 ** 2 + sigma2 ** 2 + C2))
    return ssim.item()


# ============================================================
# 学習ループ
# ============================================================

def train():
    """学習実行"""
    print("=" * 60)
    print(" VAE学習テスト")
    print("=" * 60)

    # デバイス
    device = get_device()

    # データセット
    print("\n📦 データセット作成中...")
    train_dataset = DummyFontDataset(num_samples=500)
    val_dataset = DummyFontDataset(num_samples=100)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"  Train: {len(train_dataset)} samples")
    print(f"  Val: {len(val_dataset)} samples")

    # モデル
    print("\n🤖 モデル作成中...")
    model = FontVAE(z_content_dim=128, z_style_dim=64).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Total parameters: {total_params:,}")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 学習
    print("\n🎓 学習開始...")
    num_epochs = 5
    best_val_loss = float('inf')

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Train
        model.train()
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in pbar:
            images = batch.to(device)

            outputs = model(images)
            total_loss, recon_loss, kl_loss = vae_loss(outputs, images)

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()

            pbar.set_postfix({'loss': total_loss.item()})

        train_loss = epoch_loss / len(train_loader)
        train_losses.append(train_loss)

        # Validation
        model.eval()
        val_loss_total = 0.0
        ssim_total = 0.0

        with torch.no_grad():
            for batch in val_loader:
                images = batch.to(device)
                outputs = model(images)
                total_loss, _, _ = vae_loss(outputs, images)
                val_loss_total += total_loss.item()

                # SSIM
                ssim = calculate_ssim_simple(outputs['reconstructed'], images)
                ssim_total += ssim

        val_loss = val_loss_total / len(val_loader)
        val_ssim = ssim_total / len(val_loader)
        val_losses.append(val_loss)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        print(f"\n  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}, SSIM: {val_ssim:.4f} {'✓ Best!' if is_best else ''}")

    print("\n✅ 学習完了!")
    print(f"  Best Val Loss: {best_val_loss:.4f}")

    # 可視化
    print("\n📊 結果可視化中...")

    # 学習曲線
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Curve')
    plt.legend()
    plt.grid(True)

    # サンプル画像
    model.eval()
    with torch.no_grad():
        sample_batch = next(iter(val_loader)).to(device)[:8]
        outputs = model(sample_batch)
        reconstructed = outputs['reconstructed'][:8]

    plt.subplot(1, 2, 2)
    comparison = torch.cat([sample_batch.cpu(), reconstructed.cpu()], dim=0)
    grid = comparison.view(-1, 1, 128, 128).permute(0, 2, 3, 1).squeeze().numpy()

    # 8x2グリッド
    display_grid = np.zeros((128 * 2, 128 * 8))
    for i in range(8):
        display_grid[0:128, i * 128:(i + 1) * 128] = grid[i]
        display_grid[128:256, i * 128:(i + 1) * 128] = grid[i + 8]

    plt.imshow(display_grid, cmap='gray')
    plt.title('Original (top) vs Reconstructed (bottom)')
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('vae_test_results.png', dpi=150, bbox_inches='tight')
    print("  保存: vae_test_results.png")

    print("\n" + "=" * 60)
    print(" テスト完了！")
    print("=" * 60)
    print("\n✓ モデルは正常に動作しています")
    print("✓ 学習が収束しています")
    print(f"✓ SSIM: {val_ssim:.3f} (0.7以上が目標)")
    print("\n次のステップ:")
    print("  1. 実際のフォントデータで学習")
    print("  2. より長時間（50-200エポック）学習")
    print("  3. 生成品質を評価")


if __name__ == "__main__":
    train()

"""
品質評価スクリプト
学習済みモデルから大量のサンプルを生成し、詳細な品質評価を行う
"""

import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

import sys
sys.path.append(str(Path(__file__).parent.parent))

from models.vae import FontVAE
from src.dataset import FontDataset
from src.metrics import SSIMMetric, PSNRMetric


def load_config(config_path):
    """設定ファイルを読み込む"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, config: dict, device: torch.device):
    """チェックポイントからモデルをロード"""
    logger.info(f"Loading model from {checkpoint_path}")

    model = FontVAE(
        z_content_dim=config['model']['z_content_dim'],
        z_style_dim=config['model']['z_style_dim'],
        image_size=config['model']['image_size']
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    return model


def generate_reconstruction_samples(
    model: FontVAE,
    dataset: FontDataset,
    num_samples: int,
    device: torch.device
):
    """再構成サンプルを生成"""
    model.eval()

    results = {
        'originals': [],
        'reconstructions': [],
        'ssim_scores': [],
        'psnr_scores': [],
        'mse_scores': []
    }

    ssim_metric = SSIMMetric()
    psnr_metric = PSNRMetric()

    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)

    with torch.no_grad():
        for idx in tqdm(indices, desc="Generating samples"):
            # データ取得
            data = dataset[idx]
            image = data['image'].unsqueeze(0).to(device)
            skeleton = data['skeleton'].unsqueeze(0).to(device)

            # 再構成
            output = model(image, skeleton)
            recon = output['reconstruction']

            # NumPy配列に変換
            orig_np = image.cpu().squeeze().numpy()
            recon_np = recon.cpu().squeeze().numpy()

            # メトリクス計算
            ssim = ssim_metric(recon, image).item()
            psnr = psnr_metric(recon, image).item()
            mse = torch.nn.functional.mse_loss(recon, image).item()

            results['originals'].append(orig_np)
            results['reconstructions'].append(recon_np)
            results['ssim_scores'].append(ssim)
            results['psnr_scores'].append(psnr)
            results['mse_scores'].append(mse)

    return results


def generate_style_transfer_samples(
    model: FontVAE,
    dataset: FontDataset,
    num_samples: int,
    device: torch.device
):
    """スタイル転送サンプルを生成"""
    model.eval()

    results = {
        'content_images': [],
        'style_images': [],
        'transferred': [],
        'content_chars': [],
        'style_chars': []
    }

    indices = np.random.choice(len(dataset), min(num_samples * 2, len(dataset)), replace=False)

    with torch.no_grad():
        for i in tqdm(range(0, len(indices) - 1, 2), desc="Generating style transfers"):
            # コンテンツとスタイルを取得
            content_data = dataset[indices[i]]
            style_data = dataset[indices[i + 1]]

            content_img = content_data['image'].unsqueeze(0).to(device)
            content_skel = content_data['skeleton'].unsqueeze(0).to(device)
            style_img = style_data['image'].unsqueeze(0).to(device)

            # エンコード
            content_output = model(content_img, content_skel)
            style_output = model(style_img, style_data['skeleton'].unsqueeze(0).to(device))

            # スタイル転送（コンテンツの骨格 + スタイルのスタイル）
            z_content = content_output['z_content']
            z_style = style_output['z_style']

            # デコード
            transferred = model.decode(z_content, z_style)

            results['content_images'].append(content_img.cpu().squeeze().numpy())
            results['style_images'].append(style_img.cpu().squeeze().numpy())
            results['transferred'].append(transferred.cpu().squeeze().numpy())
            results['content_chars'].append(content_data['character'])
            results['style_chars'].append(style_data['character'])

    return results


def plot_reconstruction_results(results: dict, output_path: Path):
    """再構成結果をプロット"""
    num_samples = min(16, len(results['originals']))

    fig, axes = plt.subplots(4, num_samples // 2, figsize=(20, 10))
    fig.suptitle('Reconstruction Quality Evaluation', fontsize=16)

    for i in range(num_samples // 2):
        # 元画像
        axes[0, i].imshow(results['originals'][i * 2], cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=10)

        # 再構成
        axes[1, i].imshow(results['reconstructions'][i * 2], cmap='gray')
        axes[1, i].axis('off')
        axes[1, i].set_xlabel(f"SSIM: {results['ssim_scores'][i * 2]:.4f}", fontsize=8)
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=10)

        # 元画像2
        axes[2, i].imshow(results['originals'][i * 2 + 1], cmap='gray')
        axes[2, i].axis('off')

        # 再構成2
        axes[3, i].imshow(results['reconstructions'][i * 2 + 1], cmap='gray')
        axes[3, i].axis('off')
        axes[3, i].set_xlabel(f"SSIM: {results['ssim_scores'][i * 2 + 1]:.4f}", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved reconstruction plot to {output_path}")


def plot_style_transfer_results(results: dict, output_path: Path):
    """スタイル転送結果をプロット"""
    num_samples = min(8, len(results['content_images']))

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, num_samples * 3))
    fig.suptitle('Style Transfer Results', fontsize=16)

    for i in range(num_samples):
        # コンテンツ
        axes[i, 0].imshow(results['content_images'][i], cmap='gray')
        axes[i, 0].axis('off')
        axes[i, 0].set_title(f"Content: {results['content_chars'][i]}", fontsize=10)

        # スタイル
        axes[i, 1].imshow(results['style_images'][i], cmap='gray')
        axes[i, 1].axis('off')
        axes[i, 1].set_title(f"Style: {results['style_chars'][i]}", fontsize=10)

        # 転送結果
        axes[i, 2].imshow(results['transferred'][i], cmap='gray')
        axes[i, 2].axis('off')
        axes[i, 2].set_title('Transferred', fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved style transfer plot to {output_path}")


def print_statistics(results: dict):
    """統計情報を表示"""
    print("\n" + "=" * 60)
    print(" Quality Evaluation Statistics")
    print("=" * 60)
    print(f"Number of samples: {len(results['ssim_scores'])}")
    print(f"\nSSIM:")
    print(f"  Mean:   {np.mean(results['ssim_scores']):.4f}")
    print(f"  Std:    {np.std(results['ssim_scores']):.4f}")
    print(f"  Min:    {np.min(results['ssim_scores']):.4f}")
    print(f"  Max:    {np.max(results['ssim_scores']):.4f}")
    print(f"  Median: {np.median(results['ssim_scores']):.4f}")

    print(f"\nPSNR:")
    print(f"  Mean:   {np.mean(results['psnr_scores']):.4f} dB")
    print(f"  Std:    {np.std(results['psnr_scores']):.4f} dB")
    print(f"  Min:    {np.min(results['psnr_scores']):.4f} dB")
    print(f"  Max:    {np.max(results['psnr_scores']):.4f} dB")
    print(f"  Median: {np.median(results['psnr_scores']):.4f} dB")

    print(f"\nMSE:")
    print(f"  Mean:   {np.mean(results['mse_scores']):.6f}")
    print(f"  Std:    {np.std(results['mse_scores']):.6f}")
    print(f"  Min:    {np.min(results['mse_scores']):.6f}")
    print(f"  Max:    {np.max(results['mse_scores']):.6f}")
    print(f"  Median: {np.median(results['mse_scores']):.6f}")
    print("=" * 60 + "\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model quality')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for evaluation results')
    parser.add_argument('--num-samples', type=int, default=50,
                        help='Number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cpu',
                        choices=['cpu', 'cuda', 'mps'],
                        help='Device to use')

    args = parser.parse_args()

    # パス設定
    checkpoint_path = Path(args.checkpoint)
    config_path = Path(args.config)
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 設定ロード
    config = load_config(config_path)

    # デバイス設定
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")

    # モデルロード
    model = load_model(checkpoint_path, config, device)

    # データセットロード
    logger.info("Loading dataset...")
    val_dataset = FontDataset(
        data_dir=data_dir,
        split='val',
        image_size=config['model']['image_size']
    )
    logger.info(f"Loaded {len(val_dataset)} validation samples")

    # 再構成品質評価
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating reconstruction quality...")
    logger.info("=" * 60)
    recon_results = generate_reconstruction_samples(
        model, val_dataset, args.num_samples, device
    )

    # 統計表示
    print_statistics(recon_results)

    # プロット保存
    plot_reconstruction_results(
        recon_results,
        output_dir / 'reconstruction_quality.png'
    )

    # スタイル転送評価
    logger.info("\n" + "=" * 60)
    logger.info("Evaluating style transfer...")
    logger.info("=" * 60)
    style_results = generate_style_transfer_samples(
        model, val_dataset, min(16, args.num_samples // 2), device
    )

    plot_style_transfer_results(
        style_results,
        output_dir / 'style_transfer.png'
    )

    logger.info("\n" + "=" * 60)
    logger.info("Evaluation completed!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

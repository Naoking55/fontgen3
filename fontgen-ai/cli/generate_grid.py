"""
グリッド生成スクリプト
複数の文字×複数のスタイルでフォント生成のグリッドを作成
"""

import argparse
import logging
from pathlib import Path
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib import font_manager
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.model import FontVAE
from src.dataset import FontDataset
from src.config import load_config


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


def generate_character_style_grid(
    model: FontVAE,
    dataset: FontDataset,
    num_chars: int,
    num_styles: int,
    device: torch.device
):
    """
    文字×スタイルのグリッドを生成

    行: 異なる文字（コンテンツ）
    列: 異なるスタイル
    """
    model.eval()

    # ランダムに文字とスタイルを選択
    char_indices = np.random.choice(len(dataset), num_chars, replace=False)
    style_indices = np.random.choice(len(dataset), num_styles, replace=False)

    # グリッド作成
    grid = np.zeros((num_chars, num_styles + 2, 128, 128))  # +2 for content and skeleton
    char_labels = []
    style_labels = []

    with torch.no_grad():
        # 各文字（行）に対して
        for i, char_idx in enumerate(tqdm(char_indices, desc="Generating grid")):
            char_data = dataset[char_idx]
            char_img = char_data['image'].unsqueeze(0).to(device)
            char_skel = char_data['skeleton'].unsqueeze(0).to(device)
            char_labels.append(char_data['character'])

            # 元の文字画像を最初の列に
            grid[i, 0] = char_img.cpu().squeeze().numpy()
            # スケルトンを2列目に
            grid[i, 1] = char_skel.cpu().squeeze().numpy()

            # コンテンツエンコーディング
            char_output = model(char_img, char_skel)
            z_content = char_output['z_content']

            # 各スタイル（列）に対して
            for j, style_idx in enumerate(style_indices):
                if i == 0:
                    style_data = dataset[style_idx]
                    style_labels.append(style_data['character'])

                    # スタイルエンコーディング（最初の行のみ）
                    style_img = style_data['image'].unsqueeze(0).to(device)
                    style_skel = style_data['skeleton'].unsqueeze(0).to(device)
                    style_output = model(style_img, style_skel)
                    z_style = style_output['z_style']

                    # スタイルベクトルを保存（他の行で再利用）
                    if 'style_vectors' not in locals():
                        style_vectors = []
                    style_vectors.append(z_style)
                else:
                    z_style = style_vectors[j]

                # スタイル転送
                transferred = model.decode(z_content, z_style)
                grid[i, j + 2] = transferred.cpu().squeeze().numpy()

    return grid, char_labels, style_labels


def plot_grid(grid: np.ndarray, char_labels: list, style_labels: list, output_path: Path):
    """グリッドをプロット"""
    num_chars, num_cols, _, _ = grid.shape
    num_styles = num_cols - 2

    fig, axes = plt.subplots(
        num_chars, num_cols,
        figsize=(num_cols * 1.5, num_chars * 1.5)
    )

    # タイトル
    fig.suptitle('Character × Style Generation Grid', fontsize=16, y=0.995)

    # 列ヘッダー
    col_headers = ['Original', 'Skeleton'] + [f'Style: {s}' for s in style_labels]

    for i in range(num_chars):
        for j in range(num_cols):
            ax = axes[i, j] if num_chars > 1 else axes[j]

            # 画像表示
            ax.imshow(grid[i, j], cmap='gray', vmin=0, vmax=1)
            ax.axis('off')

            # 列ヘッダー（最初の行）
            if i == 0:
                ax.set_title(col_headers[j], fontsize=9, pad=5)

            # 行ラベル（最初の列）
            if j == 0:
                ax.set_ylabel(f'Char: {char_labels[i]}', fontsize=9, rotation=0,
                             ha='right', va='center', labelpad=20)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved grid to {output_path}")


def generate_interpolation_grid(
    model: FontVAE,
    dataset: FontDataset,
    num_steps: int,
    device: torch.device
):
    """スタイル補間グリッドを生成"""
    model.eval()

    # 2つのスタイルを選択
    indices = np.random.choice(len(dataset), 3, replace=False)

    char_data = dataset[indices[0]]
    style1_data = dataset[indices[1]]
    style2_data = dataset[indices[2]]

    with torch.no_grad():
        # コンテンツ
        char_img = char_data['image'].unsqueeze(0).to(device)
        char_skel = char_data['skeleton'].unsqueeze(0).to(device)
        char_output = model(char_img, char_skel)
        z_content = char_output['z_content']

        # スタイル1
        style1_img = style1_data['image'].unsqueeze(0).to(device)
        style1_skel = style1_data['skeleton'].unsqueeze(0).to(device)
        style1_output = model(style1_img, style1_skel)
        z_style1 = style1_output['z_style']

        # スタイル2
        style2_img = style2_data['image'].unsqueeze(0).to(device)
        style2_skel = style2_data['skeleton'].unsqueeze(0).to(device)
        style2_output = model(style2_img, style2_skel)
        z_style2 = style2_output['z_style']

        # 補間
        interpolations = []
        for alpha in np.linspace(0, 1, num_steps):
            z_style_interp = z_style1 * (1 - alpha) + z_style2 * alpha
            generated = model.decode(z_content, z_style_interp)
            interpolations.append(generated.cpu().squeeze().numpy())

    return (
        interpolations,
        char_data['character'],
        style1_data['character'],
        style2_data['character'],
        char_img.cpu().squeeze().numpy(),
        style1_img.cpu().squeeze().numpy(),
        style2_img.cpu().squeeze().numpy()
    )


def plot_interpolation(results: tuple, output_path: Path):
    """補間結果をプロット"""
    (interpolations, char_label, style1_label, style2_label,
     char_img, style1_img, style2_img) = results

    num_steps = len(interpolations)

    fig, axes = plt.subplots(3, num_steps, figsize=(num_steps * 1.5, 5))

    fig.suptitle(
        f'Style Interpolation: {style1_label} → {style2_label} (Character: {char_label})',
        fontsize=14
    )

    # 1行目: オリジナル文字
    for j in range(num_steps):
        axes[0, j].imshow(char_img, cmap='gray')
        axes[0, j].axis('off')
        if j == num_steps // 2:
            axes[0, j].set_title('Original Character', fontsize=10)

    # 2行目: 補間結果
    for j in range(num_steps):
        axes[1, j].imshow(interpolations[j], cmap='gray')
        axes[1, j].axis('off')
        alpha = j / (num_steps - 1)
        axes[1, j].set_xlabel(f'α={alpha:.2f}', fontsize=8)

    # 3行目: スタイル参照
    for j in range(num_steps):
        if j < num_steps // 2:
            axes[2, j].imshow(style1_img, cmap='gray')
        else:
            axes[2, j].imshow(style2_img, cmap='gray')
        axes[2, j].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved interpolation to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate visualization grids')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--config', type=str, required=True,
                        help='Path to config file')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to processed data directory')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for grids')
    parser.add_argument('--num-chars', type=int, default=8,
                        help='Number of characters for grid')
    parser.add_argument('--num-styles', type=int, default=6,
                        help='Number of styles for grid')
    parser.add_argument('--interp-steps', type=int, default=10,
                        help='Number of interpolation steps')
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

    # 文字×スタイルグリッド生成
    logger.info("\nGenerating character × style grid...")
    grid, char_labels, style_labels = generate_character_style_grid(
        model, val_dataset, args.num_chars, args.num_styles, device
    )
    plot_grid(grid, char_labels, style_labels, output_dir / 'character_style_grid.png')

    # スタイル補間グリッド生成
    logger.info("\nGenerating style interpolation...")
    interp_results = generate_interpolation_grid(
        model, val_dataset, args.interp_steps, device
    )
    plot_interpolation(interp_results, output_dir / 'style_interpolation.png')

    logger.info("\n" + "=" * 60)
    logger.info("Grid generation completed!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()

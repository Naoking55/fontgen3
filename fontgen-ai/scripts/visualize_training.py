#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
学習結果可視化スクリプト
"""

import argparse
import sys
from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import glob

sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_training_curves(output_dir: str, save_path: str = None):
    """
    学習曲線を可視化

    Args:
        output_dir: 出力ディレクトリ
        save_path: 保存先パス
    """
    output_dir = Path(output_dir)

    # TensorBoardのログから情報を取得する代わりに、
    # サンプル画像を表示
    samples_dir = output_dir / "samples"

    if not samples_dir.exists():
        print(f"Error: Samples directory not found: {samples_dir}")
        return

    # サンプル画像を取得
    sample_files = sorted(samples_dir.glob("epoch_*.png"))

    if len(sample_files) == 0:
        print(f"Error: No sample images found in {samples_dir}")
        return

    # サンプル画像を表示
    num_samples = min(len(sample_files), 6)
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for idx, sample_file in enumerate(sample_files[:num_samples]):
        img = Image.open(sample_file)
        axes[idx].imshow(img)
        axes[idx].set_title(sample_file.stem)
        axes[idx].axis('off')

    # 残りの軸を非表示
    for idx in range(num_samples, 6):
        axes[idx].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization: {save_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="学習結果可視化")

    parser.add_argument(
        "--model-dir",
        type=str,
        required=True,
        help="モデルディレクトリ",
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="出力画像パス",
    )

    args = parser.parse_args()

    visualize_training_curves(args.model_dir, args.output)


if __name__ == "__main__":
    main()

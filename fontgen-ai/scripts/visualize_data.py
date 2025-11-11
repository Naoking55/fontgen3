#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データ可視化スクリプト
"""

import argparse
import json
import sys
from pathlib import Path
import random
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))


def visualize_dataset(data_dir: str, output_path: str, num_samples: int = 25):
    """
    データセットを可視化

    Args:
        data_dir (str): データディレクトリ
        output_path (str): 出力画像パス
        num_samples (int): サンプル数
    """
    data_dir = Path(data_dir)

    # メタデータ読み込み
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"Error: Metadata not found: {metadata_path}")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print(f"📊 データセット情報:")
    print(f"  フォント数: {metadata['num_fonts']}")
    print(f"    - Train: {metadata['num_train_fonts']}")
    print(f"    - Val: {metadata['num_val_fonts']}")
    print(f"    - Test: {metadata['num_test_fonts']}")
    print(f"  文字数: {len(metadata['characters'])}")
    print(f"  画像サイズ: {metadata['image_size']}x{metadata['image_size']}")

    # サンプルを収集
    train_dir = data_dir / "train"
    if not train_dir.exists():
        print(f"Error: Train directory not found: {train_dir}")
        return

    samples = []
    for font_dir in train_dir.iterdir():
        if not font_dir.is_dir():
            continue

        image_files = list(font_dir.glob("*.png"))
        if len(image_files) > 0:
            # ランダムに1つ選択
            image_path = random.choice(image_files)
            char = image_path.stem
            samples.append((image_path, char, font_dir.name))

    if len(samples) == 0:
        print("Error: No samples found")
        return

    # サンプル数を制限
    if len(samples) > num_samples:
        samples = random.sample(samples, num_samples)

    # グリッドサイズを計算
    grid_size = int(np.ceil(np.sqrt(num_samples)))

    # 可視化
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
    axes = axes.flatten()

    for idx, (image_path, char, font_name) in enumerate(samples):
        if idx >= len(axes):
            break

        # 画像読み込み
        image = Image.open(image_path).convert("L")

        axes[idx].imshow(image, cmap="gray")
        axes[idx].set_title(f"'{char}'\n{font_name[:15]}", fontsize=8)
        axes[idx].axis("off")

    # 残りの軸を非表示
    for idx in range(len(samples), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\n✅ 可視化画像を保存しました: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="データ可視化スクリプト")

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="データディレクトリ",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="data_visualization.png",
        help="出力画像パス (default: data_visualization.png)",
    )

    parser.add_argument(
        "--num-samples",
        type=int,
        default=25,
        help="サンプル数 (default: 25)",
    )

    args = parser.parse_args()

    visualize_dataset(
        data_dir=args.data_dir,
        output_path=args.output,
        num_samples=args.num_samples,
    )


if __name__ == "__main__":
    main()

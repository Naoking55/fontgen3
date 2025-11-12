#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データ情報表示スクリプト
"""

import argparse
import json
import sys
from pathlib import Path
from collections import Counter

sys.path.insert(0, str(Path(__file__).parent.parent))


def show_data_info(data_dir: str):
    """
    データセット情報を表示

    Args:
        data_dir (str): データディレクトリ
    """
    data_dir = Path(data_dir)

    # メタデータ読み込み
    metadata_path = data_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"❌ Error: Metadata not found: {metadata_path}")
        return

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    print("=" * 60)
    print(" データセット情報")
    print("=" * 60)

    print(f"\n📂 ディレクトリ: {data_dir}")
    print(f"📏 画像サイズ: {metadata['image_size']}x{metadata['image_size']}")

    print(f"\n🎨 フォント:")
    print(f"  総数: {metadata['num_fonts']}")
    print(f"    - Train: {metadata['num_train_fonts']}")
    print(f"    - Val: {metadata['num_val_fonts']}")
    print(f"    - Test: {metadata['num_test_fonts']}")

    print(f"\n✏️  文字:")
    print(f"  総数: {len(metadata['characters'])}")
    print(f"  文字リスト: {metadata['characters'][:20]}{'...' if len(metadata['characters']) > 20 else ''}")

    # 各分割のファイル数を数える
    splits = ["train", "val", "test"]
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue

        total_images = 0
        font_char_counts = {}

        for font_dir in split_dir.iterdir():
            if not font_dir.is_dir():
                continue

            images = list(font_dir.glob("*.png"))
            total_images += len(images)
            font_char_counts[font_dir.name] = len(images)

        print(f"\n📊 {split.upper()} 分割:")
        print(f"  フォント数: {len(font_char_counts)}")
        print(f"  総画像数: {total_images}")
        if len(font_char_counts) > 0:
            avg_chars = total_images / len(font_char_counts)
            print(f"  平均文字数/フォント: {avg_chars:.1f}")

            # 文字数が多い/少ないフォントを表示
            sorted_fonts = sorted(
                font_char_counts.items(), key=lambda x: x[1], reverse=True
            )
            print(f"\n  文字数が多いフォント (Top 3):")
            for font_name, count in sorted_fonts[:3]:
                print(f"    - {font_name}: {count}文字")

            print(f"\n  文字数が少ないフォント (Bottom 3):")
            for font_name, count in sorted_fonts[-3:]:
                print(f"    - {font_name}: {count}文字")

    print("\n" + "=" * 60)

    # フォント一覧（オプション）
    if metadata.get("fonts"):
        print(f"\n📝 フォント一覧:")
        for idx, font_name in enumerate(metadata["fonts"][:10], 1):
            print(f"  {idx}. {font_name}")
        if len(metadata["fonts"]) > 10:
            print(f"  ... and {len(metadata['fonts']) - 10} more")

    print()


def main():
    parser = argparse.ArgumentParser(description="データ情報表示スクリプト")

    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="データディレクトリ",
    )

    args = parser.parse_args()

    show_data_info(data_dir=args.data_dir)


if __name__ == "__main__":
    main()

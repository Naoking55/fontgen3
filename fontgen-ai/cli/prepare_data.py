#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データ準備スクリプト - フォントから文字画像を抽出
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List
import logging
from tqdm import tqdm
import numpy as np
from PIL import Image

# パスを追加
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.font_parser import FontParser
from src.preprocessing import Preprocessor
from src.char_sets import get_characters, get_available_charsets

logger = logging.getLogger(__name__)


def prepare_data(
    font_dir: str,
    output_dir: str,
    characters: str,
    image_size: int = 128,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    num_workers: int = 1,
):
    """
    フォントディレクトリからデータセットを準備

    Args:
        font_dir (str): フォントディレクトリ
        output_dir (str): 出力ディレクトリ
        characters (str): 文字セット名（カンマ区切り）
        image_size (int): 画像サイズ
        train_split (float): 訓練データの比率
        val_split (float): 検証データの比率
        test_split (float): テストデータの比率
        num_workers (int): ワーカー数
    """
    font_dir = Path(font_dir)
    output_dir = Path(output_dir)

    # 出力ディレクトリ作成
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "train").mkdir(exist_ok=True)
    (output_dir / "val").mkdir(exist_ok=True)
    if test_split > 0:
        (output_dir / "test").mkdir(exist_ok=True)

    # 文字セット取得
    char_list = get_characters(characters)
    logger.info(f"Target characters: {len(char_list)}")

    # フォントファイル取得 (大文字・小文字両方対応)
    font_files = (
        list(font_dir.glob("*.ttf")) +
        list(font_dir.glob("*.TTF")) +
        list(font_dir.glob("*.otf")) +
        list(font_dir.glob("*.OTF"))
    )
    logger.info(f"Found {len(font_files)} font files")

    if len(font_files) == 0:
        logger.error(f"No font files found in {font_dir}")
        return

    # 前処理器
    preprocessor = Preprocessor(image_size=image_size)

    # フォント情報を収集
    font_info = []
    all_characters = set()

    print("\n📋 フォント情報を収集中...")
    for font_file in tqdm(font_files, desc="Scanning fonts"):
        try:
            parser = FontParser(str(font_file), image_size=image_size)
            available_chars = parser.get_available_characters(char_list)

            if len(available_chars) > 0:
                font_info.append(
                    {
                        "path": font_file,
                        "name": parser.font_name,
                        "available_chars": available_chars,
                    }
                )
                all_characters.update(available_chars)

            parser.close()

        except Exception as e:
            logger.warning(f"Failed to load {font_file.name}: {e}")

    logger.info(f"Valid fonts: {len(font_info)}")
    logger.info(f"Total characters: {len(all_characters)}")

    if len(font_info) == 0:
        logger.error("No valid fonts found")
        return

    # フォントごとにデータを準備
    print("\n🎨 文字画像を生成中...")

    # 単一フォントの場合は文字を分割、複数フォントの場合はフォントを分割
    single_font_mode = len(font_info) == 1

    if single_font_mode:
        logger.info("Single font detected - splitting characters instead of fonts")

        # 文字をtrain/val/testに分割
        font_data = font_info[0]
        available_chars = list(font_data["available_chars"])
        np.random.shuffle(available_chars)

        n_train = int(len(available_chars) * train_split)
        n_val = int(len(available_chars) * val_split)

        train_chars = available_chars[:n_train]
        val_chars = available_chars[n_train : n_train + n_val]
        test_chars = available_chars[n_train + n_val :] if test_split > 0 else []

        logger.info(f"Character split - Train: {len(train_chars)}, Val: {len(val_chars)}, Test: {len(test_chars)}")

        # 各splitに同じフォントを割り当てるが、文字を分ける
        splits = [
            ("train", [{"path": font_data["path"], "name": font_data["name"], "available_chars": train_chars}]),
            ("val", [{"path": font_data["path"], "name": font_data["name"], "available_chars": val_chars}]),
            ("test", [{"path": font_data["path"], "name": font_data["name"], "available_chars": test_chars}] if test_split > 0 else []),
        ]
    else:
        # フォントをtrain/val/testに分割（従来の動作）
        np.random.shuffle(font_info)

        n_train = int(len(font_info) * train_split)
        n_val = int(len(font_info) * val_split)

        train_fonts = font_info[:n_train]
        val_fonts = font_info[n_train : n_train + n_val]
        test_fonts = font_info[n_train + n_val :] if test_split > 0 else []

        splits = [
            ("train", train_fonts),
            ("val", val_fonts),
            ("test", test_fonts),
        ]

    for split_name, split_fonts in splits:
        if len(split_fonts) == 0:
            continue

        logger.info(f"\nProcessing {split_name} split ({len(split_fonts)} fonts)...")

        for font_data in tqdm(split_fonts, desc=f"  {split_name}"):
            font_path = font_data["path"]
            font_name = font_data["name"]
            available_chars = font_data["available_chars"]

            # フォント名をサニタイズ（ファイル名として使用）
            safe_font_name = "".join(
                c if c.isalnum() or c in ("-", "_") else "_" for c in font_name
            )

            # 出力ディレクトリ
            font_output_dir = output_dir / split_name / safe_font_name
            font_output_dir.mkdir(parents=True, exist_ok=True)

            try:
                parser = FontParser(str(font_path), image_size=image_size)

                # 文字を描画
                for char in available_chars:
                    try:
                        # レンダリング
                        image = parser.render_character(char)

                        # 前処理
                        processed = preprocessor.normalize_image(
                            image, center=True, invert=True
                        )

                        # 保存
                        output_path = font_output_dir / f"{char}.png"
                        Image.fromarray((processed * 255).astype(np.uint8)).save(
                            output_path
                        )

                    except Exception as e:
                        logger.warning(f"Failed to process '{char}': {e}")

                parser.close()

            except Exception as e:
                logger.warning(f"Failed to process font {font_name}: {e}")

    # メタデータを保存
    if single_font_mode:
        metadata = {
            "image_size": image_size,
            "num_fonts": 1,
            "single_font_mode": True,
            "num_train_chars": len(train_chars),
            "num_val_chars": len(val_chars),
            "num_test_chars": len(test_chars),
            "characters": sorted(list(all_characters)),
            "train_chars": sorted(train_chars),
            "val_chars": sorted(val_chars),
            "test_chars": sorted(test_chars),
            "font_name": font_info[0]["name"],
        }
    else:
        metadata = {
            "image_size": image_size,
            "num_fonts": len(font_info),
            "single_font_mode": False,
            "num_train_fonts": len(train_fonts),
            "num_val_fonts": len(val_fonts),
            "num_test_fonts": len(test_fonts),
            "characters": sorted(list(all_characters)),
            "fonts": [f["name"] for f in font_info],
            "train_fonts": [f["name"] for f in train_fonts],
            "val_fonts": [f["name"] for f in val_fonts],
            "test_fonts": [f["name"] for f in test_fonts],
        }

    metadata_path = output_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    logger.info(f"\n✅ データ準備完了!")
    logger.info(f"  出力先: {output_dir}")
    if single_font_mode:
        logger.info(f"  モード: 単一フォント（文字分割）")
        logger.info(f"  フォント: {font_info[0]['name']}")
        logger.info(f"  文字数: {len(all_characters)}")
        logger.info(f"    - Train: {len(train_chars)}")
        logger.info(f"    - Val: {len(val_chars)}")
        logger.info(f"    - Test: {len(test_chars)}")
    else:
        logger.info(f"  モード: 複数フォント")
        logger.info(f"  フォント数: {len(font_info)}")
        logger.info(f"    - Train: {len(train_fonts)}")
        logger.info(f"    - Val: {len(val_fonts)}")
        logger.info(f"    - Test: {len(test_fonts)}")
        logger.info(f"  文字数: {len(all_characters)}")
    logger.info(f"  メタデータ: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="データ準備スクリプト")

    parser.add_argument(
        "--font-dir",
        type=str,
        required=True,
        help="フォントディレクトリ",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="出力ディレクトリ",
    )

    parser.add_argument(
        "--characters",
        type=str,
        default="hiragana,katakana",
        help=f"文字セット (利用可能: {', '.join(get_available_charsets())})",
    )

    parser.add_argument(
        "--image-size",
        type=int,
        default=128,
        help="画像サイズ (default: 128)",
    )

    parser.add_argument(
        "--train-split",
        type=float,
        default=0.8,
        help="訓練データの比率 (default: 0.8)",
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.1,
        help="検証データの比率 (default: 0.1)",
    )

    parser.add_argument(
        "--test-split",
        type=float,
        default=0.1,
        help="テストデータの比率 (default: 0.1)",
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=1,
        help="ワーカー数 (default: 1)",
    )

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="ログレベル",
    )

    args = parser.parse_args()

    # ログ設定
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # データ準備実行
    prepare_data(
        font_dir=args.font_dir,
        output_dir=args.output_dir,
        characters=args.characters,
        image_size=args.image_size,
        train_split=args.train_split,
        val_split=args.val_split,
        test_split=args.test_split,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()

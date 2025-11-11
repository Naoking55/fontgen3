#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
フォントパーサー - フォントファイルの読み込みとグリフ抽出
"""

import os
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from fontTools.ttLib import TTFont
import logging

logger = logging.getLogger(__name__)


class FontParser:
    """
    フォントファイルの読み込みとグリフ抽出

    TTF/OTFフォントから文字画像を生成します。
    """

    def __init__(self, font_path: str, image_size: int = 128):
        """
        Args:
            font_path (str): フォントファイルのパス
            image_size (int): レンダリング画像サイズ
        """
        self.font_path = Path(font_path)
        self.image_size = image_size

        if not self.font_path.exists():
            raise FileNotFoundError(f"Font file not found: {font_path}")

        # fontToolsでフォント情報を読み込む
        try:
            self.ttfont = TTFont(str(self.font_path))
            self.font_name = self._get_font_name()
            logger.info(f"Loaded font: {self.font_name}")
        except Exception as e:
            raise ValueError(f"Failed to load font: {e}")

        # PILでフォントを読み込む（レンダリング用）
        try:
            # フォントサイズを自動調整
            self.pil_font = ImageFont.truetype(
                str(self.font_path), size=int(image_size * 0.7)
            )
        except Exception as e:
            raise ValueError(f"Failed to load font for rendering: {e}")

    def _get_font_name(self) -> str:
        """フォント名を取得"""
        try:
            name_table = self.ttfont["name"]
            for record in name_table.names:
                # Full font name (nameID=4)
                if record.nameID == 4:
                    return record.toUnicode()
            # Font family name (nameID=1) をフォールバック
            for record in name_table.names:
                if record.nameID == 1:
                    return record.toUnicode()
        except Exception as e:
            logger.warning(f"Could not extract font name: {e}")

        return self.font_path.stem

    def get_available_characters(self, characters: str) -> List[str]:
        """
        フォントに含まれている文字を取得

        Args:
            characters (str): チェックする文字列

        Returns:
            List[str]: フォントに含まれる文字のリスト
        """
        cmap = self.ttfont.getBestCmap()
        if cmap is None:
            logger.warning(f"No character mapping found in {self.font_name}")
            return []

        available = []
        for char in characters:
            if ord(char) in cmap:
                available.append(char)

        return available

    def render_character(
        self,
        character: str,
        size: Optional[int] = None,
        padding: int = 10,
    ) -> np.ndarray:
        """
        文字を画像にレンダリング

        Args:
            character (str): レンダリングする文字
            size (int, optional): 画像サイズ（Noneの場合はデフォルト値）
            padding (int): パディング

        Returns:
            np.ndarray: グレースケール画像 (size x size), 値は [0, 255]
        """
        if size is None:
            size = self.image_size

        # 白背景の画像を作成
        image = Image.new("L", (size, size), color=255)
        draw = ImageDraw.Draw(image)

        try:
            # テキストのバウンディングボックスを取得
            bbox = draw.textbbox((0, 0), character, font=self.pil_font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]

            # 中央配置の座標を計算
            x = (size - text_width) // 2 - bbox[0]
            y = (size - text_height) // 2 - bbox[1]

            # テキストを描画（黒色）
            draw.text((x, y), character, font=self.pil_font, fill=0)

        except Exception as e:
            logger.warning(f"Failed to render character '{character}': {e}")
            return np.ones((size, size), dtype=np.uint8) * 255

        # NumPy配列に変換
        return np.array(image, dtype=np.uint8)

    def render_characters(
        self, characters: List[str], size: Optional[int] = None
    ) -> Dict[str, np.ndarray]:
        """
        複数の文字を一括レンダリング

        Args:
            characters (List[str]): レンダリングする文字のリスト
            size (int, optional): 画像サイズ

        Returns:
            Dict[str, np.ndarray]: 文字をキーとした画像の辞書
        """
        results = {}

        for char in characters:
            try:
                image = self.render_character(char, size=size)
                results[char] = image
            except Exception as e:
                logger.warning(f"Failed to render '{char}': {e}")

        logger.info(
            f"Rendered {len(results)}/{len(characters)} characters from {self.font_name}"
        )

        return results

    def save_character_image(
        self, character: str, output_path: str, size: Optional[int] = None
    ):
        """
        文字画像をファイルに保存

        Args:
            character (str): 文字
            output_path (str): 出力パス
            size (int, optional): 画像サイズ
        """
        image = self.render_character(character, size=size)
        Image.fromarray(image).save(output_path)
        logger.debug(f"Saved character image: {output_path}")

    def close(self):
        """リソースを解放"""
        if hasattr(self, "ttfont"):
            self.ttfont.close()


def test_font_parser():
    """テスト関数"""
    import matplotlib.pyplot as plt

    # テスト用のフォントパス（環境に応じて変更）
    test_font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
        "/System/Library/Fonts/Helvetica.ttc",  # macOS
        "C:\\Windows\\Fonts\\msgothic.ttc",  # Windows
    ]

    font_path = None
    for path in test_font_paths:
        if os.path.exists(path):
            font_path = path
            break

    if font_path is None:
        print("テスト用フォントが見つかりません")
        return

    # フォントパーサーを作成
    parser = FontParser(font_path, image_size=128)
    print(f"Font name: {parser.font_name}")

    # テスト文字
    test_chars = "あいうえおABCDE12345"

    # 利用可能な文字を確認
    available = parser.get_available_characters(test_chars)
    print(f"Available characters: {len(available)}/{len(test_chars)}")

    # 文字をレンダリング
    if available:
        results = parser.render_characters(available[:9])

        # 可視化
        fig, axes = plt.subplots(3, 3, figsize=(9, 9))
        axes = axes.flatten()

        for idx, (char, image) in enumerate(results.items()):
            if idx >= 9:
                break
            axes[idx].imshow(image, cmap="gray")
            axes[idx].set_title(f"'{char}'")
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig("font_parser_test.png")
        print("Test image saved: font_parser_test.png")

    parser.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_font_parser()

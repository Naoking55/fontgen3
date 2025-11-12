#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
前処理モジュール - データ正規化と拡張
"""

import numpy as np
from typing import Optional, Tuple
import cv2
from scipy import ndimage
import logging

logger = logging.getLogger(__name__)


class Preprocessor:
    """
    画像の前処理とデータ拡張

    - 正規化
    - 中央配置
    - データ拡張（回転、スケール、ノイズ等）
    """

    def __init__(self, image_size: int = 128):
        """
        Args:
            image_size (int): 画像サイズ
        """
        self.image_size = image_size

    def normalize_image(
        self,
        image: np.ndarray,
        center: bool = True,
        invert: bool = False,
    ) -> np.ndarray:
        """
        画像を正規化

        Args:
            image (np.ndarray): 入力画像 (H, W), 値は [0, 255]
            center (bool): 重心を中央に配置
            invert (bool): 反転（白黒反転）

        Returns:
            np.ndarray: 正規化された画像 (image_size, image_size), 値は [0, 1]
        """
        # コピーを作成
        img = image.copy().astype(np.float32)

        # 反転（白背景→黒背景）
        if invert:
            img = 255 - img

        # 二値化（閾値処理）
        img = self.binarize(img, threshold=200)

        # 中央配置
        if center:
            img = self.center_image(img)

        # [0, 1] 範囲に正規化
        img = img / 255.0

        # サイズ調整
        if img.shape[0] != self.image_size or img.shape[1] != self.image_size:
            img = cv2.resize(
                img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR
            )

        return img.astype(np.float32)

    def binarize(self, image: np.ndarray, threshold: int = 200) -> np.ndarray:
        """
        二値化

        Args:
            image (np.ndarray): 入力画像
            threshold (int): 閾値

        Returns:
            np.ndarray: 二値化された画像
        """
        # Otsuの二値化を使用（自動閾値）
        _, binary = cv2.threshold(
            image.astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        return binary.astype(np.float32)

    def center_image(self, image: np.ndarray) -> np.ndarray:
        """
        画像を重心基準で中央に配置

        Args:
            image (np.ndarray): 入力画像

        Returns:
            np.ndarray: 中央配置された画像
        """
        # 重心を計算
        moments = cv2.moments(image.astype(np.uint8))

        if moments["m00"] == 0:
            # 空の画像の場合はそのまま返す
            return image

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])

        # 中央からのオフセット
        height, width = image.shape
        offset_x = width // 2 - cx
        offset_y = height // 2 - cy

        # 平行移動行列
        M = np.float32([[1, 0, offset_x], [0, 1, offset_y]])

        # アフィン変換
        centered = cv2.warpAffine(
            image,
            M,
            (width, height),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,
        )

        return centered

    def augment_data(
        self,
        image: np.ndarray,
        rotation_range: float = 5.0,
        scale_range: Tuple[float, float] = (0.95, 1.05),
        translation_range: float = 0.05,
        noise_std: float = 0.01,
        apply_probability: float = 0.5,
    ) -> np.ndarray:
        """
        データ拡張

        Args:
            image (np.ndarray): 入力画像 [0, 1]
            rotation_range (float): 回転角度範囲（度）
            scale_range (Tuple[float, float]): スケール範囲
            translation_range (float): 平行移動範囲（画像サイズの比率）
            noise_std (float): ノイズの標準偏差
            apply_probability (float): 各拡張を適用する確率

        Returns:
            np.ndarray: 拡張された画像
        """
        img = image.copy()

        # 回転
        if np.random.rand() < apply_probability:
            angle = np.random.uniform(-rotation_range, rotation_range)
            img = self._rotate(img, angle)

        # スケール
        if np.random.rand() < apply_probability:
            scale = np.random.uniform(scale_range[0], scale_range[1])
            img = self._scale(img, scale)

        # 平行移動
        if np.random.rand() < apply_probability:
            tx = np.random.uniform(-translation_range, translation_range)
            ty = np.random.uniform(-translation_range, translation_range)
            img = self._translate(img, tx, ty)

        # ノイズ
        if np.random.rand() < apply_probability:
            img = self._add_noise(img, noise_std)

        # [0, 1] 範囲にクリップ
        img = np.clip(img, 0.0, 1.0)

        return img

    def _rotate(self, image: np.ndarray, angle: float) -> np.ndarray:
        """画像を回転"""
        center = tuple(np.array(image.shape[:2]) / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            image, M, image.shape[:2][::-1], borderMode=cv2.BORDER_CONSTANT, borderValue=1.0
        )
        return rotated

    def _scale(self, image: np.ndarray, scale: float) -> np.ndarray:
        """画像をスケール"""
        height, width = image.shape
        new_size = int(height * scale)

        # スケール変更
        scaled = cv2.resize(image, (new_size, new_size), interpolation=cv2.INTER_LINEAR)

        # 元のサイズに戻す（中央配置）
        if new_size > height:
            # クロップ
            start = (new_size - height) // 2
            result = scaled[start : start + height, start : start + width]
        else:
            # パディング
            result = np.ones((height, width), dtype=np.float32)
            start = (height - new_size) // 2
            result[start : start + new_size, start : start + new_size] = scaled

        return result

    def _translate(self, image: np.ndarray, tx: float, ty: float) -> np.ndarray:
        """画像を平行移動"""
        height, width = image.shape
        tx_pixels = int(tx * width)
        ty_pixels = int(ty * height)

        M = np.float32([[1, 0, tx_pixels], [0, 1, ty_pixels]])
        translated = cv2.warpAffine(
            image, M, (width, height), borderMode=cv2.BORDER_CONSTANT, borderValue=1.0
        )
        return translated

    def _add_noise(self, image: np.ndarray, noise_std: float) -> np.ndarray:
        """ガウシアンノイズを追加"""
        noise = np.random.normal(0, noise_std, image.shape)
        noisy = image + noise
        return noisy


def test_preprocessor():
    """テスト関数"""
    import matplotlib.pyplot as plt
    from font_parser import FontParser

    # テスト用フォント
    test_font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:\\Windows\\Fonts\\msgothic.ttc",
    ]

    font_path = None
    for path in test_font_paths:
        import os

        if os.path.exists(path):
            font_path = path
            break

    if font_path is None:
        print("テスト用フォントが見つかりません")
        return

    # フォントパーサーでサンプル画像を生成
    parser = FontParser(font_path, image_size=128)
    test_char = "あ"

    available = parser.get_available_characters(test_char)
    if not available:
        test_char = "A"
        available = parser.get_available_characters(test_char)

    if not available:
        print("テスト文字が見つかりません")
        return

    original = parser.render_character(test_char)

    # 前処理
    preprocessor = Preprocessor(image_size=128)
    normalized = preprocessor.normalize_image(original, center=True, invert=True)

    # データ拡張
    augmented_images = [
        preprocessor.augment_data(normalized) for _ in range(8)
    ]

    # 可視化
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))

    axes[0, 0].imshow(original, cmap="gray")
    axes[0, 0].set_title("Original")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(normalized, cmap="gray")
    axes[0, 1].set_title("Normalized")
    axes[0, 1].axis("off")

    for idx, aug_img in enumerate(augmented_images):
        row = (idx + 2) // 5
        col = (idx + 2) % 5
        axes[row, col].imshow(aug_img, cmap="gray")
        axes[row, col].set_title(f"Augmented {idx + 1}")
        axes[row, col].axis("off")

    plt.tight_layout()
    plt.savefig("preprocessing_test.png")
    print("Test image saved: preprocessing_test.png")

    parser.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_preprocessor()

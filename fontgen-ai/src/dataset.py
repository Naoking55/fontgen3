#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
データセット - PyTorch Dataset実装
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import logging

from preprocessing import Preprocessor

logger = logging.getLogger(__name__)


class FontDataset(Dataset):
    """
    フォントデータセット

    前処理済みの文字画像を読み込むPyTorch Dataset
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: int = 128,
        augmentation: bool = True,
        augmentation_params: Optional[Dict] = None,
    ):
        """
        Args:
            data_dir (str): データディレクトリ
            split (str): "train", "val", "test"
            image_size (int): 画像サイズ
            augmentation (bool): データ拡張を行うか
            augmentation_params (Dict, optional): データ拡張パラメータ
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.image_size = image_size
        self.augmentation = augmentation and (split == "train")

        # 前処理器
        self.preprocessor = Preprocessor(image_size=image_size)

        # データ拡張パラメータ
        self.aug_params = augmentation_params or {
            "rotation_range": 5.0,
            "scale_range": (0.95, 1.05),
            "translation_range": 0.05,
            "noise_std": 0.01,
            "apply_probability": 0.5,
        }

        # メタデータ読み込み
        self.metadata_path = self.data_dir / "metadata.json"
        self.metadata = self._load_metadata()

        # データインデックス構築
        self.data_index = self._build_index()

        logger.info(
            f"Loaded {len(self.data_index)} samples from {split} split"
        )

    def _load_metadata(self) -> Dict:
        """メタデータを読み込む"""
        if not self.metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {self.metadata_path}")

        with open(self.metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _build_index(self) -> List[Dict]:
        """データインデックスを構築"""
        index = []
        split_dir = self.data_dir / self.split

        if not split_dir.exists():
            logger.warning(f"Split directory not found: {split_dir}")
            return index

        # フォントごとにディレクトリを探索
        for font_dir in sorted(split_dir.iterdir()):
            if not font_dir.is_dir():
                continue

            font_id = font_dir.name

            # 文字画像ファイルを探索
            for image_path in sorted(font_dir.glob("*.png")):
                char = image_path.stem  # ファイル名が文字

                index.append(
                    {
                        "image_path": str(image_path),
                        "font_id": font_id,
                        "character": char,
                    }
                )

        return index

    def __len__(self) -> int:
        return len(self.data_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        データを取得

        Returns:
            Dict containing:
                - image (Tensor): 画像 (1, H, W), [0, 1]
                - font_id (int): フォントID
                - char_id (int): 文字ID
        """
        item = self.data_index[idx]

        # 画像読み込み
        image = Image.open(item["image_path"]).convert("L")
        image = np.array(image, dtype=np.float32) / 255.0

        # 画像サイズ確認
        if image.shape[0] != self.image_size or image.shape[1] != self.image_size:
            image = np.array(
                Image.fromarray((image * 255).astype(np.uint8)).resize(
                    (self.image_size, self.image_size), Image.BILINEAR
                )
            ).astype(np.float32) / 255.0

        # データ拡張
        if self.augmentation:
            image = self.preprocessor.augment_data(image, **self.aug_params)

        # Tensorに変換 (C, H, W)
        image_tensor = torch.from_numpy(image).unsqueeze(0)  # (1, H, W)

        # フォントIDと文字IDを数値化
        font_id = self._font_to_id(item["font_id"])
        char_id = self._char_to_id(item["character"])

        return {
            "image": image_tensor,
            "font_id": font_id,
            "char_id": char_id,
            "font_name": item["font_id"],
            "character": item["character"],
        }

    def _font_to_id(self, font_name: str) -> int:
        """フォント名を数値IDに変換"""
        if "fonts" not in self.metadata:
            return 0

        fonts = self.metadata["fonts"]
        if font_name in fonts:
            return fonts.index(font_name)
        return 0

    def _char_to_id(self, char: str) -> int:
        """文字を数値IDに変換"""
        if "characters" not in self.metadata:
            return 0

        characters = self.metadata["characters"]
        if char in characters:
            return characters.index(char)
        return 0


class FontDataModule:
    """
    データモジュール - データローダーを管理
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int = 64,
        num_workers: int = 4,
        image_size: int = 128,
        augmentation: bool = True,
        augmentation_params: Optional[Dict] = None,
    ):
        """
        Args:
            data_dir (str): データディレクトリ
            batch_size (int): バッチサイズ
            num_workers (int): ワーカー数
            image_size (int): 画像サイズ
            augmentation (bool): データ拡張
            augmentation_params (Dict, optional): データ拡張パラメータ
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.augmentation = augmentation
        self.augmentation_params = augmentation_params

        # データセット
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def setup(self):
        """データセットをセットアップ"""
        # Train
        self.train_dataset = FontDataset(
            self.data_dir,
            split="train",
            image_size=self.image_size,
            augmentation=self.augmentation,
            augmentation_params=self.augmentation_params,
        )

        # Validation
        self.val_dataset = FontDataset(
            self.data_dir,
            split="val",
            image_size=self.image_size,
            augmentation=False,
        )

        # Test
        test_dir = Path(self.data_dir) / "test"
        if test_dir.exists():
            self.test_dataset = FontDataset(
                self.data_dir,
                split="test",
                image_size=self.image_size,
                augmentation=False,
            )

    def train_dataloader(self):
        """訓練データローダー"""
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        """検証データローダー"""
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        """テストデータローダー"""
        if self.test_dataset is None:
            return None

        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )


def test_dataset():
    """テスト関数"""
    import matplotlib.pyplot as plt

    # データディレクトリ（仮）
    data_dir = "../data/processed"

    try:
        # データモジュール作成
        data_module = FontDataModule(
            data_dir=data_dir, batch_size=16, num_workers=0, image_size=128
        )

        data_module.setup()

        # データローダー取得
        train_loader = data_module.train_dataloader()

        # 1バッチ取得
        batch = next(iter(train_loader))

        print(f"Batch keys: {batch.keys()}")
        print(f"Image shape: {batch['image'].shape}")
        print(f"Font IDs: {batch['font_id'][:5]}")
        print(f"Char IDs: {batch['char_id'][:5]}")
        print(f"Characters: {batch['character'][:5]}")

        # 可視化
        images = batch["image"][:16].numpy()
        characters = batch["character"][:16]

        fig, axes = plt.subplots(4, 4, figsize=(12, 12))
        axes = axes.flatten()

        for idx, (img, char) in enumerate(zip(images, characters)):
            axes[idx].imshow(img[0], cmap="gray")
            axes[idx].set_title(f"'{char}'")
            axes[idx].axis("off")

        plt.tight_layout()
        plt.savefig("dataset_test.png")
        print("Test image saved: dataset_test.png")

    except Exception as e:
        print(f"Test failed: {e}")
        print("データが準備されていない可能性があります。")
        print("python cli/prepare_data.py を実行してください。")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataset()

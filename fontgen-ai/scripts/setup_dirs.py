#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ディレクトリ構造セットアップスクリプト
"""

import os
from pathlib import Path


def create_directories():
    """必要なディレクトリを作成"""

    # ベースディレクトリ
    base_dir = Path(__file__).parent.parent

    # 作成するディレクトリリスト
    directories = [
        "data/fonts",
        "data/processed/train",
        "data/processed/val",
        "data/processed/test",
        "data/skeleton_db",
        "models/pretrained",
        "output/samples",
        "output/fonts",
        "output/glyphs",
        "logs",
    ]

    print("📁 ディレクトリを作成しています...")

    for directory in directories:
        dir_path = base_dir / directory
        dir_path.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ {directory}")

    # .gitkeep を作成（空のディレクトリをGitで管理するため）
    gitkeep_dirs = [
        "data/fonts",
        "data/processed/train",
        "data/processed/val",
        "data/processed/test",
        "data/skeleton_db",
        "models/pretrained",
        "output/samples",
        "output/fonts",
        "output/glyphs",
    ]

    print("\n📝 .gitkeep ファイルを作成しています...")

    for directory in gitkeep_dirs:
        gitkeep_path = base_dir / directory / ".gitkeep"
        gitkeep_path.touch(exist_ok=True)
        print(f"  ✓ {directory}/.gitkeep")

    print("\n✅ セットアップ完了!")
    print("\n次のステップ:")
    print("  1. フォントファイルを data/fonts/ に配置")
    print("  2. python cli/prepare_data.py でデータを準備")
    print("  3. python cli/train.py で学習を開始")


if __name__ == "__main__":
    create_directories()

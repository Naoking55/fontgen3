#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
FGCLCGYM.TTC (TrueType Collection) の漢字サポートを確認
"""

from PIL import Image, ImageDraw, ImageFont
import sys
sys.path.append('.')

from src.char_sets import KANJI_COMMON_LEVEL1, KANJI_COMMON_LEVEL2

def check_font_support(font_path, font_index=0):
    """
    TTC フォントの漢字サポートを確認

    Args:
        font_path: フォントファイルのパス
        font_index: TTC内のフォントインデックス（デフォルト: 0）
    """
    try:
        # TTC フォントは index パラメータで指定
        font = ImageFont.truetype(font_path, 64, index=font_index)
    except Exception as e:
        print(f"Error loading font (index {font_index}): {e}")
        return None

    all_kanji = KANJI_COMMON_LEVEL1 + KANJI_COMMON_LEVEL2

    supported = []
    unsupported = []

    for char in all_kanji:
        try:
            # getbboxでサイズを取得できればサポートされている
            bbox = font.getbbox(char)
            if bbox and (bbox[2] - bbox[0]) > 0:  # 幅が0より大きい
                supported.append(char)
            else:
                unsupported.append(char)
        except Exception:
            unsupported.append(char)

    return {
        'supported': supported,
        'unsupported': unsupported,
        'support_rate': len(supported) / len(all_kanji) * 100
    }

print("=" * 80)
print("FGCLCGYM.TTC 漢字サポート確認")
print("=" * 80)

font_path = 'data/fonts/FGCLCGYM.TTC'

# TTC は複数のフォントを含む可能性があるので、最初のいくつかをチェック
for font_index in range(5):  # 最大5つまでチェック
    print(f"\n--- Font Index {font_index} ---")
    result = check_font_support(font_path, font_index)

    if result is None:
        print(f"Index {font_index} does not exist or failed to load")
        if font_index == 0:
            print("ERROR: Cannot load any font from TTC file!")
            sys.exit(1)
        break

    print(f"Kanji Level 1 (101文字): {len([c for c in KANJI_COMMON_LEVEL1 if c in result['supported']])}/101")
    print(f"Kanji Level 2 (96文字): {len([c for c in KANJI_COMMON_LEVEL2 if c in result['supported']])}/96")
    print(f"Total: {len(result['supported'])}/197 ({result['support_rate']:.1f}%)")

    if result['unsupported']:
        print(f"Unsupported: {len(result['unsupported'])} characters")
        if len(result['unsupported']) <= 20:
            print(f"  {' '.join(result['unsupported'])}")
    else:
        print("✅ All kanji characters supported!")

print("\n" + "=" * 80)
print("推奨: Font Index 0 を使用")
print("=" * 80)

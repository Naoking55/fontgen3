#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
文字セット定義
"""

# ひらがな (83文字)
HIRAGANA = (
    "あいうえおかきくけこさしすせそたちつてとなにぬねの"
    "はひふへほまみむめもやゆよらりるれろわをん"
    "がぎぐげござじずぜぞだぢづでどばびぶべぼぱぴぷぺぽ"
    "ぁぃぅぇぉゃゅょっゎ"
)

# カタカナ (86文字)
KATAKANA = (
    "アイウエオカキクケコサシスセソタチツテトナニヌネノ"
    "ハヒフヘホマミムメモヤユヨラリルレロワヲンー"
    "ガギグゲゴザジズゼゾダヂヅデドバビブベボパピプペポ"
    "ァィゥェォヵヶャュョッヮヴ"
)

# 英字（大文字）
ALPHABET_UPPER = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

# 英字（小文字）
ALPHABET_LOWER = "abcdefghijklmnopqrstuvwxyz"

# 数字
NUMBERS = "0123456789"

# 基本記号
SYMBOLS_BASIC = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"

# 日本語記号
SYMBOLS_JP = "。、・「」『』（）［］｛｝〈〉《》【】〔〕〘〙〚〛"

# 常用漢字（一部、学習時間の関係で必要に応じて追加）
KANJI_COMMON_LEVEL1 = (
    "日一国会人年大十二本中長出三同時政事自行社見月分議民連対部全"
    "市内四五間理金子定今回新場報以外開手力学代明実天入下地員立生"
    "作山成表等化法方問主意発体制度度路教世界用済当合取最初現解経"
    "加前米動物東道進問題語"
)

# よく使う漢字（200文字程度、必要に応じて拡張）
KANJI_COMMON_LEVEL2 = (
    "山川田中村木林森水火土石金銀銅鉄空雨雪雲風花草木森川海岸島岩"
    "谷峠池河湖畑園田野原高低広狭深浅遠近長短大小多少軽重明暗新古"
    "良悪美醜強弱速遅熱冷暑寒春夏秋冬朝昼夜晩早遅今昔前後左右上下"
    "東西南北内外"
)

# 文字セット定義
CHAR_SETS = {
    "hiragana": HIRAGANA,
    "katakana": KATAKANA,
    "alphabet_upper": ALPHABET_UPPER,
    "alphabet_lower": ALPHABET_LOWER,
    "alphabet": ALPHABET_UPPER + ALPHABET_LOWER,
    "numbers": NUMBERS,
    "symbols_basic": SYMBOLS_BASIC,
    "symbols_jp": SYMBOLS_JP,
    "kanji_level1": KANJI_COMMON_LEVEL1,
    "kanji_level2": KANJI_COMMON_LEVEL2,
    "kanji_joyo": KANJI_COMMON_LEVEL1 + KANJI_COMMON_LEVEL2,
    "basic": HIRAGANA + KATAKANA + ALPHABET_UPPER + ALPHABET_LOWER + NUMBERS,
}


def get_characters(charset_names):
    """
    文字セット名から文字列を取得

    Args:
        charset_names (str or list): 文字セット名（カンマ区切りまたはリスト）

    Returns:
        str: 文字列

    Examples:
        >>> get_characters("hiragana")
        'あいうえお...'

        >>> get_characters("hiragana,katakana")
        'あいうえお...アイウエオ...'

        >>> get_characters(["hiragana", "katakana"])
        'あいうえお...アイウエオ...'
    """
    if isinstance(charset_names, str):
        charset_names = [name.strip() for name in charset_names.split(",")]

    characters = ""
    for name in charset_names:
        if name in CHAR_SETS:
            characters += CHAR_SETS[name]
        else:
            raise ValueError(f"Unknown character set: {name}")

    # 重複を削除
    return "".join(sorted(set(characters)))


def get_available_charsets():
    """利用可能な文字セット一覧を取得"""
    return list(CHAR_SETS.keys())


if __name__ == "__main__":
    # テスト
    print("利用可能な文字セット:")
    for name in get_available_charsets():
        chars = CHAR_SETS[name]
        print(f"  {name}: {len(chars)}文字 - {chars[:20]}...")

    print("\n文字セット取得テスト:")
    test_chars = get_characters("hiragana,katakana")
    print(f"  hiragana,katakana: {len(test_chars)}文字")

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
偏旁抽出ツール - 完全版 GUI v2.9 (2025-11-09)
動的境界検出アルゴリズム統合 + 2048x2048解像度対応
"""

import os
import sys
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, scrolledtext
from pathlib import Path
from PIL import Image, ImageFont, ImageDraw, ImageOps, ImageTk
import threading
import math  # [ADD] 2025-10-10: 補間計算用
import numpy as np  # [ADD] 2025-11-09: 動的境界検出用
from typing import List, Tuple, Dict, Optional

# macOS対策
os.environ['TK_SILENCE_DEPRECATION'] = '1'

# ============================================================
# [CONFIG] 設定 (2025-11-09)
# ============================================================

class Config:
    """偏旁抽出ツールの設定"""

    # ===== 動的境界検出設定 (2025-11-09) =====
    DYNAMIC_BOUNDARY_DETECTION = True  # 動的境界検出を有効にする
    BOUNDARY_SEARCH_RANGE_LR = (0.25, 0.75)  # 左右分割の探索範囲
    BOUNDARY_SEARCH_RANGE_TB = (0.25, 0.75)  # 上下分割の探索範囲
    BOUNDARY_SCAN_STEP = 0.02  # スキャンステップ（2%刻み）
    BINARY_THRESHOLD = 200  # 二値化閾値

    # ===== レンダリング設定 (2025-11-09) =====
    RENDER_SIZE = 2048  # 文字レンダリング解像度 (2048x2048)

# ============================================================
# [BLOCK1-BEGIN] 偏旁カタログ (2025-10-10)
# ============================================================
"""
■ サンプル文字の修正方法

PARTS_CATALOG の各エントリの "sample" を変更してください。

例：
"ひへん": {"char": "火", "sample": "灯", "split": "left", "ratio": 0.35},
                                    ↑ ここを変更

修正が必要な例：
- 偏が明確でない文字（例：「炎」→「灯」に変更済み）
- 旁が明確でない文字
- 抽出結果が不適切な文字

修正後、再度「抽出開始」を実行すると、新しいサンプル文字で抽出されます。
"""

PARTS_CATALOG = {
    # ===== 偏（へん）: 左側配置のみ =====
    "hen": {
        # 人に関する偏
        "にんべん": {"char": "亻", "sample": "仁", "split": "left", "ratio": 0.35, "alternatives": ["人", "他", "住", "作", "使"]},
        "ぎょうにんべん": {"char": "彳", "sample": "行", "split": "left", "ratio": 0.3, "alternatives": ["往", "待", "役"]},
        "りっしんべん": {"char": "忄", "sample": "情", "split": "left", "ratio": 0.3, "alternatives": ["性", "怖", "悩", "快"]},

        # 手・動作に関する偏
        "てへん": {"char": "扌", "sample": "持", "split": "left", "ratio": 0.35, "alternatives": ["手", "打", "投", "押", "拾"]},
        "さんずい": {"char": "氵", "sample": "海", "split": "left", "ratio": 0.3, "alternatives": ["江", "河", "波", "池", "湖"]},

        # 言葉に関する偏
        "ごんべん": {"char": "訁", "sample": "語", "split": "left", "ratio": 0.4, "alternatives": ["話", "説", "訳", "記", "論"]},
        "くちへん": {"char": "口", "sample": "呼", "split": "left", "ratio": 0.4, "alternatives": ["味", "吸", "鳴", "唱"]},

        # 木・植物に関する偏
        "きへん": {"char": "木", "sample": "林", "split": "left", "ratio": 0.4, "alternatives": ["村", "森", "桜", "松"]},
        "のぎへん": {"char": "禾", "sample": "秋", "split": "left", "ratio": 0.4, "alternatives": ["和", "私", "秀"]},
        "たけへん": {"char": "⺮", "sample": "竹", "split": "left", "ratio": 0.4, "alternatives": ["笑", "箱", "管"]},

        # 金属・鉱物に関する偏
        "かねへん": {"char": "金", "sample": "鉄", "split": "left", "ratio": 0.45, "alternatives": ["銅", "銀", "鋼", "鉱"]},
        "いしへん": {"char": "石", "sample": "砂", "split": "left", "ratio": 0.4, "alternatives": ["岩", "研", "硬", "確"]},

        # 糸・衣に関する偏
        "いとへん": {"char": "糸", "sample": "結", "split": "left", "ratio": 0.45, "alternatives": ["線", "紙", "級", "紅"]},
        "ころもへん": {"char": "衤", "sample": "被", "split": "left", "ratio": 0.35, "alternatives": ["袖", "裕", "補"]},

        # 食べ物に関する偏
        "しょくへん": {"char": "飠", "sample": "館", "split": "left", "ratio": 0.4, "alternatives": ["飯", "飲", "飾"]},

        # 動物に関する偏
        "けものへん": {"char": "犭", "sample": "狼", "split": "left", "ratio": 0.35, "alternatives": ["犬", "猫", "狐", "狩"]},
        "うおへん": {"char": "魚", "sample": "鮮", "split": "left", "ratio": 0.5, "alternatives": ["鯨", "鮭", "鯛"]},
        "むしへん": {"char": "虫", "sample": "蛇", "split": "left", "ratio": 0.4, "alternatives": ["蚊", "蝶", "蜂"]},

        # 土・自然に関する偏
        "つちへん": {"char": "土", "sample": "城", "split": "left", "ratio": 0.35, "alternatives": ["地", "坂", "堂"]},
        "やまへん": {"char": "山", "sample": "峰", "split": "left", "ratio": 0.4, "alternatives": ["岩", "崎", "岳"]},

        # 火・水に関する偏
        "ひへん": {"char": "火", "sample": "灯", "split": "left", "ratio": 0.35, "alternatives": ["焼", "炎", "煙"]},
        "にすい": {"char": "冫", "sample": "冷", "split": "left", "ratio": 0.25, "alternatives": ["凍", "次"]},

        # 体の部位に関する偏
        "にくづき": {"char": "月", "sample": "胸", "split": "left", "ratio": 0.4, "alternatives": ["胴", "腕", "脳"]},
        "ほねへん": {"char": "骨", "sample": "骸", "split": "left", "ratio": 0.5},
        "めへん": {"char": "目", "sample": "眼", "split": "left", "ratio": 0.4, "alternatives": ["眠", "瞳"]},
        "みみへん": {"char": "耳", "sample": "聴", "split": "left", "ratio": 0.4, "alternatives": ["聞", "職"]},
        "てへん2": {"char": "手", "sample": "拳", "split": "left", "ratio": 0.4},

        # その他の重要な偏
        "おんなへん": {"char": "女", "sample": "妹", "split": "left", "ratio": 0.4, "alternatives": ["姉", "娘", "嫁"]},
        "こざとへん": {"char": "阝", "sample": "防", "split": "left", "ratio": 0.3, "alternatives": ["陽", "阪", "院"]},
        "しめすへん": {"char": "礻", "sample": "祈", "split": "left", "ratio": 0.35, "alternatives": ["神", "祝", "祭"]},

        # 追加の偏
        "ゆみへん": {"char": "弓", "sample": "張", "split": "left", "ratio": 0.35},
        "かわへん": {"char": "革", "sample": "靴", "split": "left", "ratio": 0.45},
        "かいへん": {"char": "貝", "sample": "販", "split": "left", "ratio": 0.4, "alternatives": ["財", "貨", "貧"]},
        "あしへん": {"char": "足", "sample": "跡", "split": "left", "ratio": 0.45, "alternatives": ["跳", "蹴", "踏"]},
        "くるまへん": {"char": "車", "sample": "輪", "split": "left", "ratio": 0.45, "alternatives": ["軽", "転"]},
        "さけのとり": {"char": "酉", "sample": "配", "split": "left", "ratio": 0.4, "alternatives": ["酒", "酔"]},
        "うしへん": {"char": "牛", "sample": "牡", "split": "left", "ratio": 0.4, "alternatives": ["物", "特"]},
    },
    
    # ===== 旁（つくり）: 右側配置のみ =====
    "tsukuri": {
        # 基本的な旁
        "おおざと": {"char": "阝", "sample": "部", "split": "right", "ratio": 0.7, "alternatives": ["郎", "都", "郭"]},
        "りっとう": {"char": "刂", "sample": "則", "split": "right", "ratio": 0.7, "alternatives": ["刊", "判", "削"]},
        "ちから": {"char": "力", "sample": "助", "split": "right", "ratio": 0.65, "alternatives": ["動", "努", "勇"]},
        "おおがい": {"char": "頁", "sample": "順", "split": "right", "ratio": 0.55, "alternatives": ["頭", "顔", "頬"]},
        "ぼくづくり": {"char": "攵", "sample": "政", "split": "right", "ratio": 0.65, "alternatives": ["救", "敗", "教"]},

        # 鳥・動物系
        "ふるとり": {"char": "隹", "sample": "雑", "split": "right", "ratio": 0.6, "alternatives": ["集", "雀", "難"]},
        "とり": {"char": "鳥", "sample": "鳩", "split": "right", "ratio": 0.55, "alternatives": ["鳴", "鶏"]},
        "うま": {"char": "馬", "sample": "駅", "split": "right", "ratio": 0.55, "alternatives": ["駆", "騎", "験"]},

        # 武器・道具系
        "きづくり": {"char": "斤", "sample": "新", "split": "right", "ratio": 0.65, "alternatives": ["断", "斬"]},
        "ほこづくり": {"char": "戈", "sample": "成", "split": "right", "ratio": 0.6, "alternatives": ["戦", "戯"]},
        "かたな": {"char": "刀", "sample": "切", "split": "right", "ratio": 0.65, "alternatives": ["分", "列"]},
        "ほこ": {"char": "殳", "sample": "殴", "split": "right", "ratio": 0.6},

        # 自然・天体系
        "つき": {"char": "月", "sample": "朝", "split": "right", "ratio": 0.6, "alternatives": ["期", "明"]},
        "ひ": {"char": "日", "sample": "旧", "split": "right", "ratio": 0.6, "alternatives": ["時", "昭"]},

        # 体・感覚系
        "みる": {"char": "見", "sample": "規", "split": "right", "ratio": 0.6, "alternatives": ["視", "覧", "観"]},
        "おと": {"char": "音", "sample": "韻", "split": "right", "ratio": 0.55, "alternatives": ["章"]},
        "あくび": {"char": "欠", "sample": "歌", "split": "right", "ratio": 0.65, "alternatives": ["次", "欧"]},

        # 食物・植物系
        "むぎ": {"char": "麦", "sample": "麺", "split": "right", "ratio": 0.55, "alternatives": ["麹"]},

        # その他
        "おに": {"char": "鬼", "sample": "魅", "split": "right", "ratio": 0.55, "alternatives": ["魂", "魔"]},
        "ふでづくり": {"char": "聿", "sample": "律", "split": "right", "ratio": 0.6, "alternatives": ["建", "筆"]},
        "よう": {"char": "羊", "sample": "養", "split": "right", "ratio": 0.6, "alternatives": ["美", "義"]},
        "おおがね": {"char": "金", "sample": "鉱", "split": "right", "ratio": 0.6},
        "ぼく": {"char": "攴", "sample": "牧", "split": "right", "ratio": 0.65, "alternatives": ["収"]},

        # 形・装飾系
        "さんづくり": {"char": "彡", "sample": "彩", "split": "right", "ratio": 0.65, "alternatives": ["形", "影", "彫"]},
        "のぶん": {"char": "文", "sample": "紋", "split": "right", "ratio": 0.6, "alternatives": ["斑"]},

        # 自然・方向系
        "かぜ": {"char": "風", "sample": "颯", "split": "right", "ratio": 0.55, "alternatives": ["飄"]},
        "ほう": {"char": "方", "sample": "族", "split": "right", "ratio": 0.65, "alternatives": ["旗", "旋", "施"]},
    },
    
    # ===== 冠（かんむり）: 上側配置 =====
    "kanmuri": {
        # 植物に関する冠
        "くさかんむり": {"char": "艹", "sample": "花", "split": "top", "ratio": 0.3, "alternatives": ["草", "茶", "英", "菜"]},
        "たけかんむり": {"char": "⺮", "sample": "笑", "split": "top", "ratio": 0.35, "alternatives": ["竹", "筆", "箱"]},

        # 自然・天候に関する冠
        "あめかんむり": {"char": "雨", "sample": "雷", "split": "top", "ratio": 0.4, "alternatives": ["雪", "雲", "電"]},
        "やまかんむり": {"char": "山", "sample": "岩", "split": "top", "ratio": 0.35, "alternatives": ["嵩", "嶺", "峠"]},

        # 建物・覆うものに関する冠
        "うかんむり": {"char": "宀", "sample": "宇", "split": "top", "ratio": 0.25, "alternatives": ["宙", "宝", "家", "安"]},
        "あなかんむり": {"char": "穴", "sample": "空", "split": "top", "ratio": 0.35, "alternatives": ["究", "窓"]},
        "わかんむり": {"char": "冖", "sample": "冠", "split": "top", "ratio": 0.25, "alternatives": ["冬", "冷", "写", "冗"]},

        # 網・枠に関する冠
        "あみがしら": {"char": "罒", "sample": "買", "split": "top", "ratio": 0.3, "alternatives": ["罪", "置"]},
        "よこめ": {"char": "⺫", "sample": "置", "split": "top", "ratio": 0.3},

        # 形・記号的な冠
        "なべぶた": {"char": "亠", "sample": "市", "split": "top", "ratio": 0.2, "alternatives": ["京", "享", "亡"]},
        "はちがしら": {"char": "八", "sample": "公", "split": "top", "ratio": 0.25, "alternatives": ["六", "共"]},
        "ひとやね": {"char": "𠆢", "sample": "会", "split": "top", "ratio": 0.2, "alternatives": ["合", "令"]},
        "つめかんむり": {"char": "爫", "sample": "受", "split": "top", "ratio": 0.3, "alternatives": ["愛", "採"]},

        # その他の冠
        "だいかんむり": {"char": "大", "sample": "奇", "split": "top", "ratio": 0.3, "alternatives": ["奈", "奔"]},
        "ひとがしら": {"char": "人", "sample": "介", "split": "top", "ratio": 0.25, "alternatives": ["今", "傘"]},
        "おいがしら": {"char": "老", "sample": "考", "split": "top", "ratio": 0.35},
        "ちいさい": {"char": "小", "sample": "尖", "split": "top", "ratio": 0.3},
        "そうにょう": {"char": "⺍", "sample": "学", "split": "top", "ratio": 0.25, "alternatives": ["党", "堂"]},
        "おおいかんむり": {"char": "覀", "sample": "要", "split": "top", "ratio": 0.35, "alternatives": ["覆"]},
    },
    
    # ===== 脚（あし）: 下側配置 =====
    "ashi": {
        "こころ": {"char": "心", "sample": "念", "split": "bottom", "ratio": 0.65, "alternatives": ["恋", "慕", "思"]},
        "れっか": {"char": "灬", "sample": "熱", "split": "bottom", "ratio": 0.75, "alternatives": ["煮", "点", "煎", "黒"]},
        "ひとあし": {"char": "儿", "sample": "児", "split": "bottom", "ratio": 0.7, "alternatives": ["兄", "元", "光"]},
        "さら": {"char": "皿", "sample": "盛", "split": "bottom", "ratio": 0.7, "alternatives": ["益", "盗"]},
        "したみず": {"char": "水", "sample": "泰", "split": "bottom", "ratio": 0.7},
        "こがい": {"char": "貝", "sample": "買", "split": "bottom", "ratio": 0.65, "alternatives": ["貨", "賀", "貸"]},
        "き": {"char": "夂", "sample": "条", "split": "bottom", "ratio": 0.7, "alternatives": ["桑", "冬"]},
        "くち": {"char": "口", "sample": "含", "split": "bottom", "ratio": 0.7, "alternatives": ["否", "杏"]},
        "ころも": {"char": "衣", "sample": "製", "split": "bottom", "ratio": 0.65, "alternatives": ["装", "襲", "裏"]},
        "したごころ": {"char": "心", "sample": "恭", "split": "bottom", "ratio": 0.7, "alternatives": ["慕", "忠"]},
        "つち": {"char": "土", "sample": "墓", "split": "bottom", "ratio": 0.7, "alternatives": ["型", "塞"]},
        "いわく": {"char": "曰", "sample": "書", "split": "bottom", "ratio": 0.7, "alternatives": ["替", "曹"]},
    },
    
    # ===== 繞（にょう）: 左下を囲む =====
    "nyou": {
        "しんにょう": {"char": "辶", "sample": "近", "split": "left_bottom", "ratio": 0.6, "alternatives": ["道", "進", "通", "遠"]},
        "えんにょう": {"char": "廴", "sample": "延", "split": "left_bottom", "ratio": 0.55, "alternatives": ["建", "廷"]},
        "そうにょう": {"char": "走", "sample": "起", "split": "left_bottom", "ratio": 0.65},
        "きにょう": {"char": "鬼", "sample": "魅", "split": "left_bottom", "ratio": 0.6, "alternatives": ["魁", "魂"]},
        "ばくにょう": {"char": "麦", "sample": "麺", "split": "left_bottom", "ratio": 0.6, "alternatives": ["麹"]},
        "ばくにょう2": {"char": "麥", "sample": "麩", "split": "left_bottom", "ratio": 0.6, "alternatives": ["麭"]},
    },
    
    # ===== 垂（たれ）: 上から左へ垂れる =====
    "tare": {
        "がんだれ": {"char": "厂", "sample": "原", "split": "top_left", "ratio": 0.5, "alternatives": ["厚", "雁", "厳"]},
        "まだれ": {"char": "广", "sample": "広", "split": "top_left", "ratio": 0.45, "alternatives": ["店", "座", "庁"]},
        "やまいだれ": {"char": "疒", "sample": "痛", "split": "top_left", "ratio": 0.45, "alternatives": ["病", "痩", "療"]},
        "しかばねだれ": {"char": "尸", "sample": "局", "split": "top_left", "ratio": 0.45, "alternatives": ["屋", "屍", "居"]},
        "とだれ": {"char": "戶", "sample": "戻", "split": "top_left", "ratio": 0.5},
    },
    
    # ===== 構（かまえ）: 周りを囲む =====
    "kamae": {
        "もんがまえ": {"char": "門", "sample": "間", "split": "frame", "ratio": 0.5, "alternatives": ["門", "問", "開"]},
        "くにがまえ": {"char": "囗", "sample": "国", "split": "frame", "ratio": 0.5, "alternatives": ["四", "回", "囲"]},
        "ぎょうがまえ": {"char": "行", "sample": "衛", "split": "frame", "ratio": 0.5},
        "かくしがまえ": {"char": "匸", "sample": "匹", "split": "frame", "ratio": 0.5},
        "はこがまえ": {"char": "匚", "sample": "匠", "split": "frame", "ratio": 0.45, "alternatives": ["区", "医"]},
        "けいがまえ": {"char": "冂", "sample": "円", "split": "frame", "ratio": 0.45, "alternatives": ["冊", "周"]},
        "とがまえ": {"char": "戸", "sample": "房", "split": "frame", "ratio": 0.5, "alternatives": ["扉", "所"]},
        "かぜがまえ": {"char": "風", "sample": "凪", "split": "frame", "ratio": 0.5},
        "ほこがまえ": {"char": "戈", "sample": "成", "split": "frame", "ratio": 0.5, "alternatives": ["戒", "戚"]},
        "つつみがまえ": {"char": "勹", "sample": "勾", "split": "frame", "ratio": 0.5, "alternatives": ["匂", "包", "旬"]},
    },
}

# ============================================================
# [BLOCK1-END]
# ============================================================


# ============================================================
# [DYNAMIC-BOUNDARY] 動的境界検出アルゴリズム (2025-11-09)
# ============================================================

class DynamicBoundaryDetector:
    """動的境界検出器 - 画像解析で最適な分割位置を自動検出（v1.82.9）"""

    def __init__(self, binary_threshold: int = 200):
        self.binary_threshold = binary_threshold

    def find_optimal_split(self, img: Image.Image, direction: str = "vertical",
                          search_range: Tuple[float, float] = (0.3, 0.7),
                          num_candidates: int = 3) -> List[Tuple[float, float, Dict]]:
        """
        最適な分割位置を検出

        Args:
            img: 入力画像
            direction: "vertical" (左右分割) or "horizontal" (上下分割)
            search_range: 探索範囲 (min_ratio, max_ratio)
            num_candidates: 返す候補数

        Returns:
            [(ratio, score, info), ...] のリスト
            - ratio: 分割比率（0.0～1.0）
            - score: スコア（低いほど境界らしい）
            - info: 詳細情報
        """
        w, h = img.size
        img_array = np.array(img)
        binary = img_array < self.binary_threshold

        candidates = []

        if direction == "vertical":
            # 縦方向に走査（左右分割）
            for ratio in np.arange(search_range[0], search_range[1], Config.BOUNDARY_SCAN_STEP):
                x = int(w * ratio)
                if x <= 0 or x >= w:
                    continue

                # この位置での垂直線上の黒ピクセル密度
                line = binary[:, x]
                density = np.sum(line) / h

                # 周辺の密度変化も考慮（境界っぽさを強調）
                edge_score = self._calculate_edge_score(binary, x, "vertical")

                # 総合スコア（密度が低く、エッジが強いほど良い）
                score = density * 0.7 + (1.0 - edge_score) * 0.3

                candidates.append((ratio, score, {
                    'density': density,
                    'edge_score': edge_score,
                    'position': x
                }))
        else:
            # 横方向に走査（上下分割）
            for ratio in np.arange(search_range[0], search_range[1], Config.BOUNDARY_SCAN_STEP):
                y = int(h * ratio)
                if y <= 0 or y >= h:
                    continue

                line = binary[y, :]
                density = np.sum(line) / w

                edge_score = self._calculate_edge_score(binary, y, "horizontal")

                score = density * 0.7 + (1.0 - edge_score) * 0.3

                candidates.append((ratio, score, {
                    'density': density,
                    'edge_score': edge_score,
                    'position': y
                }))

        # スコアが低い順（境界らしい順）にソート
        candidates.sort(key=lambda x: x[1])

        # トップN候補を返す
        return candidates[:num_candidates]

    def _calculate_edge_score(self, binary: np.ndarray, position: int, direction: str) -> float:
        """エッジスコアを計算（境界の強さ）"""
        h, w = binary.shape

        if direction == "vertical":
            if position <= 2 or position >= w - 3:
                return 0.0

            # 左右の密度差
            left_region = binary[:, max(0, position - 5):position]
            right_region = binary[:, position:min(w, position + 5)]

            left_density = np.sum(left_region) / (left_region.size + 1e-8)
            right_density = np.sum(right_region) / (right_region.size + 1e-8)

            # 密度差が大きいほど境界らしい
            edge_strength = abs(left_density - right_density)

            return edge_strength
        else:
            if position <= 2 or position >= h - 3:
                return 0.0

            top_region = binary[max(0, position - 5):position, :]
            bottom_region = binary[position:min(h, position + 5), :]

            top_density = np.sum(top_region) / (top_region.size + 1e-8)
            bottom_density = np.sum(bottom_region) / (bottom_region.size + 1e-8)

            edge_strength = abs(top_density - bottom_density)

            return edge_strength










# ============================================================
# [BLOCK2-BEGIN] 画像処理ユーティリティ (2025-10-10)
# ============================================================

def render_char_to_bitmap(char, font_path, size=None, font_index=0):
    """文字をビットマップにレンダリング（TTC対応）"""
    if size is None:
        size = Config.RENDER_SIZE
    try:
        # TTCファイル対応：indexパラメータを追加
        font = ImageFont.truetype(font_path, size, index=font_index)
        img = Image.new("L", (size, size), 255)
        draw = ImageDraw.Draw(img)

        bbox = draw.textbbox((0, 0), char, font=font)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]

        x = (size - w) / 2 - bbox[0]
        y = (size - h) / 2 - bbox[1]

        draw.text((x, y), char, fill=0, font=font)
        return img
    except Exception as e:
        # エラー詳細をログに出力
        print(f"[ERROR] render_char_to_bitmap: {e}")
        return None


def find_split_position(img, direction="vertical", ratio=0.5):
    """画像の分割位置を検出（ratio指定対応）"""
    w, h = img.size
    
    if direction == "vertical":
        return int(w * ratio)
    else:
        return int(h * ratio)


def split_glyph(img, split_type, ratio=0.5):
    """グリフを分割してパーツを抽出（比率指定対応）"""
    if img is None:
        return None
    
    w, h = img.size
    
    if split_type == "left":
        split_x = find_split_position(img, "vertical", ratio)
        return img.crop((0, 0, split_x, h))
    elif split_type == "right":
        split_x = find_split_position(img, "vertical", ratio)
        return img.crop((split_x, 0, w, h))
    elif split_type == "top":
        split_y = find_split_position(img, "horizontal", ratio)
        return img.crop((0, 0, w, split_y))
    elif split_type == "bottom":
        split_y = find_split_position(img, "horizontal", ratio)
        return img.crop((0, split_y, w, h))
    elif split_type == "left_bottom":
        split_x = int(w * ratio)
        split_y = int(h * 0.7)
        
        result = Image.new("L", (w, h), 255)
        result.paste(img.crop((0, 0, split_x, h)), (0, 0))
        result.paste(img.crop((0, split_y, w, h)), (0, split_y))
        
        return result
    elif split_type == "top_left":
        split_x = int(w * ratio)
        split_y = int(h * 0.4)
        
        result = Image.new("L", (w, h), 255)
        result.paste(img.crop((0, 0, w, split_y)), (0, 0))
        result.paste(img.crop((0, 0, split_x, h)), (0, 0))
        
        return result
    elif split_type == "top_right":
        split_x = int(w * (1.0 - ratio))
        split_y = int(h * 0.4)
        
        result = Image.new("L", (w, h), 255)
        result.paste(img.crop((0, 0, w, split_y)), (0, 0))
        result.paste(img.crop((split_x, 0, w, h)), (split_x, 0))
        
        return result
    elif split_type == "right_bottom":
        split_x = int(w * (1.0 - ratio))
        split_y = int(h * 0.7)
        
        result = Image.new("L", (w, h), 255)
        result.paste(img.crop((split_x, 0, w, h)), (split_x, 0))
        result.paste(img.crop((0, split_y, w, h)), (0, split_y))
        
        return result
    elif split_type == "frame":
        return img
    else:
        return img


def remove_noise(img, min_size=50):
    """ノイズ除去（孤立した小さなピクセル塊を削除）"""
    if img is None:
        return None
    
    pixels = img.load()
    w, h = img.size
    visited = set()
    
    def flood_fill_count(start_x, start_y):
        """連結成分のサイズをカウント"""
        stack = [(start_x, start_y)]
        count = 0
        coords = []
        
        while stack:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            if not (0 <= x < w and 0 <= y < h):
                continue
            if pixels[x, y] >= 128:
                continue
            
            visited.add((x, y))
            coords.append((x, y))
            count += 1
            
            stack.extend([(x+1, y), (x-1, y), (x, y+1), (x, y-1)])
        
        return count, coords
    
    result = img.copy()
    result_pixels = result.load()
    
    for y in range(h):
        for x in range(w):
            if (x, y) not in visited and pixels[x, y] < 128:
                size, coords = flood_fill_count(x, y)
                if size < min_size:
                    for cx, cy in coords:
                        result_pixels[cx, cy] = 255
    
    return result


def trim_whitespace(img):
    """余白を削除"""
    if img is None:
        return None
    bbox = img.getbbox()
    if bbox:
        return img.crop(bbox)
    return img


def save_as_transparent_png(img, output_path, canvas_size=2048):
    """グレースケール画像を2048x2048の透過PNGとして保存（中央配置）"""
    if img is None:
        return False

    try:
        # 2048x2048の透明キャンバスを作成
        canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

        # トリミング後の画像をRGBAに変換
        rgba = Image.new("RGBA", img.size, (0, 0, 0, 0))
        pixels = img.load()
        rgba_pixels = rgba.load()

        for y in range(img.height):
            for x in range(img.width):
                gray = pixels[x, y]
                alpha = 255 - gray
                rgba_pixels[x, y] = (0, 0, 0, alpha)

        # キャンバスの中央に配置
        x_offset = (canvas_size - img.width) // 2
        y_offset = (canvas_size - img.height) // 2
        canvas.paste(rgba, (x_offset, y_offset), rgba)

        # 保存
        canvas.save(output_path, "PNG")

        # サムネイルも同時に生成（100x100）
        try:
            thumbnail_path = output_path.replace('.png', '_thumb.png')
            # 元の rgba 画像（トリミング済み）からサムネイル作成
            thumb_bg = Image.new('RGB', rgba.size, (255, 255, 255))
            thumb_bg.paste(rgba, mask=rgba.split()[3])
            thumb_bg.thumbnail((100, 100), Image.Resampling.LANCZOS)
            thumb_bg.save(thumbnail_path, "PNG")
        except Exception as thumb_error:
            print(f"[WARNING] サムネイル生成失敗: {thumb_error}")

        return True
    except Exception as e:
        print(f"[ERROR] PNG保存失敗: {e}")
        return False

# ============================================================
# [BLOCK2-END]
# ============================================================











# ============================================================
# [BLOCK3-BEGIN] パーツ抽出コア処理 (2025-10-10)
# ============================================================

def extract_single_part(font_path, part_name, part_info, output_path, noise_removal=True, log_callback=None, font_index=0):
    """単一パーツを抽出（動的境界検出対応・TTC対応）"""
    try:
        split_type = part_info["split"]
        ratio = part_info.get("ratio", 0.5)

        # 試行する文字のリスト（プライマリ + 代替文字）
        candidates = [part_info["sample"]]
        if "alternatives" in part_info:
            candidates.extend(part_info["alternatives"])

        # 各候補でレンダリングを試行
        img = None
        used_char = None
        for candidate_char in candidates:
            img = render_char_to_bitmap(candidate_char, font_path, font_index=font_index)
            if img is not None:
                used_char = candidate_char
                break

        # 全ての候補で失敗した場合
        if img is None:
            error_msg = f"レンダリング失敗（全ての代替文字でも失敗）\n試行した文字: {', '.join(candidates)}"
            return False, None, error_msg, None, ratio

        # 動的境界検出（オプション機能）
        used_ratio = ratio
        dynamic_detection_used = False
        dynamic_detection_error = None

        if Config.DYNAMIC_BOUNDARY_DETECTION:
            try:
                detector = DynamicBoundaryDetector(binary_threshold=Config.BINARY_THRESHOLD)

                # split_typeから方向を決定
                if split_type in ["left", "right"]:
                    direction = "vertical"
                    search_range = Config.BOUNDARY_SEARCH_RANGE_LR
                elif split_type in ["top", "bottom"]:
                    direction = "horizontal"
                    search_range = Config.BOUNDARY_SEARCH_RANGE_TB
                else:
                    # frame, left_bottom, top_left は動的検出非対応（固定ratioを使用）
                    direction = None

                if direction:
                    # 最適な分割位置を検出
                    candidates_dynamic = detector.find_optimal_split(img, direction, search_range, num_candidates=1)
                    if candidates_dynamic:
                        old_ratio = used_ratio
                        used_ratio = candidates_dynamic[0][0]  # トップ候補のratio
                        dynamic_detection_used = True
                        if log_callback and abs(used_ratio - old_ratio) > 0.01:
                            log_callback(f"    [動的検出] {part_name}: {old_ratio:.3f} → {used_ratio:.3f}")
            except Exception as e:
                # 動的検出に失敗した場合は固定ratioを使用
                dynamic_detection_error = str(e)
                if log_callback:
                    log_callback(f"    [動的検出エラー] {part_name}: {e}")

        # 分割処理
        part_img = split_glyph(img, split_type, used_ratio)
        if part_img is None:
            return False, None, "分割失敗", used_char, used_ratio

        # ノイズ除去
        if noise_removal:
            part_img = remove_noise(part_img)

        # 余白トリミング
        part_img = trim_whitespace(part_img)

        # 保存
        if save_as_transparent_png(part_img, output_path):
            return True, part_img, None, used_char, used_ratio
        else:
            return False, None, "保存失敗", used_char, used_ratio

    except Exception as e:
        return False, None, str(e), None, ratio


def extract_all_parts(font_path, output_dir, progress_callback=None, log_callback=None, font_index=0):
    """フォントから全パーツを抽出（TTC対応）"""

    def log(msg):
        if log_callback:
            log_callback(msg)
        else:
            print(msg)
    
    os.makedirs(output_dir, exist_ok=True)
    
    stats = {
        "total": 0,
        "success": 0,
        "failed": 0,
        "by_category": {}
    }
    
    catalog_json = {}
    
    log("=" * 70)
    log("偏旁抽出ツール")
    log("=" * 70)
    log(f"フォント: {font_path}")
    log(f"出力先: {output_dir}")
    log("=" * 70)
    log("")
    
    total_parts = sum(len(parts) for parts in PARTS_CATALOG.values())
    current_idx = 0
    
    for category, parts in PARTS_CATALOG.items():
        category_name = {
            "hen": "偏（へん）",
            "tsukuri": "旁（つくり）",
            "kanmuri": "冠（かんむり）",
            "ashi": "脚（あし）",
            "nyou": "繞（にょう）",
            "tare": "垂（たれ）",
            "kamae": "構（かまえ）"
        }.get(category, category)
        
        log(f"\n【{category_name}】")
        log("-" * 70)
        
        category_stats = {"success": 0, "failed": 0}
        catalog_json[category] = {}
        
        for part_name, part_info in parts.items():
            current_idx += 1
            stats["total"] += 1
            
            filename = f"{category}_{part_name}_{part_info['char']}.png"
            output_path = os.path.join(output_dir, filename)
            
            msg = f"  {part_name} ({part_info['char']}) [例: {part_info['sample']}]"

            if progress_callback:
                progress_callback(current_idx, total_parts, f"{part_name} 処理中...")

            success, img, error, used_char, used_ratio = extract_single_part(
                font_path, part_name, part_info, output_path, log_callback=log, font_index=font_index
            )

            if success:
                log(f"{msg} ... ✅ 保存完了")
                stats["success"] += 1
                category_stats["success"] += 1

                catalog_json[category][part_name] = {
                    "char": part_info["char"],
                    "sample": part_info["sample"],
                    "file": filename,
                    "split": part_info["split"],
                    "ratio": part_info.get("ratio", 0.5),
                    "used_ratio": used_ratio  # 実際に使用された分割比率を記録
                }
            else:
                log(f"{msg} ... ❌ {error}")
                stats["failed"] += 1
                category_stats["failed"] += 1
        
        stats["by_category"][category] = category_stats
    
    catalog_path = os.path.join(output_dir, "parts_catalog.json")
    with open(catalog_path, 'w', encoding='utf-8') as f:
        json.dump(catalog_json, f, ensure_ascii=False, indent=2)
    
    log("\n" + "=" * 70)
    log("抽出完了")
    log("=" * 70)
    log(f"✅ 成功: {stats['success']}")
    log(f"❌ 失敗: {stats['failed']}")
    log(f"📊 合計: {stats['total']}")
    log("\nカテゴリ別:")
    for cat, cat_stats in stats["by_category"].items():
        log(f"  {cat:10s}: 成功 {cat_stats['success']:2d} / 失敗 {cat_stats['failed']:2d}")
    log(f"\n📁 保存先: {os.path.abspath(output_dir)}")
    log(f"📋 カタログ: {catalog_path}")
    log("=" * 70)
    
    return stats

# ============================================================
# [BLOCK3-END]
# ============================================================











# ============================================================
# [BLOCK4-BEGIN] パーツプレビュー・編集GUI (2025-10-10: 補間描画追加)
# ============================================================

class PartsPreviewWindow(tk.Toplevel):
    """偏旁エディタウィンドウ"""

    def __init__(self, parent, parts_dir, font_path, font_index=0):
        super().__init__(parent)
        self.title("偏旁エディタ")
        self.geometry("1500x850")

        self.parts_dir = parts_dir
        self.font_path = font_path
        self.font_index = font_index  # TTCファイル用のインデックス
        self.catalog = self._load_catalog()
        self.current_category = None
        self.current_part = None
        self.photo_cache = {}
        
        self.eraser_mode = False
        self.eraser_size = 20
        self.eraser_shape = 'circle'
        self.current_image = None
        self.modified = False
        self.current_split_type = 'left'  # デフォルトの分割タイプ

        self.undo_stack = []
        self.redo_stack = []
        
        self.zoom_level = 1.0
        self.zoom_levels = [0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
        self.preview_scale = 1.0

        # パン（スクロール）機能用
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.is_panning = False

        # 表示オフセット（座標変換用）
        self.display_x_offset = 0
        self.display_y_offset = 0

        self.eraser_cursor_id = None

        # [ADD] 2025-10-10: 補間描画用
        self.last_erase_pos = None
        
        self._setup_ui()
        self._load_preview()
    
    def _load_catalog(self):
        """カタログJSON読み込み"""
        catalog_path = os.path.join(self.parts_dir, "parts_catalog.json")
        if os.path.exists(catalog_path):
            with open(catalog_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {}
    
    def _setup_ui(self):
        """UI構築"""
        # 左側: カテゴリリスト
        left_frame = ttk.Frame(self, width=200)
        left_frame.pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=5)
        
        ttk.Label(left_frame, text="カテゴリ", font=("", 12, "bold")).pack(pady=5)
        
        self.category_listbox = tk.Listbox(left_frame, font=("", 22))
        self.category_listbox.pack(fill=tk.BOTH, expand=True, pady=5)
        self.category_listbox.bind('<<ListboxSelect>>', self._on_category_select)
        
        for category in ["hen", "tsukuri", "kanmuri", "ashi", "nyou", "tare", "kamae"]:
            display_name = {
                "hen": "偏（へん）",
                "tsukuri": "旁（つくり）",
                "kanmuri": "冠（かんむり）",
                "ashi": "脚（あし）",
                "nyou": "繞（にょう）",
                "tare": "垂（たれ）",
                "kamae": "構（かまえ）"
            }.get(category, category)
            self.category_listbox.insert(tk.END, display_name)
        
        # 中央: パーツ一覧
        center_frame = ttk.Frame(self)
        center_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        ttk.Label(center_frame, text="パーツ一覧", font=("", 12, "bold")).pack(pady=5)
        
        canvas_frame = ttk.Frame(center_frame)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        self.parts_canvas = tk.Canvas(canvas_frame, bg="white")
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=self.parts_canvas.yview)
        self.scrollable_frame = ttk.Frame(self.parts_canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.parts_canvas.configure(scrollregion=self.parts_canvas.bbox("all"))
        )

        # キャンバスリサイズ時にグリッドを再描画
        self._parts_canvas_resize_after_id = None
        self.parts_canvas.bind('<Configure>', self._on_parts_canvas_resize)
        
        self.parts_canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.parts_canvas.configure(yscrollcommand=scrollbar.set)
        
        self.parts_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # 右側: 詳細編集パネル
        right_frame = ttk.Frame(self, width=500)
        right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, padx=5, pady=5)
        
        ttk.Label(right_frame, text="パーツ編集", font=("", 12, "bold")).pack(pady=5)
        
        # パーツ情報
        info_frame = ttk.LabelFrame(right_frame, text="情報", padding=5)
        info_frame.pack(fill=tk.X, pady=5)
        
        self.info_label = ttk.Label(info_frame, text="パーツを選択してください", wraplength=450)
        self.info_label.pack()
        
        # プレビュー
        preview_frame = ttk.LabelFrame(right_frame, text="プレビュー", padding=5)
        preview_frame.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # ズームコントロール
        zoom_control_frame = ttk.Frame(preview_frame)
        zoom_control_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(zoom_control_frame, text="ズーム:").pack(side=tk.LEFT, padx=2)
        ttk.Button(zoom_control_frame, text="-", command=self._zoom_out, width=3).pack(side=tk.LEFT, padx=1)
        self.zoom_label = ttk.Label(zoom_control_frame, text="100%", width=6)
        self.zoom_label.pack(side=tk.LEFT, padx=1)
        ttk.Button(zoom_control_frame, text="+", command=self._zoom_in, width=3).pack(side=tk.LEFT, padx=1)
        ttk.Button(zoom_control_frame, text="リセット", command=self._zoom_reset, width=6).pack(side=tk.LEFT, padx=1)
        
        ttk.Button(zoom_control_frame, text="↶元に戻す", command=self._undo, width=10).pack(side=tk.LEFT, padx=5)
        ttk.Button(zoom_control_frame, text="↷やり直し", command=self._redo, width=10).pack(side=tk.LEFT, padx=1)
        
        # プレビューキャンバス
        self.preview_canvas = tk.Canvas(preview_frame, bg="white", relief="sunken", borderwidth=2)
        self.preview_canvas.pack(fill=tk.BOTH, expand=True)
        # リサイズ時にプレビューを更新（デバウンス付き）
        self._resize_after_id = None
        self.preview_canvas.bind('<Configure>', self._on_canvas_resize)
        self.preview_canvas.bind('<Button-1>', self._on_canvas_click)
        self.preview_canvas.bind('<B1-Motion>', self._on_canvas_drag)
        self.preview_canvas.bind('<ButtonRelease-1>', self._on_canvas_release)
        self.preview_canvas.bind('<Motion>', self._on_canvas_motion)
        # パン機能（中ボタンまたはShift+左ボタン）
        self.preview_canvas.bind('<Button-2>', self._on_pan_start)
        self.preview_canvas.bind('<B2-Motion>', self._on_pan_drag)
        self.preview_canvas.bind('<ButtonRelease-2>', self._on_pan_end)
        self.preview_canvas.bind('<Shift-Button-1>', self._on_pan_start)
        self.preview_canvas.bind('<Shift-B1-Motion>', self._on_pan_drag)
        self.preview_canvas.bind('<Shift-ButtonRelease-1>', self._on_pan_end)
        # マウスホイールでズーム
        self.preview_canvas.bind('<MouseWheel>', self._on_mousewheel_zoom)
        self.preview_canvas.bind('<Button-4>', self._on_mousewheel_zoom)  # Linux
        self.preview_canvas.bind('<Button-5>', self._on_mousewheel_zoom)  # Linux

        # 矢印キーでパン（ウィンドウ全体にバインド）
        self.bind('<Up>', self._on_arrow_key)
        self.bind('<Down>', self._on_arrow_key)
        self.bind('<Left>', self._on_arrow_key)
        self.bind('<Right>', self._on_arrow_key)

        # 編集ツール
        tools_frame = ttk.LabelFrame(right_frame, text="編集ツール", padding=5)
        tools_frame.pack(fill=tk.X, pady=5)
        
        # サンプル文字 + 分割タイプ
        row0_frame = ttk.Frame(tools_frame)
        row0_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(row0_frame, text="サンプル:").pack(side=tk.LEFT)
        self.sample_entry = ttk.Entry(row0_frame, width=4)
        self.sample_entry.pack(side=tk.LEFT, padx=2)
        
        ttk.Label(row0_frame, text="分割:").pack(side=tk.LEFT, padx=(10, 2))
        self.split_type_var = tk.StringVar(value='left')
        split_combo = ttk.Combobox(row0_frame, textvariable=self.split_type_var, width=12, state='readonly')
        split_combo['values'] = [
            '左 (←)',
            '右 (→)',
            '上 (↑)',
            '下 (↓)',
            'L字 (└)',
            '逆L (┐)',
            '┌字',
            '┘字',
            '囲み'
        ]
        split_combo.pack(side=tk.LEFT, padx=2)
        split_combo.bind('<<ComboboxSelected>>', self._on_split_type_change)
        
        # 分割比率
        row1_frame = ttk.Frame(tools_frame)
        row1_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(row1_frame, text="分割比率:").pack(side=tk.LEFT)
        self.ratio_var = tk.DoubleVar(value=0.5)
        ratio_scale = ttk.Scale(row1_frame, from_=0.0, to=1.0, variable=self.ratio_var, orient=tk.HORIZONTAL)
        ratio_scale.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        self.ratio_label = ttk.Label(row1_frame, text="50%", width=5)
        self.ratio_label.pack(side=tk.LEFT)
        self.ratio_var.trace_add('write', lambda *args: self.ratio_label.config(text=f"{int(self.ratio_var.get()*100)}%"))
        
        # ノイズ除去
        self.noise_removal_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(tools_frame, text="ノイズ自動除去", variable=self.noise_removal_var).pack(anchor=tk.W, pady=2)
        
        # 消しゴム
        self.eraser_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(tools_frame, text="消しゴムモード", variable=self.eraser_var, command=self._toggle_eraser).pack(anchor=tk.W, pady=2)
        
        # 消しゴム形状 + サイズ
        row2_frame = ttk.Frame(tools_frame)
        row2_frame.pack(fill=tk.X, pady=2)
        
        ttk.Label(row2_frame, text="形状:").pack(side=tk.LEFT)
        self.eraser_shape_var = tk.StringVar(value='circle')
        ttk.Radiobutton(row2_frame, text="●", variable=self.eraser_shape_var, value='circle', command=self._on_shape_change, width=3).pack(side=tk.LEFT)
        ttk.Radiobutton(row2_frame, text="■", variable=self.eraser_shape_var, value='square', command=self._on_shape_change, width=3).pack(side=tk.LEFT)
        ttk.Radiobutton(row2_frame, text="◆", variable=self.eraser_shape_var, value='diamond', command=self._on_shape_change, width=3).pack(side=tk.LEFT)
        
        ttk.Label(row2_frame, text="サイズ:").pack(side=tk.LEFT, padx=(10, 2))
        self.eraser_size_var = tk.IntVar(value=20)
        eraser_scale = ttk.Scale(row2_frame, from_=5, to=50, variable=self.eraser_size_var, orient=tk.HORIZONTAL, length=100)
        eraser_scale.pack(side=tk.LEFT, padx=2)
        self.eraser_size_var.trace_add('write', lambda *args: self._update_eraser_cursor())
        
        # ボタン
        button_frame = ttk.Frame(tools_frame)
        button_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(button_frame, text="🔄 再抽出", command=self._re_extract, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="💾 保存", command=self._save_current, width=12).pack(side=tk.LEFT, padx=2)
        ttk.Button(button_frame, text="🗑️ 削除", command=self._delete_current, width=12).pack(side=tk.LEFT, padx=2)
        
        # キーボードショートカット
        self.bind('<Control-z>', lambda e: self._undo())
        self.bind('<Control-y>', lambda e: self._redo())
        self.bind('<Control-Shift-Z>', lambda e: self._redo())
    
    def _load_preview(self):
        pass
    
    def _on_split_type_change(self, event):
        """分割タイプ変更時"""
        selected = self.split_type_var.get()
        split_type_map = {
            '左 (←)': 'left',
            '右 (→)': 'right',
            '上 (↑)': 'top',
            '下 (↓)': 'bottom',
            'L字 (└)': 'left_bottom',
            '逆L (┐)': 'top_left',
            '┌字': 'top_right',
            '┘字': 'right_bottom',
            '囲み': 'frame'
        }
        self.current_split_type = split_type_map.get(selected, 'left')
    
    def _on_category_select(self, event):
        """カテゴリ選択時"""
        selection = self.category_listbox.curselection()
        if not selection:
            return

        idx = selection[0]
        categories = ["hen", "tsukuri", "kanmuri", "ashi", "nyou", "tare", "kamae"]
        self.current_category = categories[idx]

        self._display_parts_grid()

    def _on_parts_canvas_resize(self, event):
        """パーツキャンバスリサイズ時の処理（デバウンス付き）"""
        # 既存のタイマーをキャンセル
        if self._parts_canvas_resize_after_id:
            self.after_cancel(self._parts_canvas_resize_after_id)
        # 300ms後にグリッド再描画（リサイズ中の頻繁な再描画を防ぐ）
        self._parts_canvas_resize_after_id = self.after(300, self._redisplay_if_category_selected)

    def _redisplay_if_category_selected(self):
        """カテゴリが選択されている場合のみグリッドを再描画"""
        if self.current_category:
            self._display_parts_grid()
    
    def _display_parts_grid(self):
        """パーツ一覧表示"""
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()

        self.photo_cache.clear()

        if self.current_category not in self.catalog:
            ttk.Label(self.scrollable_frame, text="パーツがありません").pack(pady=20)
            return

        parts = self.catalog[self.current_category]

        # キャンバス幅に基づいて動的にカラム数を決定
        self.parts_canvas.update_idletasks()
        canvas_width = self.parts_canvas.winfo_width()
        # 各アイテムの幅を約120px（パディング含む）として計算
        item_width = 120
        max_cols = max(1, (canvas_width - 20) // item_width)

        col_count = 0
        row_count = 0
        
        for part_name, part_data in parts.items():
            filename = part_data["file"]
            filepath = os.path.join(self.parts_dir, filename)

            if not os.path.exists(filepath):
                continue

            try:
                # サムネイルがある場合はそれを使用（高速化）
                thumbnail_path = filepath.replace('.png', '_thumb.png')
                if os.path.exists(thumbnail_path):
                    bg = Image.open(thumbnail_path).convert('RGB')
                else:
                    # サムネイルがない場合は従来通り生成
                    img = Image.open(filepath).convert('RGBA')
                    bg = Image.new('RGB', img.size, (255, 255, 255))
                    bg.paste(img, mask=img.split()[3])
                    bg.thumbnail((100, 100), Image.Resampling.LANCZOS)

                photo = ImageTk.PhotoImage(bg)
                self.photo_cache[part_name] = photo
                
                frame = ttk.Frame(self.scrollable_frame, relief="solid", borderwidth=1, padding=5)
                frame.grid(row=row_count, column=col_count, padx=5, pady=5)
                
                label = tk.Label(frame, image=photo, bg="white", cursor="hand2")
                label.pack()
                label.bind('<Button-1>', lambda e, p=part_name: self._select_part(p))
                
                name_label = ttk.Label(frame, text=part_name, font=("", 9))
                name_label.pack()
                
                col_count += 1
                if col_count >= max_cols:
                    col_count = 0
                    row_count += 1
                    
            except Exception as e:
                print(f"[ERROR] サムネイル作成失敗: {part_name} - {e}")
    
    def _select_part(self, part_name):
        """パーツ選択"""
        self.current_part = part_name
        part_data = self.catalog[self.current_category][part_name]
        
        info_text = f"名前: {part_name}\n"
        info_text += f"文字: {part_data['char']}\n"
        info_text += f"サンプル: {part_data['sample']}\n"
        info_text += f"分割: {part_data['split']}\n"
        info_text += f"比率: {part_data['ratio']}"
        self.info_label.config(text=info_text)
        
        self.sample_entry.delete(0, tk.END)
        self.sample_entry.insert(0, part_data['sample'])
        self.ratio_var.set(part_data['ratio'])
        
        # 分割タイプを設定
        split_type_reverse_map = {
            'left': '左 (←)',
            'right': '右 (→)',
            'top': '上 (↑)',
            'bottom': '下 (↓)',
            'left_bottom': 'L字 (└)',
            'top_left': '逆L (┐)',
            'top_right': '┌字',
            'right_bottom': '┘字',
            'frame': '囲み'
        }
        display_split = split_type_reverse_map.get(part_data['split'], '左 (←)')
        self.split_type_var.set(display_split)
        self.current_split_type = part_data['split']
        
        filepath = os.path.join(self.parts_dir, part_data['file'])
        if os.path.exists(filepath):
            img = Image.open(filepath).convert('RGBA')
            
            bg = Image.new('L', img.size, 255)
            for y in range(img.height):
                for x in range(img.width):
                    r, g, b, a = img.getpixel((x, y))
                    if a > 0:
                        bg.putpixel((x, y), 0)
                    else:
                        bg.putpixel((x, y), 255)
            
            self.current_image = bg
            self.zoom_level = 1.0
            
            self.undo_stack = [self.current_image.copy()]
            self.redo_stack = []
            
            self._update_preview()
    
    def _save_to_undo(self):
        """現在の状態をアンドゥスタックに保存"""
        if self.current_image:
            self.undo_stack.append(self.current_image.copy())
            if len(self.undo_stack) > 50:
                self.undo_stack.pop(0)
            self.redo_stack.clear()
    
    def _undo(self):
        """元に戻す"""
        if len(self.undo_stack) > 1:
            self.redo_stack.append(self.undo_stack.pop())
            self.current_image = self.undo_stack[-1].copy()
            self._update_preview()
    
    def _redo(self):
        """やり直し"""
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.undo_stack.append(state)
            self.current_image = state.copy()
            self._update_preview()
    
    def _zoom_in(self):
        """ズームイン"""
        current_idx = self.zoom_levels.index(self.zoom_level) if self.zoom_level in self.zoom_levels else 2
        if current_idx < len(self.zoom_levels) - 1:
            self.zoom_level = self.zoom_levels[current_idx + 1]
            self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
            self._update_preview()
    
    def _zoom_out(self):
        """ズームアウト"""
        current_idx = self.zoom_levels.index(self.zoom_level) if self.zoom_level in self.zoom_levels else 2
        if current_idx > 0:
            self.zoom_level = self.zoom_levels[current_idx - 1]
            self.zoom_label.config(text=f"{int(self.zoom_level * 100)}%")
            self._update_preview()
    
    def _zoom_reset(self):
        """ズームリセット"""
        self.zoom_level = 1.0
        self.pan_offset_x = 0
        self.pan_offset_y = 0
        self.zoom_label.config(text="100%")
        self._update_preview()
    
    def _update_preview(self):
        """プレビュー更新"""
        if self.current_image is None:
            return

        # キャンバスの実際のサイズを取得（動的対応）
        self.preview_canvas.update_idletasks()
        canvas_w = self.preview_canvas.winfo_width()
        canvas_h = self.preview_canvas.winfo_height()

        # 初期化前の場合はデフォルトサイズを使用
        if canvas_w <= 1:
            canvas_w = 400
        if canvas_h <= 1:
            canvas_h = 350

        orig_w = self.current_image.width
        orig_h = self.current_image.height

        scale_w = canvas_w / orig_w if orig_w > 0 else 1.0
        scale_h = canvas_h / orig_h if orig_h > 0 else 1.0
        fit_scale = min(scale_w, scale_h, 1.0)

        final_scale = fit_scale * self.zoom_level

        new_w = int(orig_w * final_scale)
        new_h = int(orig_h * final_scale)

        # LANCZOS補間で滑らかに拡大（ガクつき防止）
        display_img = self.current_image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # チェッカーボード背景を作成（透過部分と外側を区別）
        bg = self._create_checkerboard_bg(canvas_w, canvas_h)

        # パンオフセットを適用
        x_offset = (canvas_w - new_w) // 2 + self.pan_offset_x
        y_offset = (canvas_h - new_h) // 2 + self.pan_offset_y

        # 表示オフセットを保存（座標変換用）
        self.display_x_offset = x_offset
        self.display_y_offset = y_offset

        if x_offset >= 0 and y_offset >= 0 and x_offset + new_w <= canvas_w and y_offset + new_h <= canvas_h:
            bg.paste(display_img, (x_offset, y_offset))
        else:
            paste_x = max(0, x_offset)
            paste_y = max(0, y_offset)

            crop_x = max(0, -x_offset)
            crop_y = max(0, -y_offset)
            crop_w = min(new_w - crop_x, canvas_w - paste_x)
            crop_h = min(new_h - crop_y, canvas_h - paste_y)

            if crop_w > 0 and crop_h > 0:
                cropped = display_img.crop((crop_x, crop_y, crop_x + crop_w, crop_y + crop_h))
                bg.paste(cropped, (paste_x, paste_y))

        self.preview_photo = ImageTk.PhotoImage(bg)
        self.preview_canvas.delete("all")
        self.preview_canvas.create_image(canvas_w//2, canvas_h//2, image=self.preview_photo)

        self.preview_scale = final_scale

    def _create_checkerboard_bg(self, width, height, grid_size=16):
        """チェッカーボード背景を作成（透過部分の可視化）"""
        bg = Image.new('L', (width, height), 255)
        pixels = bg.load()

        for y in range(height):
            for x in range(width):
                # チェッカーパターン
                if ((x // grid_size) + (y // grid_size)) % 2 == 0:
                    pixels[x, y] = 240  # 明るいグレー
                else:
                    pixels[x, y] = 220  # やや暗いグレー

        return bg
    
    def _toggle_eraser(self):
        """消しゴムモード切り替え"""
        self.eraser_mode = self.eraser_var.get()
        if self.eraser_mode:
            self.preview_canvas.config(cursor="none")
        else:
            self.preview_canvas.config(cursor="")
            if self.eraser_cursor_id:
                self.preview_canvas.delete(self.eraser_cursor_id)
                self.eraser_cursor_id = None
    
    def _on_shape_change(self):
        """消しゴム形状変更"""
        self.eraser_shape = self.eraser_shape_var.get()
        self._update_eraser_cursor()
    
    def _on_canvas_motion(self, event):
        """マウス移動時の処理"""
        if self.eraser_mode:
            self._update_eraser_cursor_position(event.x, event.y)

    def _on_canvas_resize(self, event):
        """キャンバスリサイズ時の処理（デバウンス付き）"""
        # 既存のタイマーをキャンセル
        if self._resize_after_id:
            self.after_cancel(self._resize_after_id)
        # 100ms後にプレビュー更新
        self._resize_after_id = self.after(100, self._update_preview)

    def _update_eraser_cursor(self):
        """消しゴムカーソルの形状を更新"""
        pass
    
    def _update_eraser_cursor_position(self, x, y):
        """消しゴムカーソル位置更新"""
        if self.eraser_cursor_id:
            self.preview_canvas.delete(self.eraser_cursor_id)
        
        if not self.eraser_mode:
            return
        
        radius = int(self.eraser_size_var.get() * self.preview_scale)
        
        if self.eraser_shape == 'circle':
            self.eraser_cursor_id = self.preview_canvas.create_oval(
                x - radius, y - radius, x + radius, y + radius,
                outline='red', width=2, dash=(3, 3)
            )
        elif self.eraser_shape == 'square':
            self.eraser_cursor_id = self.preview_canvas.create_rectangle(
                x - radius, y - radius, x + radius, y + radius,
                outline='red', width=2, dash=(3, 3)
            )
        elif self.eraser_shape == 'diamond':
            points = [
                x, y - radius,
                x + radius, y,
                x, y + radius,
                x - radius, y
            ]
            self.eraser_cursor_id = self.preview_canvas.create_polygon(
                points, outline='red', width=2, dash=(3, 3), fill=''
            )
    
    def _on_canvas_click(self, event):
        """キャンバスクリック"""
        if self.is_panning:
            return

        if self.eraser_mode and self.current_image:
            # 消しゴムモード：消去
            self._save_to_undo()
            img_x, img_y = self._canvas_to_image_coords(event.x, event.y)
            self.last_erase_pos = (img_x, img_y)
            self._erase_at_image(img_x, img_y)
        elif self.current_image:
            # 通常モード：パン開始
            self._on_pan_start(event)

    def _on_canvas_drag(self, event):
        """キャンバスドラッグ"""
        if self.is_panning:
            self._on_pan_drag(event)
            return

        if self.eraser_mode and self.current_image:
            # 消しゴムモード：消去
            img_x, img_y = self._canvas_to_image_coords(event.x, event.y)

            if self.last_erase_pos:
                # 前回の位置から現在の位置まで補間
                self._interpolate_erase(self.last_erase_pos[0], self.last_erase_pos[1], img_x, img_y)
            else:
                self._erase_at_image(img_x, img_y)

            self.last_erase_pos = (img_x, img_y)
    
    def _on_canvas_release(self, event):
        """マウスボタン解放"""
        self.last_erase_pos = None
        # パンモード終了
        if self.is_panning and not self.eraser_mode:
            self._on_pan_end(event)

    def _on_arrow_key(self, event):
        """矢印キーでパン"""
        if not self.current_image:
            return

        # パン移動量（ピクセル）
        pan_step = 20

        if event.keysym == 'Up':
            self.pan_offset_y += pan_step
        elif event.keysym == 'Down':
            self.pan_offset_y -= pan_step
        elif event.keysym == 'Left':
            self.pan_offset_x += pan_step
        elif event.keysym == 'Right':
            self.pan_offset_x -= pan_step

        self._update_preview()

    def _on_pan_start(self, event):
        """パン開始"""
        self.is_panning = True
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        if not self.eraser_mode:
            self.preview_canvas.config(cursor="fleur")

    def _on_pan_drag(self, event):
        """パン中"""
        if self.is_panning:
            dx = event.x - self.pan_start_x
            dy = event.y - self.pan_start_y
            self.pan_offset_x += dx
            self.pan_offset_y += dy
            self.pan_start_x = event.x
            self.pan_start_y = event.y
            self._update_preview()

    def _on_pan_end(self, event):
        """パン終了"""
        self.is_panning = False
        if self.eraser_mode:
            self.preview_canvas.config(cursor="none")
        else:
            self.preview_canvas.config(cursor="")

    def _on_mousewheel_zoom(self, event):
        """マウスホイールでズーム"""
        # Windowsとmacではevent.delta、Linuxではevent.num
        if hasattr(event, 'delta'):
            delta = event.delta
        elif event.num == 4:
            delta = 120
        elif event.num == 5:
            delta = -120
        else:
            return

        if delta > 0:
            self._zoom_in()
        else:
            self._zoom_out()
    
    def _canvas_to_image_coords(self, canvas_x, canvas_y):
        """キャンバス座標を画像座標に変換"""
        if self.current_image is None:
            return 0, 0

        # 表示オフセットを使用して正確に変換
        img_x = int((canvas_x - self.display_x_offset) / self.preview_scale)
        img_y = int((canvas_y - self.display_y_offset) / self.preview_scale)

        # 画像範囲内にクリップ
        img_x = max(0, min(img_x, self.current_image.width - 1))
        img_y = max(0, min(img_y, self.current_image.height - 1))

        return img_x, img_y
    
    def _interpolate_erase(self, x1, y1, x2, y2):
        """2点間を補間して消去（デコボコ軽減）"""
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        steps = max(int(distance / 2), 1)  # ブラシサイズの半分ごとに補間

        for i in range(steps + 1):
            t = i / steps
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            self._erase_at_image(x, y)

        self._update_preview()

        # 画像座標からキャンバス座標に変換してカーソル更新
        canvas_x = int(self.display_x_offset + x2 * self.preview_scale)
        canvas_y = int(self.display_y_offset + y2 * self.preview_scale)
        self._update_eraser_cursor_position(canvas_x, canvas_y)
    
    def _erase_at_image(self, img_x, img_y):
        """画像座標で消去"""  # [RENAME] 2025-10-10: _erase_atから名称変更
        if self.current_image is None:
            return
        
        if not (0 <= img_x < self.current_image.width and 0 <= img_y < self.current_image.height):
            return
        
        draw = ImageDraw.Draw(self.current_image)
        radius = int(self.eraser_size_var.get())
        
        if self.eraser_shape == 'circle':
            draw.ellipse([img_x-radius, img_y-radius, img_x+radius, img_y+radius], fill=255)
        elif self.eraser_shape == 'square':
            draw.rectangle([img_x-radius, img_y-radius, img_x+radius, img_y+radius], fill=255)
        elif self.eraser_shape == 'diamond':
            points = [
                (img_x, img_y - radius),
                (img_x + radius, img_y),
                (img_x, img_y + radius),
                (img_x - radius, img_y)
            ]
            draw.polygon(points, fill=255)
        
        self.modified = True
    
    def _re_extract(self):
        """再抽出"""
        if not self.current_part:
            messagebox.showwarning("警告", "パーツを選択してください")
            return

        sample_char = self.sample_entry.get()
        if not sample_char:
            messagebox.showwarning("警告", "サンプル文字を入力してください")
            return

        part_data = self.catalog[self.current_category][self.current_part]

        # part_infoに必要な情報を全て含める（alternativesも）
        part_info = {
            "sample": sample_char,
            "split": self.current_split_type,
            "ratio": self.ratio_var.get(),
            "char": part_data.get("char", ""),
            "alternatives": part_data.get("alternatives", [])
        }

        filename = part_data["file"]
        output_path = os.path.join(self.parts_dir, filename)

        # エラーログ用のコールバック
        error_log = []
        def log_error(msg):
            error_log.append(msg)
            print(msg)

        try:
            success, img, error, used_char, used_ratio = extract_single_part(
                self.font_path,
                self.current_part,
                part_info,
                output_path,
                noise_removal=self.noise_removal_var.get(),
                log_callback=log_error,
                font_index=self.font_index
            )

            if success:
                img_rgba = Image.open(output_path).convert('RGBA')
                bg = Image.new('L', img_rgba.size, 255)
                for y in range(img_rgba.height):
                    for x in range(img_rgba.width):
                        r, g, b, a = img_rgba.getpixel((x, y))
                        if a > 0:
                            bg.putpixel((x, y), 0)

                self.current_image = bg
                self.zoom_level = 1.0
                self.pan_offset_x = 0
                self.pan_offset_y = 0

                self.undo_stack = [self.current_image.copy()]
                self.redo_stack = []

                self._update_preview()

                # カタログ更新
                self.catalog[self.current_category][self.current_part]["sample"] = sample_char
                self.catalog[self.current_category][self.current_part]["split"] = self.current_split_type
                self.catalog[self.current_category][self.current_part]["ratio"] = self.ratio_var.get()
                if used_char and used_char != sample_char:
                    msg = f"再抽出しました\n（使用文字: {used_char}）"
                else:
                    msg = "再抽出しました"

                self._save_catalog()
                self._display_parts_grid()

                messagebox.showinfo("成功", msg)
            else:
                error_msg = f"再抽出失敗: {error}"
                if error_log:
                    error_msg += "\n\n詳細:\n" + "\n".join(error_log)
                messagebox.showerror("エラー", error_msg)
        except Exception as e:
            error_msg = f"予期しないエラー: {str(e)}"
            if error_log:
                error_msg += "\n\n詳細:\n" + "\n".join(error_log)
            messagebox.showerror("エラー", error_msg)
    
    def _save_current(self):
        """現在の編集を保存"""
        if not self.current_part or not self.current_image or not self.modified:
            return
        
        part_data = self.catalog[self.current_category][self.current_part]
        filepath = os.path.join(self.parts_dir, part_data['file'])
        
        if save_as_transparent_png(self.current_image, filepath):
            self.modified = False
            messagebox.showinfo("保存", "保存しました")
            self._display_parts_grid()
        else:
            messagebox.showerror("エラー", "保存失敗")
    
    def _delete_current(self):
        """現在のパーツを削除"""
        if not self.current_part:
            return
        
        if not messagebox.askyesno("確認", f"{self.current_part} を削除しますか？"):
            return
        
        part_data = self.catalog[self.current_category][self.current_part]
        filepath = os.path.join(self.parts_dir, part_data['file'])
        
        try:
            # メインファイルとサムネイルを削除
            if os.path.exists(filepath):
                os.remove(filepath)
            thumbnail_path = filepath.replace('.png', '_thumb.png')
            if os.path.exists(thumbnail_path):
                os.remove(thumbnail_path)

            del self.catalog[self.current_category][self.current_part]
            self._save_catalog()
            self._display_parts_grid()
            self.current_part = None
            self.current_image = None
            messagebox.showinfo("削除", "削除しました")
        except Exception as e:
            messagebox.showerror("エラー", f"削除失敗: {e}")
    
    def _save_catalog(self):
        """カタログJSON保存"""
        catalog_path = os.path.join(self.parts_dir, "parts_catalog.json")
        with open(catalog_path, 'w', encoding='utf-8') as f:
            json.dump(self.catalog, f, ensure_ascii=False, indent=2)

# ============================================================
# [BLOCK4-END]
# ============================================================











# ============================================================
# [BLOCK5-BEGIN] メインGUI (2025-10-10)
# ============================================================

class PartsExtractorGUI:
    """偏旁抽出ツールGUI"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("偏旁抽出ツール v2.9 (2025-11-09) - 動的境界検出 + 2048解像度対応")

        self.font_path = None
        self.font_index = 0  # TTCファイル用のフォントインデックス
        self.output_dir = "assets/parts"
        self.is_running = False

        self._setup_ui()
    
    def _setup_ui(self):
        """UI構築"""
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        file_frame = ttk.LabelFrame(main_frame, text="入力設定", padding=10)
        file_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(file_frame, text="フォントファイル:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.font_label = ttk.Label(file_frame, text="未選択", foreground="gray")
        self.font_label.grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Button(file_frame, text="参照...", command=self._select_font, width=10).grid(row=0, column=2, padx=5)
        
        ttk.Label(file_frame, text="出力ディレクトリ:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.output_label = ttk.Label(file_frame, text=self.output_dir)
        self.output_label.grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Button(file_frame, text="変更...", command=self._select_output, width=10).grid(row=1, column=2, padx=5)
        
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=10)
        
        self.extract_button = ttk.Button(button_frame, text="抽出開始", command=self._start_extraction)
        self.extract_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="プレビュー・編集", command=self._open_preview).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="出力先を開く", command=self._open_output).pack(side=tk.LEFT, padx=5)
        
        progress_frame = ttk.Frame(main_frame)
        progress_frame.pack(fill=tk.X, pady=5)
        
        self.progress_bar = ttk.Progressbar(progress_frame, mode='determinate')
        self.progress_bar.pack(fill=tk.X, pady=5)
        
        self.progress_label = ttk.Label(progress_frame, text="準備完了")
        self.progress_label.pack()
        
        log_frame = ttk.LabelFrame(main_frame, text="処理ログ", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=15,
            wrap=tk.WORD,
            font=("Monaco", 10) if sys.platform == "darwin" else ("Consolas", 9)
        )
        self.log_text.pack(fill=tk.BOTH, expand=True)

        self._log("偏旁抽出ツール v2.9 - 動的境界検出 + 2048解像度対応")
        self._log("=" * 70)
        self._log("【更新内容】")
        self._log("  ✅ 動的境界検出アルゴリズム統合")
        self._log("     - 画像解析で最適な分割位置を自動検出")
        self._log("     - 密度スキャン + エッジ検出による高精度抽出")
        self._log("  ✅ レンダリング解像度を2048x2048に向上")
        self._log("     - より高品質なパーツ抽出が可能に")
        self._log(f"  🔧 動的境界検出: {'有効' if Config.DYNAMIC_BOUNDARY_DETECTION else '無効'}")
        self._log(f"  📐 探索範囲(左右): {Config.BOUNDARY_SEARCH_RANGE_LR}")
        self._log(f"  📐 探索範囲(上下): {Config.BOUNDARY_SEARCH_RANGE_TB}")
        self._log("=" * 70)
    
    def _select_font(self):
        path = filedialog.askopenfilename(
            title="フォントファイルを選択",
            filetypes=[("フォントファイル", "*.ttf *.otf *.ttc"), ("すべてのファイル", "*.*")]
        )
        if path:
            self.font_path = path
            self.font_index = 0  # デフォルトは最初のフォント

            # TTCファイルの場合はフォントインデックスを選択
            if path.lower().endswith('.ttc'):
                self.font_index = self._select_ttc_font_index(path)

            font_info = os.path.basename(path)
            if self.font_index > 0:
                font_info += f" (フォント #{self.font_index})"

            self.font_label.config(text=font_info, foreground="black")
            self._log(f"\n✅ フォント選択: {path}")
            if self.font_index > 0:
                self._log(f"   フォントインデックス: {self.font_index}")

    def _select_ttc_font_index(self, ttc_path):
        """TTCファイルからフォントインデックスを選択"""
        # TTCファイルに含まれるフォント数を調べる
        font_count = self._get_ttc_font_count(ttc_path)

        if font_count <= 1:
            return 0

        # ダイアログでフォントインデックスを選択
        dialog = tk.Toplevel(self.root)
        dialog.title("TTCフォント選択")
        dialog.geometry("400x200")
        dialog.transient(self.root)
        dialog.grab_set()

        ttk.Label(dialog, text=f"このTTCファイルには{font_count}個のフォントが含まれています。\n使用するフォントを選択してください:").pack(pady=10)

        # プレビューフレーム
        preview_frame = ttk.Frame(dialog)
        preview_frame.pack(pady=10)

        selected_index = tk.IntVar(value=0)

        for i in range(min(font_count, 10)):  # 最大10個まで表示
            name = self._get_font_name_from_ttc(ttc_path, i)
            label = f"フォント {i}: {name}" if name else f"フォント {i}"
            ttk.Radiobutton(preview_frame, text=label, variable=selected_index, value=i).pack(anchor=tk.W)

        if font_count > 10:
            ttk.Label(preview_frame, text=f"... 他 {font_count - 10} 個のフォント", foreground="gray").pack(anchor=tk.W)

        # OKボタン
        ttk.Button(dialog, text="OK", command=dialog.destroy).pack(pady=10)

        dialog.wait_window()
        return selected_index.get()

    def _get_ttc_font_count(self, ttc_path):
        """TTCファイルに含まれるフォント数を取得"""
        try:
            count = 0
            while True:
                try:
                    ImageFont.truetype(ttc_path, 20, index=count)
                    count += 1
                    if count > 100:  # 安全のため上限を設定
                        break
                except:
                    break
            return max(count, 1)
        except:
            return 1

    def _get_font_name_from_ttc(self, ttc_path, index):
        """TTCファイルから指定インデックスのフォント名を取得"""
        try:
            font = ImageFont.truetype(ttc_path, 20, index=index)
            # フォント名を取得する試み（完全ではない）
            return f"インデックス {index}"
        except:
            return None
    
    def _select_output(self):
        path = filedialog.askdirectory(title="出力ディレクトリを選択")
        if path:
            self.output_dir = path
            self.output_label.config(text=path)
            self._log(f"\n📁 出力先変更: {path}")
    
    def _open_output(self):
        if os.path.exists(self.output_dir):
            if sys.platform == "darwin":
                os.system(f'open "{self.output_dir}"')
            elif sys.platform == "win32":
                os.startfile(self.output_dir)
            else:
                os.system(f'xdg-open "{self.output_dir}"')
        else:
            messagebox.showwarning("警告", "出力ディレクトリが存在しません")
    
    def _open_preview(self):
        if not os.path.exists(self.output_dir):
            messagebox.showwarning("警告", "まず抽出を実行してください")
            return
        if not self.font_path:
            messagebox.showwarning("警告", "フォントファイルを選択してください")
            return
        PartsPreviewWindow(self.root, self.output_dir, self.font_path, self.font_index)
    
    def _log(self, message):
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update()
    
    def _update_progress(self, current, total, message):
        self.progress_bar['maximum'] = total
        self.progress_bar['value'] = current
        self.progress_label.config(text=f"{message} ({current}/{total})")
        self.root.update()
    
    def _start_extraction(self):
        if self.is_running:
            return
        if not self.font_path:
            messagebox.showwarning("警告", "フォントファイルを選択してください")
            return
        
        self.is_running = True
        self.extract_button.config(state=tk.DISABLED)
        self.progress_bar['value'] = 0
        
        self._log("\n" + "=" * 70)
        self._log("抽出開始...")
        self._log("=" * 70)
        
        thread = threading.Thread(target=self._run_extraction, daemon=True)
        thread.start()
    
    def _run_extraction(self):
        try:
            stats = extract_all_parts(
                self.font_path,
                self.output_dir,
                progress_callback=self._update_progress,
                log_callback=self._log,
                font_index=self.font_index
            )
            
            self.root.after(0, lambda: messagebox.showinfo(
                "完了",
                f"抽出完了\n\n✅ 成功: {stats['success']}\n❌ 失敗: {stats['failed']}\n\n保存先: {self.output_dir}"
            ))
        except Exception as e:
            self._log(f"\n❌ エラー: {e}")
            import traceback
            self._log(traceback.format_exc())
            self.root.after(0, lambda: messagebox.showerror("エラー", f"抽出中にエラーが発生しました:\n{e}"))
        finally:
            self.is_running = False
            self.root.after(0, lambda: self.extract_button.config(state=tk.NORMAL))
            self.root.after(0, lambda: self.progress_label.config(text="完了"))

# ============================================================
# [BLOCK5-END]
# ============================================================











# ============================================================
# [BLOCK6-BEGIN] メインエントリポイント (2025-10-10)
# ============================================================

def main():
    try:
        root = tk.Tk()
        root.geometry("900x750")
        
        app = PartsExtractorGUI(root)
        
        if sys.platform == "darwin":
            root.createcommand("tk::mac::Quit", root.quit)
        
        root.mainloop()
    except Exception as e:
        print(f"[FATAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

# ============================================================
# [BLOCK6-END]
# ============================================================
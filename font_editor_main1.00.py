#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
フォントエディタ - ハイブリッド方式 v1.01
ビットマップ編集 → BDF/TTF書き出し対応
作成日: 2025-10-03
更新日: 2025-10-11 スレッド安全性・参照管理・型ヒント改善
"""

# === 標準ライブラリ ===
import os
import json
import threading
import queue
import subprocess
import tempfile
import re
import shutil
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple, Set, List, Callable, Any
from types import MethodType
from contextlib import contextmanager

# === サードパーティライブラリ ===
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

from PIL import Image, ImageDraw, ImageFont, ImageTk, ImageChops








# ===== [BLOCK1-BEGIN] コンフィグ・定数定義 (2025-10-17: 高解像度対応・フォント制作最適化) =====

class Config:
    """エディタ設定"""
    
    # ===== 解像度設定 (2025-10-17: 高品質フォント制作用に2048px) =====
    CANVAS_SIZE = 2048  # 編集キャンバスサイズ (px) - 高品質フォント制作用
    GRID_THUMB_SIZE = 128  # グリッド表示時のサムネイルサイズ
    TARGET_DPI = 300  # 目標DPI
    
    # ===== フォントレンダリング設定 (2025-10-17: 2048px用に最適化) =====
    FONT_RENDER_SIZE = 1800  # 2048pxキャンバス用の最適フォントサイズ (約88%使用)
    MIN_BLACK_PIXELS = 50  # ブランクグリフ判定の最小黒ピクセル数
    
    # ===== PNG書き出し設定 (2025-10-17: デフォルト2048px) =====
    DEFAULT_PNG_EXPORT_SIZE = 2048  # PNG書き出し時のデフォルトサイズ
    
    # ===== TTF書き出し設定 =====
    ASCENT_RATIO = 0.8  # アセント比率
    DESCENT_RATIO = 0.2  # ディセント比率
    NOTDEF_MARGIN_RATIO = 0.1  # .notdefグリフの外枠マージン比率
    NOTDEF_INNER_MARGIN_RATIO = 0.15  # .notdefグリフの内枠マージン比率
    
    # ===== ウィンドウ設定 =====
    WINDOW_WIDTH = 1400  # メインウィンドウ幅
    WINDOW_HEIGHT = 900  # メインウィンドウ高さ
    
    # ===== グリッド表示設定 =====
    GRID_COLUMNS = 8  # グリッド表示の列数
    
    # ===== デフォルト設定 =====
    DEFAULT_RANGE = '基本ラテン文字 (ASCII)'  # デフォルト文字範囲
    
    # ===== 文字コード範囲プリセット =====
    CHAR_RANGES = {
        '基本ラテン文字 (ASCII)': (0x0020, 0x007F),  # 95文字
        'ラテン補助文字': (0x0080, 0x00FF),  # 128文字
        'ひらがな': (0x3040, 0x309F),  # 96文字
        'カタカナ': (0x30A0, 0x30FF),  # 96文字
        # 漢字を500文字単位に細分化
        '漢字 1/20 (一～乞)': (0x4E00, 0x4FFF),  # 512文字
        '漢字 2/20 (乢～你)': (0x5000, 0x51FF),  # 512文字
        '漢字 3/20 (倀～傿)': (0x5200, 0x53FF),  # 512文字
        '漢字 4/20 (僀～势)': (0x5400, 0x55FF),  # 512文字
        '漢字 5/20 (匀～呿)': (0x5600, 0x57FF),  # 512文字
        '漢字 6/20 (唀～哿)': (0x5800, 0x59FF),  # 512文字
        '漢字 7/20 (喀～嗿)': (0x5A00, 0x5BFF),  # 512文字
        '漢字 8/20 (嘀～囿)': (0x5C00, 0x5DFF),  # 512文字
        '漢字 9/20 (圀～夿)': (0x5E00, 0x5FFF),  # 512文字
        '漢字 10/20 (央～奿)': (0x6000, 0x61FF),  # 512文字
        '漢字 11/20 (妀～嫿)': (0x6200, 0x63FF),  # 512文字
        '漢字 12/20 (嬀～尿)': (0x6400, 0x65FF),  # 512文字
        '漢字 13/20 (局～峿)': (0x6600, 0x67FF),  # 512文字
        '漢字 14/20 (崀～帿)': (0x6800, 0x69FF),  # 512文字
        '漢字 15/20 (幀～廿)': (0x6A00, 0x6BFF),  # 512文字
        '漢字 16/20 (开～忿)': (0x6C00, 0x6DFF),  # 512文字
        '漢字 17/20 (怀～懿)': (0x6E00, 0x6FFF),  # 512文字
        '漢字 18/20 (戀～揿)': (0x7000, 0x71FF),  # 512文字
        '漢字 19/20 (搀～政)': (0x7200, 0x73FF),  # 512文字
        '漢字 20/20 (收～瓿)': (0x7400, 0x77FF),  # 1024文字
        'CJK統合漢字拡張A': (0x3400, 0x4DBF),  # 6,592文字
        '記号・約物': (0x2000, 0x206F),  # 112文字
        '全角記号': (0xFF00, 0xFFEF),  # 240文字
        '数学記号': (0x2200, 0x22FF),  # 256文字
        '矢印': (0x2190, 0x21FF),  # 112文字
        'ギリシャ文字': (0x0370, 0x03FF),  # 144文字
        'キリル文字': (0x0400, 0x04FF),  # 256文字
        '絵文字1': (0x1F300, 0x1F5FF),  # 768文字
        '絵文字2': (0x1F600, 0x1F64F),  # 80文字
    }
    
    # ===== UI設定 =====
    COLOR_BG = '#F0F0F0'  # 背景色
    COLOR_ACTIVE = '#ADD8E6'  # アクティブボタン色
    COLOR_CANVAS = '#FFFFFF'  # キャンバス背景色
    COLOR_EMPTY = '#FFE0E0'  # 空グリフ背景色
    
    # グリッド設定 (2025-10-17: 2048px用に調整)
    GRID_SPACING = 64  # グリッド線の間隔 (px) - 2048/32 = 64px間隔
    GRID_COLOR = '#E0E0E0'  # グリッド線の色
    GRID_CENTER_COLOR = '#FF0000'  # 中央線の色
    
    # ナビゲーション設定
    NAV_SIZE = 150  # ナビゲーションウィンドウのサイズ

# ===== [BLOCK1-END] =====










# ===== [BLOCK2-BEGIN] データモデル (2025-01-15: 異体字マッピング機能追加) =====

class GlyphData:
    """1文字分のグリフデータ"""
    
    def __init__(self, char_code: int, bitmap: Optional[Image.Image] = None, is_edited: bool = False):
        self.char_code = char_code
        self.bitmap = bitmap
        self.is_empty = bitmap is None
        self.is_edited = is_edited
        self.mapping_char = None  # [ADD] 2025-01-15: 読みマッピング
    
    def get_char(self) -> str:
        """文字コードから文字を取得"""
        # [ADD] 2025-01-15: マッピング文字があればそれを返す
        if self.mapping_char:
            return self.mapping_char
        try:
            return chr(self.char_code)
        except ValueError:
            return ''
    
    def get_code_label(self) -> str:
        """U+XXXX形式のラベル取得"""
        # [ADD] 2025-01-15: マッピング表示
        label = f'U+{self.char_code:04X}'
        if self.mapping_char:
            label += f' [{self.mapping_char}]'
        return label
    
    def set_mapping(self, char: str) -> None:
        """読みマッピングを設定 (2025-01-15: 新規追加)"""  # [ADD]
        self.mapping_char = char if char else None
    
    def get_mapping(self) -> Optional[str]:
        """読みマッピングを取得 (2025-01-15: 新規追加)"""  # [ADD]
        return self.mapping_char


class FontProject:
    """フォントプロジェクト管理"""
    
    def __init__(self):
        self.glyphs: Dict[int, GlyphData] = {}
        self.font_path: Optional[str] = None
        self.original_ttf_path: Optional[str] = None
        self.char_range: Tuple[int, int] = Config.CHAR_RANGES[Config.DEFAULT_RANGE]
        self.clipboard: Optional[Image.Image] = None
        self.loaded_ranges: Set[Tuple[int, int]] = set()
        self._lock = threading.Lock()
        self.glyph_mappings: Dict[int, str] = {}  # [ADD] 2025-01-15: 異体字マッピング

        # [ADD] 2025-10-23: 偏旁エディタ統合用のパーツ辞書。
        # キーは偏旁名、値は辞書 { 'image': Image.Image, 'meta': dict } を想定。
        self.parts: Dict[str, Dict[str, Any]] = {}

    @property
    def dirty(self) -> bool:
        """未保存判定"""
        try:
            return any(getattr(g, 'is_edited', False) for g in self.glyphs.values())
        except Exception:
            return False

    def save_project(self, folder_path: str, progress_callback: Optional[Callable[[int, int, str], None]] = None):
        """プロジェクト保存（*.fproj）(2025-11-10: 進捗バー対応)"""
        import os, json
        from PIL import Image
        os.makedirs(folder_path, exist_ok=True)
        os.makedirs(os.path.join(folder_path, 'glyphs'), exist_ok=True)

        # 総ステップ数を計算
        glyphs_with_bitmap = [g for g in self.glyphs.values() if getattr(g, 'bitmap', None) is not None]
        parts_count = len(getattr(self, 'parts', {}))
        total_steps = 1 + len(glyphs_with_bitmap) + parts_count  # メタデータ + グリフ + パーツ

        current_step = 0

        # [ADD] 2025-01-15: マッピング情報を保存
        mappings = {}
        for code, glyph in self.glyphs.items():
            if hasattr(glyph, 'mapping_char') and glyph.mapping_char:
                mappings[code] = glyph.mapping_char

        meta = {
            'font_path': getattr(self, 'font_path', None),
            'original_ttf_path': getattr(self, 'original_ttf_path', None),
            'char_range': list(getattr(self, 'char_range', (0,0))),
            'edited_codes': [code for code, g in self.glyphs.items() if getattr(g, 'is_edited', False)],
            'glyph_mappings': mappings  # [ADD] 2025-01-15
        }
        with open(os.path.join(folder_path, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        current_step += 1
        if progress_callback:
            progress_callback(current_step, total_steps, 'メタデータを保存中...')

        # グリフ保存
        for code, g in self.glyphs.items():
            bmp = getattr(g, 'bitmap', None)
            if bmp is None:
                continue
            fn = os.path.join(folder_path, 'glyphs', f'U+{code:04X}.png')
            bmp.save(fn, 'PNG')

            current_step += 1
            if progress_callback and current_step % 10 == 0:  # 10個ごとに更新
                progress_callback(current_step, total_steps, f'グリフ保存中... ({current_step}/{total_steps})')

        # 最後のグリフ保存完了を報告
        if progress_callback:
            progress_callback(current_step, total_steps, 'グリフ保存完了')

        # [ADD] 2025-10-23: 偏旁エディタ用パーツデータを保存
        if getattr(self, 'parts', None):
            parts_dir = os.path.join(folder_path, 'parts')
            os.makedirs(parts_dir, exist_ok=True)
            parts_meta = {}
            for name, info in self.parts.items():
                img = info.get('image')
                meta = info.get('meta', {})
                if img is None:
                    continue
                # ファイル名は偏旁名をそのまま使用 (UTF-8で扱えるようエスケープ不要とする)
                part_fn = os.path.join(parts_dir, f'{name}.png')
                try:
                    img.save(part_fn, 'PNG')
                except Exception:
                    # 失敗した場合はスキップ
                    continue
                parts_meta[name] = meta

                current_step += 1
                if progress_callback:
                    progress_callback(current_step, total_steps, f'パーツ保存中... {name}')

            # 偏旁のメタデータをJSONで保存
            if parts_meta:
                with open(os.path.join(parts_dir, 'metadata.json'), 'w', encoding='utf-8') as pf:
                    json.dump(parts_meta, pf, ensure_ascii=False, indent=2)

        # 完了
        if progress_callback:
            progress_callback(total_steps, total_steps, '保存完了')

    def load_project(self, folder_path: str):
        """プロジェクト読込"""
        import os, json
        from PIL import Image
        with open(os.path.join(folder_path, 'metadata.json'), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        self.font_path = meta.get('font_path')
        self.original_ttf_path = meta.get('original_ttf_path')
        cr = meta.get('char_range')
        if isinstance(cr, list) and len(cr) == 2:
            self.char_range = (int(cr[0]), int(cr[1]))
        
        # [ADD] 2025-01-15: マッピング情報を読込
        self.glyph_mappings = {}
        mappings = meta.get('glyph_mappings', {})
        
        self.glyphs.clear()
        edited = set(meta.get('edited_codes', []))
        glyph_dir = os.path.join(folder_path, 'glyphs')
        if os.path.isdir(glyph_dir):
            for name in os.listdir(glyph_dir):
                if not name.lower().endswith('.png'):
                    continue
                try:
                    codepoint = int(name[2:6], 16)
                except Exception:
                    continue
                img = Image.open(os.path.join(glyph_dir, name)).convert('L')
                glyph = GlyphData(codepoint, img, is_edited=(codepoint in edited))
                
                # [ADD] 2025-01-15: マッピングを設定
                if str(codepoint) in mappings:
                    glyph.set_mapping(mappings[str(codepoint)])
                
                self.glyphs[codepoint] = glyph

        # [ADD] 2025-10-23: 偏旁エディタ用パーツデータを読み込む
        # 保存時に parts ディレクトリに保存されていれば読み出す
        self.parts = {}
        parts_dir = os.path.join(folder_path, 'parts')
        if os.path.isdir(parts_dir):
            # メタデータを読み込む
            meta_path = os.path.join(parts_dir, 'metadata.json')
            parts_meta = {}
            if os.path.isfile(meta_path):
                try:
                    with open(meta_path, 'r', encoding='utf-8') as pf:
                        parts_meta = json.load(pf)
                except Exception:
                    parts_meta = {}
            # 個々のパーツ画像を読み込み、辞書に格納
            for fname in os.listdir(parts_dir):
                if not fname.lower().endswith('.png'):
                    continue
                part_name = os.path.splitext(fname)[0]
                img_path = os.path.join(parts_dir, fname)
                try:
                    img = Image.open(img_path).convert('L')
                except Exception:
                    continue
                # メタデータを取得（存在しない場合は空辞書）
                meta = parts_meta.get(part_name, {}) if isinstance(parts_meta, dict) else {}
                self.parts[part_name] = {'image': img, 'meta': meta}

    def set_range(self, range_name: str):
        """文字範囲を設定（表示フィルタのみ、データは保持）"""
        if range_name in Config.CHAR_RANGES:
            self.char_range = Config.CHAR_RANGES[range_name]
    
    def get_char_codes(self) -> list:
        """現在の範囲の文字コードリストを取得"""
        start, end = self.char_range
        return list(range(start, end + 1))
    
    def get_empty_count(self) -> int:
        """空白グリフ数をカウント（現在の範囲のみ）"""
        return sum(1 for code in self.get_char_codes() 
                  if code in self.glyphs and self.glyphs[code].is_empty)
    
    def set_glyph(self, char_code: int, bitmap: Image.Image, is_edited: bool = False):
        """グリフを設定"""
        glyph = GlyphData(char_code, bitmap, is_edited)
        # [ADD] 2025-01-15: 既存のマッピングを保持
        if char_code in self.glyphs and hasattr(self.glyphs[char_code], 'mapping_char'):
            glyph.set_mapping(self.glyphs[char_code].mapping_char)
        self.glyphs[char_code] = glyph
    
    def set_glyph_mapping(self, char_code: int, mapping_char: str):
        """グリフにマッピングを設定 (2025-01-15: 新規追加)"""  # [ADD]
        if char_code in self.glyphs:
            self.glyphs[char_code].set_mapping(mapping_char)
            self.glyph_mappings[char_code] = mapping_char
        else:
            # グリフが存在しない場合は空のグリフを作成
            glyph = GlyphData(char_code, None, False)
            glyph.set_mapping(mapping_char)
            self.glyphs[char_code] = glyph
            self.glyph_mappings[char_code] = mapping_char
    
    def mark_as_edited(self, char_code: int):
        """グリフを編集済みとしてマーク"""
        if char_code in self.glyphs:
            self.glyphs[char_code].is_edited = True
    
    def get_edited_glyphs(self) -> list:
        """編集済みグリフのリストを取得"""
        return [(code, glyph) for code, glyph in self.glyphs.items() 
                if not glyph.is_empty and glyph.is_edited]
    
    def is_range_loaded(self, range_tuple: Tuple[int, int]) -> bool:
        """指定範囲が読み込み済みか確認"""
        return range_tuple in self.loaded_ranges
    
    def mark_range_loaded(self, range_tuple: Tuple[int, int]):
        """範囲を読み込み済みとしてマーク"""
        self.loaded_ranges.add(range_tuple)

# ===== [BLOCK2-END] =====










# ===== [BLOCK3-BEGIN] フォント読み込み・レンダリング (2025-10-11: 型ヒント追加、定数使用) =====

class FontRenderer:
    """フォントレンダリング処理"""
    
    @staticmethod
    def load_font(
        font_path: str,
        char_codes: List[int],
        project: FontProject,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        font_index: int = 0
    ) -> bool:
        """フォントを読み込んで各文字をレンダリング (2025-10-11: 型ヒント追加、2025-11-10: TTC対応)"""
        try:
            # PIL ImageFontでフォント読み込み (2025-10-11: 定数使用、2025-11-10: TTC対応)
            pil_font = ImageFont.truetype(font_path, size=Config.FONT_RENDER_SIZE, index=font_index)
            
            # 元のTTFパスを保存（マージ用）
            if not project.original_ttf_path:
                project.original_ttf_path = font_path
            
            total = len(char_codes)
            
            for idx, code in enumerate(char_codes):
                # 既に手動編集されたグリフはスキップ (2025-10-03)
                if code in project.glyphs and not project.glyphs[code].is_empty:
                    # プログレス更新のみ
                    if progress_callback and idx % 10 == 0:
                        progress_callback(idx + 1, total)
                    continue  # スキップ
                
                try:
                    char = chr(code)
                    
                    # 文字をレンダリング
                    bitmap = FontRenderer._render_char(char, pil_font)
                    
                    if bitmap:
                        project.set_glyph(code, bitmap, is_edited=False)  # 未編集としてマーク
                    else:
                        # 空グリフとして登録（既存がなければ）
                        with project._lock:  # (2025-10-11: スレッドセーフ化)
                            if code not in project.glyphs:
                                project.glyphs[code] = GlyphData(code, None, False)
                        
                except (ValueError, OSError):
                    # レンダリング失敗は空グリフ
                    with project._lock:  # (2025-10-11: スレッドセーフ化)
                        if code not in project.glyphs:
                            project.glyphs[code] = GlyphData(code, None, False)
                
                # プログレス更新 (2025-10-03: 10文字ごとに更新)
                if progress_callback and idx % 10 == 0:
                    progress_callback(idx + 1, total)
                    
            # 最終プログレス
            if progress_callback:
                progress_callback(total, total)
            
            project.font_path = font_path
            return True
            
        except Exception as e:
            messagebox.showerror("読み込みエラー", f"フォント読み込み失敗:\n{e}")
            return False
    
    @staticmethod
    def _render_char(char: str, font: ImageFont.FreeTypeFont) -> Optional[Image.Image]:
        """1文字をビットマップ化 (2025-10-11: 定数使用)"""
        try:
            # バウンディングボックス取得
            bbox = font.getbbox(char)
            if bbox[2] - bbox[0] == 0 or bbox[3] - bbox[1] == 0:
                return None  # 空グリフ
            
            # 768x768キャンバス作成
            canvas = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 255)
            draw = ImageDraw.Draw(canvas)
            
            # 中央配置計算
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
            x = (Config.CANVAS_SIZE - w) // 2 - bbox[0]
            y = (Config.CANVAS_SIZE - h) // 2 - bbox[1]
            
            # 描画
            draw.text((x, y), char, font=font, fill=0)
            
            # ブランクグリフ検出（枠だけで中身が空白の場合）
            pixels = canvas.load()
            black_pixels = sum(1 for py in range(Config.CANVAS_SIZE)
                             for px in range(Config.CANVAS_SIZE)
                             if pixels[px, py] < 128)
            
            # 黒ピクセルが少なすぎる場合はブランクグリフと判定 (2025-10-11: 定数使用)
            if black_pixels < Config.MIN_BLACK_PIXELS:
                return None
            
            return canvas
            
        except Exception:
            return None

# ===== [BLOCK3-END] =====










# ===== [BLOCK4-BEGIN] グリッドビューGUI (2025-01-15: マッピング機能追加、PhotoImage参照保持改善、型ヒント追加) =====

class GridView(tk.Frame):
    """グリッド一覧表示"""
    
    def __init__(
        self, 
        parent: tk.Widget, 
        project: FontProject, 
        on_click_callback: Callable[[int], None]
    ) -> None:
        super().__init__(parent, bg=Config.COLOR_BG)
        self.project: FontProject = project
        self.on_click: Callable[[int], None] = on_click_callback
        self.thumb_cache: Dict[int, ImageTk.PhotoImage] = {}  # サムネイルキャッシュ
        self._photo_refs: List[ImageTk.PhotoImage] = []  # (2025-10-11: GC対策で明示的リスト保持)
        
        # スクロール可能なキャンバス
        self.canvas = tk.Canvas(self, bg=Config.COLOR_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient='vertical', command=self.canvas.yview)
        self.scrollable_frame = tk.Frame(self.canvas, bg=Config.COLOR_BG)
        
        self.scrollable_frame.bind(
            '<Configure>',
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox('all'))
        )
        
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor='nw')
        self.canvas.configure(yscrollcommand=scrollbar.set)
        
        self.canvas.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
        # マウスホイールスクロール対応 (2025-10-03: 修正)
        self.canvas.bind('<MouseWheel>', self._on_mousewheel)  # Windows/Mac
        self.canvas.bind('<Button-4>', self._on_mousewheel)  # Linux上スクロール
        self.canvas.bind('<Button-5>', self._on_mousewheel)  # Linux下スクロール
        self.scrollable_frame.bind('<MouseWheel>', self._on_mousewheel)
        self.scrollable_frame.bind('<Button-4>', self._on_mousewheel)
        self.scrollable_frame.bind('<Button-5>', self._on_mousewheel)
        
        self.filter: str = 'all'  # 初期フィルタ
    
    def _on_mousewheel(self, event: tk.Event) -> None:
        """マウスホイールでスクロール"""
        if event.num == 5 or event.delta < 0:
            # 下にスクロール
            self.canvas.yview_scroll(1, 'units')
        elif event.num == 4 or event.delta > 0:
            # 上にスクロール
            self.canvas.yview_scroll(-1, 'units')
    
    def set_filter(self, filter_type: str) -> None:
        """フィルタを設定して再描画"""
        self.filter = filter_type
        self.refresh()
    
    def refresh(self) -> None:
        """グリッド再描画 (2025-10-11: PhotoImage参照保持改善)"""
        # 既存ウィジェット削除
        for widget in self.scrollable_frame.winfo_children():
            widget.destroy()
        
        self.thumb_cache.clear()
        self._photo_refs.clear()  # (2025-10-11: 参照リストもクリア)
        
        # 固定列数を使用 (2025-10-04: 動的計算を削除)
        columns = Config.GRID_COLUMNS
        
        # グリッド生成
        char_codes = self.project.get_char_codes()
        
        # フィルタ適用
        filtered = []
        for code in char_codes:
            g = self.project.glyphs.get(code)
            if self.filter == 'all':
                filtered.append(code)
            elif self.filter == 'edited':
                if g and not g.is_empty and g.is_edited:
                    filtered.append(code)
            elif self.filter == 'unedited':
                if g and not g.is_empty and not g.is_edited:
                    filtered.append(code)
            elif self.filter == 'empty':
                if (g is None) or g.is_empty:
                    filtered.append(code)
            elif self.filter == 'defined':
                if g and not g.is_empty:
                    filtered.append(code)
        char_codes = filtered

        for idx, code in enumerate(char_codes):
            row = idx // columns
            col = idx % columns
            
            self._create_cell(code, row, col)
        
        # スクロール領域を更新
        self.scrollable_frame.update_idletasks()
        self.canvas.configure(scrollregion=self.canvas.bbox('all'))
    
    def destroy(self) -> None:
        """ウィジェット破棄時の処理"""
        # 個別バインドは自動的に解除されるので、特別な処理不要
        super().destroy()
    
    def _create_cell(self, char_code: int, row: int, col: int) -> None:
        """1セル作成 (2025-01-15: マッピング表示対応)"""  # [ADD]
        frame = tk.Frame(
            self.scrollable_frame,
            bg=Config.COLOR_BG,
            relief='solid',
            borderwidth=1,
            padx=5,
            pady=5
        )
        frame.grid(row=row, column=col, padx=2, pady=2)
        
        # グリフデータ取得（存在しない場合は空グリフとして扱う）
        glyph = self.project.glyphs.get(char_code)
        
        if glyph and not glyph.is_empty:
            # サムネイル生成
            thumb = glyph.bitmap.resize(
                (Config.GRID_THUMB_SIZE, Config.GRID_THUMB_SIZE),
                Image.Resampling.LANCZOS
            )
            photo = ImageTk.PhotoImage(thumb)
            self.thumb_cache[char_code] = photo  # 辞書に保持
            self._photo_refs.append(photo)  # (2025-10-11: リストにも保持してGC防止)
            
            label = tk.Label(frame, image=photo, bg=Config.COLOR_BG)
            label.image = photo  # (2025-10-11: ラベル自体にも参照を持たせる)
        else:
            # 空グリフ (2025-10-03: 文字プレビュー追加)
            try:
                char_preview = chr(char_code)
                display_text = f'[空]\n{char_preview}'
            except ValueError:
                display_text = '[空]'
            
            label = tk.Label(
                frame,
                text=display_text,
                bg=Config.COLOR_EMPTY,
                width=10,
                height=5,
                font=('Arial', 20),
                relief='sunken'
            )
        
        label.pack()
        
        # 文字コードラベル + 文字表示 (2025-01-15: マッピング表示追加)  # [ADD]
        try:
            char_display = chr(char_code) if char_code < 0x10000 else ''
            label_text = f'U+{char_code:04X} {char_display}'
            
            # マッピングがある場合は表示
            if glyph and hasattr(glyph, 'mapping_char') and glyph.mapping_char:
                label_text += f'\n[{glyph.mapping_char}]'
                
        except ValueError:
            label_text = f'U+{char_code:04X}'
            if glyph and hasattr(glyph, 'mapping_char') and glyph.mapping_char:
                label_text += f'\n[{glyph.mapping_char}]'
        
        code_label = tk.Label(
            frame,
            text=label_text,
            bg=Config.COLOR_BG,
            font=('Arial', 8),
            fg='blue' if (glyph and hasattr(glyph, 'mapping_char') and glyph.mapping_char) else 'black'  # [ADD] マッピングがある場合は青色
        )
        code_label.pack()
        
        # クリックイベント
        frame.bind('<Button-1>', lambda e, c=char_code: self.on_click(c))
        label.bind('<Button-1>', lambda e, c=char_code: self.on_click(c))
        
        # 右クリックメニュー (2025-10-03)
        frame.bind('<Button-2>', lambda e, c=char_code: self._show_context_menu(e, c))
        label.bind('<Button-2>', lambda e, c=char_code: self._show_context_menu(e, c))
        # Windows/Mac用の右クリック
        frame.bind('<Button-3>', lambda e, c=char_code: self._show_context_menu(e, c))  # [ADD]
        label.bind('<Button-3>', lambda e, c=char_code: self._show_context_menu(e, c))  # [ADD]
    
    def _show_context_menu(self, event: tk.Event, char_code: int) -> None:
        """右クリックメニュー表示 (2025-01-15: マッピング機能追加)"""  # [ADD]
        menu = tk.Menu(self, tearoff=0)
        
        glyph = self.project.glyphs.get(char_code)
        
        if glyph and not glyph.is_empty:
            menu.add_command(
                label=f'U+{char_code:04X} をPNG保存',
                command=lambda: self._save_glyph_png(char_code)
            )
        
        menu.add_command(
            label='編集',
            command=lambda: self.on_click(char_code)
        )
        
        # [ADD] 2025-01-15: マッピング設定
        menu.add_separator()
        menu.add_command(
            label='読みマッピングを設定...',
            command=lambda: self._set_glyph_mapping(char_code)
        )
        
        if glyph and hasattr(glyph, 'mapping_char') and glyph.mapping_char:
            menu.add_command(
                label=f'マッピング解除 [{glyph.mapping_char}]',
                command=lambda: self._clear_glyph_mapping(char_code)
            )
        
        menu.tk_popup(event.x_root, event.y_root)
    
    def _save_glyph_png(self, char_code: int) -> None:
        """グリフをPNG保存"""
        glyph = self.project.glyphs.get(char_code)
        if glyph and not glyph.is_empty:
            default_name = f'U+{char_code:04X}.png'
            path = filedialog.asksaveasfilename(
                title='PNG保存',
                defaultextension='.png',
                initialfile=default_name,
                filetypes=[('PNG Image', '*.png'), ('All Files', '*.*')]
            )
            
            if path:
                glyph.bitmap.save(path)
                messagebox.showinfo('保存完了', f'保存しました:\n{path}')
    
    def _set_glyph_mapping(self, char_code: int) -> None:
        """グリフマッピングを設定 (2025-01-15: 新規追加)"""  # [ADD]
        dialog = tk.Toplevel(self)
        dialog.title('読みマッピング設定')
        dialog.geometry('300x150')
        dialog.transient(self)
        dialog.grab_set()  # モーダル化
        
        tk.Label(dialog, text=f'U+{char_code:04X} の読みを設定:', font=('Arial', 11)).pack(pady=10)
        
        # 現在の文字を表示
        try:
            current_char = chr(char_code)
            tk.Label(dialog, text=f'元の文字: {current_char}', font=('Arial', 10), fg='gray').pack()
        except ValueError:
            pass
        
        entry = tk.Entry(dialog, font=('Arial', 14), width=20)
        entry.pack(pady=10)
        
        # 既存のマッピングがあれば表示
        glyph = self.project.glyphs.get(char_code)
        if glyph and hasattr(glyph, 'mapping_char') and glyph.mapping_char:
            entry.insert(0, glyph.mapping_char)
        
        entry.focus()
        entry.select_range(0, tk.END)
        
        def apply():
            mapping = entry.get().strip()
            if mapping:
                self.project.set_glyph_mapping(char_code, mapping)
                self.refresh()
                dialog.destroy()
                messagebox.showinfo('設定完了', f'U+{char_code:04X} に「{mapping}」を設定しました')
            else:
                messagebox.showwarning('警告', '読みを入力してください')
        
        button_frame = tk.Frame(dialog)
        button_frame.pack(pady=10)
        
        tk.Button(button_frame, text='設定', command=apply, width=10).pack(side='left', padx=5)
        tk.Button(button_frame, text='キャンセル', command=dialog.destroy, width=10).pack(side='left', padx=5)
        
        entry.bind('<Return>', lambda e: apply())
        dialog.bind('<Escape>', lambda e: dialog.destroy())
    
    def _clear_glyph_mapping(self, char_code: int) -> None:
        """グリフマッピングをクリア (2025-01-15: 新規追加)"""  # [ADD]
        if messagebox.askyesno('確認', f'U+{char_code:04X} のマッピングを解除しますか？'):
            glyph = self.project.glyphs.get(char_code)
            if glyph:
                glyph.set_mapping(None)
                if char_code in self.project.glyph_mappings:
                    del self.project.glyph_mappings[char_code]
            self.refresh()
            messagebox.showinfo('解除完了', f'U+{char_code:04X} のマッピングを解除しました')

# ===== [BLOCK4-END] =====











# ===== [BLOCK5-BEGIN] 編集エディタGUI (2025-10-13: 基本部分) =====

class GlyphEditor(tk.Toplevel):
    """グリフ編集ウィンドウ(レイヤー方式テキスト挿入対応)"""
    
    def __init__(
        self, 
        parent: tk.Widget, 
        project: FontProject, 
        char_code: int, 
        on_save_callback: Callable[[], None]
    ) -> None:
        super().__init__(parent)
        self.project: FontProject = project
        self.char_code: int = char_code
        self.on_save: Callable[[], None] = on_save_callback
        self.glyph: Optional[GlyphData] = project.glyphs.get(char_code)
        
        # 編集用ビットマップ(作業用コピー) - ベースレイヤー
        if self.glyph and not self.glyph.is_empty:
            self.edit_bitmap: Image.Image = self.glyph.bitmap.copy()
        else:
            self.edit_bitmap: Image.Image = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 255)
        
        # テキストレイヤー (2025-10-12: 新機能)
        self.text_layer: Optional[Image.Image] = None
        self.text_layer_pos: Tuple[int, int] = (0, 0)
        self.text_layer_original: Optional[Image.Image] = None  # リサイズ用の元画像
        self.is_text_mode: bool = False
        self.text_input_dialog: Optional[tk.Toplevel] = None

        # [ADD] 2025-10-22: グリッド表示フラグとエッジマスク
        # キャンバスの白・赤のグリッド線の表示を切り替えるためのフラグ。初期状態では非表示。
        self.grid_visible_var: tk.BooleanVar = tk.BooleanVar(value=False)
        # テキスト挿入時のエッジ領域をプレビューで表示するためのマスク画像。
        # エッジが存在しない場合やエッジ幅が0の場合はNoneとなる。
        self.text_edge_mask: Optional[Image.Image] = None

        # [ADD] コミット用のエッジマスク。エッジを透過に置き換える際に使用する。
        self.text_edge_mask_commit: Optional[Image.Image] = None

        # [ADD] 2025-10-23: エッジ形状設定 ('sharp' または 'round')
        self.edge_style_var: tk.StringVar = tk.StringVar(value='sharp')
        
        # アンドゥ・リドゥ用履歴
        self.undo_stack: List[Image.Image] = []
        self.redo_stack: List[Image.Image] = []
        self._save_to_undo()
        
        # 描画ツール状態
        self.current_tool: str = 'pen'
        self.brush_size: int = 5
        self.is_drawing: bool = False
        self.last_x: Optional[int] = None
        self.last_y: Optional[int] = None
        
        # 選択領域
        self.selection_start: Optional[Tuple[int, int]] = None
        self.selection_end: Optional[Tuple[int, int]] = None
        self.selection_rect: Optional[int] = None
        self.selected_image: Optional[Image.Image] = None

        # 移動操作用フラグと座標
        self.is_moving: bool = False
        self.move_start_offset: Optional[Tuple[int, int]] = None
        self.move_current_pos: Optional[Tuple[int, int]] = None

        # 拡大縮小操作用フラグと座標
        self.is_resizing: bool = False
        self.resize_origin: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None
        self.resize_handle: Optional[str] = None
        self.resize_start_point: Optional[Tuple[int, int]] = None
        self.resize_preview_rect: Optional[Tuple[int, int, int, int]] = None

        # 図形ツール用座標
        self.shape_start: Optional[Tuple[int, int]] = None
        self.shape_end: Optional[Tuple[int, int]] = None

        # ガイドライン保存
        self.guidelines: List[Tuple[str, int]] = []

        # ナビゲーション用キャンバスのPhotoImage保持
        self._move_photo: Optional[ImageTk.PhotoImage] = None
        self._resize_photo: Optional[ImageTk.PhotoImage] = None
        self._nav_photo: Optional[ImageTk.PhotoImage] = None
        self._text_layer_photo: Optional[ImageTk.PhotoImage] = None

        # 背景チェックパターンおよび表示用PhotoImage
        # パターンは遅延生成とする。_bg_patternはタイル用の小さなチェック柄、
        # _bg_fullはキャンバス全体サイズにタイル貼りした画像を保持する。
        self._bg_pattern: Optional[Image.Image] = None
        self._bg_full: Optional[Image.Image] = None
        self._bg_photo: Optional[ImageTk.PhotoImage] = None
        
        # ブラシカーソル
        self.brush_cursor: Optional[int] = None
        
        # ズーム機能
        self.zoom_level: float = 1.0
        self.zoom_levels: List[float] = [0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        self.pan_offset: List[int] = [0, 0]
        self.is_panning: bool = False
        self.pan_start: Optional[Tuple[int, int]] = None
        
        # PhotoImage参照保持
        self.photo: Optional[ImageTk.PhotoImage] = None
        
        # ツールボタン管理
        self.tool_buttons: Dict[str, tk.Button] = {}
        
        # ドラッグ開始座標
        self.drag_start: Optional[Tuple[int, int]] = None
        
        # ハンドル描画用ID保存
        self.resize_handle_ids: List[int] = []
        
        self.title(f'編集: U+{char_code:04X}')
        self.geometry('1400x900')
        
        self._setup_ui()

        # 初期化時に背景パターンを生成
        # パターンサイズは8px単位で作成し、全体用のタイルも生成する
        # チェック柄のタイルサイズを大きくし、約30ピクセル四方の格子にする
        self._bg_pattern = self._create_bg_pattern(30)
        self._create_full_bg()

        # 初回プレビューを更新
        self._update_preview()

        # ウィンドウサイズに応じてズームレベルをフィットさせる
        # UI構築後に少し遅延させてキャンバスサイズが計算されてから調整する
        self.after(100, self._fit_zoom_to_window)
        
        # キーボードショートカット
        self.bind('<Control-s>', lambda e: self._save())
        self.bind('<Control-z>', lambda e: self._undo())
        self.bind('<Control-y>', lambda e: self._redo())
        self.bind('<Control-c>', lambda e: self._copy_selection())
        self.bind('<Control-x>', lambda e: self._cut_selection())
        self.bind('<Control-v>', lambda e: self._paste())
        self.bind('<Delete>', lambda e: self._delete_selection())
        self.bind('<Escape>', lambda e: self._clear_selection())
        self.bind('<KeyPress-space>', self._on_space_press)
        self.bind('<KeyRelease-space>', self._on_space_release)
        self.bind('<Control-0>', lambda e: self._reset_zoom())

        # 矢印キーによる1px単位移動
        self.bind('<Left>', lambda e: self._nudge(-1, 0))
        self.bind('<Right>', lambda e: self._nudge(1, 0))
        self.bind('<Up>', lambda e: self._nudge(0, -1))
        self.bind('<Down>', lambda e: self._nudge(0, 1))
    
    def _save_to_undo(self) -> None:
        """現在の状態をアンドゥスタックに保存"""
        self.undo_stack.append(self.edit_bitmap.copy())
        if len(self.undo_stack) > 50:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

    # ===== 背景チェックパターン関連 =====
    def _create_bg_pattern(self, tile_size: int = 30) -> Image.Image:
        """
        透過部分とキャンバス余白を見分けやすくするためのチェック柄を生成する。
        tile_sizeピクセル四方のタイルを2色で塗り分ける。
        少し濃淡の違うグレーを使用し、全体的に薄めのトーンにする。
        """
        # 2×2タイルを1つのパターンとして作成する
        pattern = Image.new('L', (tile_size * 2, tile_size * 2), 0)
        draw = ImageDraw.Draw(pattern)
        # 薄いグレーとやや濃いグレーを交互に塗り分ける
        light = 200
        dark = 180
        for by in range(2):
            for bx in range(2):
                color = light if (bx + by) % 2 == 0 else dark
                x0 = bx * tile_size
                y0 = by * tile_size
                x1 = x0 + tile_size
                y1 = y0 + tile_size
                draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=color)
        return pattern

    def _create_full_bg(self) -> None:
        """
        キャンバス全体サイズ(CANVAS_SIZE x CANVAS_SIZE)の背景パターン画像を生成する。
        すでに生成済みの場合は再生成しない。
        """
        if self._bg_pattern is None:
            # 後で生成される場合があるため、未設定なら何もしない
            return
        if self._bg_full is not None and self._bg_full.size == (Config.CANVAS_SIZE, Config.CANVAS_SIZE):
            return
        tile = self._bg_pattern
        w, h = Config.CANVAS_SIZE, Config.CANVAS_SIZE
        bg = Image.new('L', (w, h), 255)
        # タイルを繰り返し貼り付け
        for y in range(0, h, tile.height):
            for x in range(0, w, tile.width):
                bg.paste(tile, (x, y))
        self._bg_full = bg

    def _fit_zoom_to_window(self) -> None:
        """
        ウィンドウ内の表示領域に合わせてズームレベルを自動調整する。
        キャンバスがウィンドウからはみ出ないように、最適な倍率を選択する。
        """
        try:
            canvas_width = self.preview_canvas.winfo_width()
            canvas_height = self.preview_canvas.winfo_height()
        except Exception:
            return
        if canvas_width <= 1 or canvas_height <= 1:
            # ウィンドウ生成中は値がまだ取れないので再試行
            self.after(100, self._fit_zoom_to_window)
            return
        # キャンバスサイズに対するウィンドウサイズの比率を計算
        ratio_w = canvas_width / Config.CANVAS_SIZE
        ratio_h = canvas_height / Config.CANVAS_SIZE
        target_ratio = min(ratio_w, ratio_h)
        # 既存のズームレベルリストから最適な倍率を選択
        # 一番近いが小さめの倍率を選ぶことで余白を確保
        candidates = [z for z in self.zoom_levels if z <= target_ratio]
        if not candidates:
            # 全ての定義済み倍率よりも小さい場合は最小値を採用
            new_zoom = min(self.zoom_levels)
        else:
            new_zoom = max(candidates)
        if abs(self.zoom_level - new_zoom) > 1e-3:
            self.zoom_level = new_zoom
            self.zoom_label.config(text=f'{int(self.zoom_level * 100)}%')
            self._update_preview()
    
    def _setup_ui(self) -> None:
        """UI構築 (2025-10-12: テキスト挿入ツール追加)"""
        # ツールバー
        toolbar = tk.Frame(self, bg=Config.COLOR_BG)
        toolbar.pack(side='top', fill='x', padx=5, pady=5)
        
        tk.Button(toolbar, text='💾 保存', command=self._save).pack(side='left', padx=2)
        tk.Button(toolbar, text='📸 PNG保存', command=self._save_png).pack(side='left', padx=2)
        tk.Button(toolbar, text='📋 コピー', command=self._copy).pack(side='left', padx=2)
        tk.Button(toolbar, text='✂️ 切り取り', command=self._cut_selection).pack(side='left', padx=2)
        tk.Button(toolbar, text='📄 貼付', command=self._paste).pack(side='left', padx=2)
        tk.Button(toolbar, text='🗑️ クリア', command=self._clear).pack(side='left', padx=2)
        tk.Button(toolbar, text='⭕ 空白化', command=self._mark_as_empty).pack(side='left', padx=2)
        tk.Button(toolbar, text='🔥 他フォント読込', command=self._load_from_other_font).pack(side='left', padx=2)
        
        # ズームコントロール
        tk.Label(toolbar, text='🔍', bg=Config.COLOR_BG).pack(side='left', padx=(10, 0))
        tk.Button(toolbar, text='-', command=self._zoom_out, width=2).pack(side='left', padx=2)
        self.zoom_label = tk.Label(toolbar, text='100%', bg=Config.COLOR_BG, width=6)
        self.zoom_label.pack(side='left', padx=2)
        tk.Button(toolbar, text='+', command=self._zoom_in, width=2).pack(side='left', padx=2)
        tk.Button(toolbar, text='0', command=self._reset_zoom, width=2).pack(side='left', padx=2)
        
        # メインフレーム
        main_frame = tk.Frame(self, bg=Config.COLOR_BG)
        main_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # ===== 左側: ツールパネル（2列レイアウト） =====
        tool_panel_container = tk.Frame(main_frame, bg=Config.COLOR_BG)
        tool_panel_container.pack(side='left', fill='y', padx=(0, 10))
        
        tk.Label(tool_panel_container, text='ツール', bg=Config.COLOR_BG, font=('Arial', 12, 'bold')).pack(pady=5)
        
        # 2列レイアウト用フレーム
        tool_columns_frame = tk.Frame(tool_panel_container, bg=Config.COLOR_BG)
        tool_columns_frame.pack(fill='both', expand=True)
        
        # 左列: ツールボタン
        left_column = tk.Frame(tool_columns_frame, bg=Config.COLOR_BG, width=100)
        left_column.pack(side='left', fill='y', padx=(0, 5))
        
        # 右列: その他のコントロール
        right_column = tk.Frame(tool_columns_frame, bg=Config.COLOR_BG, width=200)
        right_column.pack(side='left', fill='both', expand=True)
        
        # ===== 左列: ツールボタン (2025-10-12: 文字挿入追加) =====
        tools = [
            ('✏️ ペン', 'pen'),
            ('📗 消しゴム', 'eraser'),
            ('🪣 塗りつぶし', 'fill'),
            ('選択', 'select'),
            ('✋ 移動', 'move'),
            ('🪜 拡大縮小', 'resize'),
            ('✒️ 文字挿入', 'text'),
            ('／ 直線', 'line'),
            ('□ 四角', 'rect'),
            ('○ 円', 'ellipse'),
            ('📐 ガイド', 'guide')
        ]

        for label, tool in tools:
            btn = tk.Button(
                left_column,
                text=label,
                command=lambda t=tool: self._set_tool(t),
                relief='sunken' if tool == 'pen' else 'raised',
                bg=Config.COLOR_ACTIVE if tool == 'pen' else Config.COLOR_BG,
                width=12
            )
            btn.pack(fill='x', padx=2, pady=2)
            self.tool_buttons[tool] = btn

        # ===== 右列: その他のコントロール =====
        
        # アンドゥ・リドゥボタン
        undo_redo_frame = tk.Frame(right_column, bg=Config.COLOR_BG)
        undo_redo_frame.pack(pady=(5, 0), fill='x')
        tk.Button(undo_redo_frame, text='↩️', command=self._undo, width=3).pack(side='left', padx=2)
        tk.Button(undo_redo_frame, text='↪️', command=self._redo, width=3).pack(side='left', padx=2)

        # 設定ボタン
        tk.Button(right_column, text='⚙ 設定', command=self._show_settings_dialog, width=10).pack(fill='x', padx=2, pady=(10, 2))

        # ナビゲーションウィンドウ
        nav_frame = tk.Frame(right_column, bg=Config.COLOR_BG)
        nav_frame.pack(pady=(10, 5))
        tk.Label(nav_frame, text='ナビ', bg=Config.COLOR_BG, font=('Arial', 10, 'bold')).pack()
        self.nav_canvas = tk.Canvas(nav_frame, width=Config.NAV_SIZE, height=Config.NAV_SIZE, bg='#F8F8F8', highlightthickness=1, highlightbackground='gray')
        self.nav_canvas.pack()
        # ナビゲーションクリックでスクロール移動を可能にする
        self.nav_canvas.bind('<Button-1>', self._on_nav_click)

        # [ADD] グリッド表示切り替え
        # 白線・赤線で構成されるグリッドのON/OFFを切り替えるチェックボックス。
        # 初期状態では非表示とし、チェック時にプレビューを更新する。
        grid_toggle = tk.Checkbutton(
            right_column,
            text='グリッド表示',
            variable=self.grid_visible_var,
            command=self._update_preview,
            bg=Config.COLOR_BG
        )
        grid_toggle.pack(anchor='w', padx=5, pady=(2, 5))
        
        # ブラシサイズ調整
        tk.Label(right_column, text='ブラシサイズ', bg=Config.COLOR_BG, font=('Arial', 10, 'bold')).pack(pady=(15, 5))
        
        self.brush_size_var = tk.IntVar(value=self.brush_size)
        brush_scale = tk.Scale(
            right_column,
            from_=1,
            to=50,
            orient='horizontal',
            variable=self.brush_size_var,
            command=self._on_brush_size_changed,
            length=150
        )
        brush_scale.pack(padx=5)
        
        self.brush_size_label = tk.Label(right_column, text=f'{self.brush_size}px', bg=Config.COLOR_BG)
        self.brush_size_label.pack()
        
        # 変形ツール
        tk.Label(right_column, text='変形', bg=Config.COLOR_BG, font=('Arial', 10, 'bold')).pack(pady=(15, 5))
        
        tk.Button(right_column, text='↔️ 左右反転', command=self._flip_horizontal, width=12).pack(fill='x', padx=5, pady=1)
        tk.Button(right_column, text='↕️ 上下反転', command=self._flip_vertical, width=12).pack(fill='x', padx=5, pady=1)
        tk.Button(right_column, text='🔄 90度回転', command=self._rotate_90, width=12).pack(fill='x', padx=5, pady=1)
        tk.Button(right_column, text='↔️ 左右中央', command=self._center_horizontal, width=12).pack(fill='x', padx=5, pady=1)
        tk.Button(right_column, text='↕️ 上下中央', command=self._center_vertical, width=12).pack(fill='x', padx=5, pady=1)
        tk.Button(right_column, text='🎯 上下左右', command=self._center_both, width=12).pack(fill='x', padx=5, pady=1)
        
        # ===== 右側: プレビューキャンバス =====
        preview_frame = tk.Frame(main_frame, bg=Config.COLOR_BG)
        preview_frame.pack(side='left', fill='both', expand=True)
        
        canvas_size = int(Config.CANVAS_SIZE * 1.2)

        canvas_container = tk.Frame(preview_frame, bg=Config.COLOR_BG)
        canvas_container.pack(fill='both', expand=True)

        self.preview_canvas = tk.Canvas(
            canvas_container,
            width=canvas_size,
            height=canvas_size,
            bg=Config.COLOR_CANVAS,
            highlightthickness=1,
            highlightbackground='gray'
        )
        h_scroll = ttk.Scrollbar(canvas_container, orient='horizontal', command=self._on_xscroll)
        v_scroll = ttk.Scrollbar(canvas_container, orient='vertical', command=self._on_yscroll)
        self.preview_canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
        self.preview_canvas.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        canvas_container.grid_rowconfigure(0, weight=1)
        canvas_container.grid_columnconfigure(0, weight=1)

        self.preview_canvas.bind('<Button-1>', self._on_mouse_down)
        self.preview_canvas.bind('<B1-Motion>', self._on_mouse_drag)
        self.preview_canvas.bind('<ButtonRelease-1>', self._on_mouse_up)
        self.preview_canvas.bind('<Motion>', self._on_mouse_move)
    
    def _set_tool(self, tool: str) -> None:
        """ツール切り替え (2025-10-12: テキストツール追加)"""
        # テキストツール選択時
        if tool == 'text':
            self._start_text_input_mode()
            return
        
        self.current_tool = tool
        
        for t, btn in self.tool_buttons.items():
            if t == tool:
                btn.config(relief='sunken', bg=Config.COLOR_ACTIVE)
            else:
                btn.config(relief='raised', bg=Config.COLOR_BG)
    
    def _on_brush_size_changed(self, value: str) -> None:
        """ブラシサイズ変更"""
        self.brush_size = int(float(value))
        self.brush_size_label.config(text=f'{self.brush_size}px')

# ===== [BLOCK5-END] =====







# ===== [BLOCK5.5-BEGIN] 編集エディタGUI - テキスト挿入 (2025-10-17: 透過エッジ正しく実装) =====
# ===== GlyphEditorクラスの続き =====
    
    def _start_text_input_mode(self) -> None:
        """テキスト入力モード開始"""
        if self.text_input_dialog:
            self.text_input_dialog.lift()
            return
        
        self.is_text_mode = True
        
        self.text_layer_resized_size = None
        self.text_layer_resized_pos = None
        
        dialog = tk.Toplevel(self)
        dialog.title('文字挿入')
        dialog.geometry('450x550')
        dialog.transient(self)
        
        self.text_input_dialog = dialog
        
        tab_container = ttk.Notebook(dialog)
        tab_container.pack(fill='both', expand=True, padx=10, pady=10)
        
        # ===== テキスト入力タブ =====
        text_tab = ttk.Frame(tab_container)
        tab_container.add(text_tab, text='テキスト入力')
        
        tk.Label(text_tab, text='挿入する文字を入力:', font=('Arial', 12, 'bold')).pack(pady=(10, 5))
        
        text_entry_frame = tk.Frame(text_tab)
        text_entry_frame.pack(fill='x', padx=20, pady=5)
        
        self.text_entry = tk.Entry(text_entry_frame, font=('Arial', 16), width=20)
        self.text_entry.pack(fill='x')
        self.text_entry.bind('<KeyRelease>', self._on_text_changed)
        self.text_entry.focus()
        
        # エッジ設定
        edge_frame = tk.Frame(text_tab)
        edge_frame.pack(fill='x', padx=20, pady=10)
        
        self.text_edge_var = tk.BooleanVar(value=False)
        edge_check = tk.Checkbutton(
            edge_frame, 
            text='白エッジを追加（透過領域を追加）', 
            variable=self.text_edge_var,
            command=self._on_text_changed,
            font=('Arial', 11)
        )
        edge_check.pack(anchor='w')
        
        tk.Label(edge_frame, text='エッジ幅:', font=('Arial', 10)).pack(anchor='w', pady=(10, 2))
        
        self.text_edge_width_var = tk.IntVar(value=3)
        # [MOD] エッジ幅のスライダを0〜100まで調整可能に拡大
        self.text_edge_scale = tk.Scale(
            edge_frame,
            from_=0,
            to=100,
            orient='horizontal',
            variable=self.text_edge_width_var,
            command=lambda v: self._on_text_changed() if self.text_edge_var.get() else None,
            length=200
        )
        self.text_edge_scale.pack(fill='x')

        # [ADD] エッジ幅の数値入力ボックス
        width_entry_frame = tk.Frame(edge_frame)
        width_entry_frame.pack(anchor='w', pady=(2, 2))
        tk.Label(width_entry_frame, text='幅入力:', font=('Arial', 9)).pack(side='left')
        self.text_edge_width_entry = tk.Entry(width_entry_frame, textvariable=self.text_edge_width_var, width=4)
        self.text_edge_width_entry.pack(side='left')
        # 値変更時にスライダと連動して更新
        def on_edge_width_entry_change(*args):
            try:
                val = int(self.text_edge_width_var.get())
            except Exception:
                return
            # 範囲を0-100に制限
            if val < 0:
                self.text_edge_width_var.set(0)
            elif val > 100:
                self.text_edge_width_var.set(100)
            if self.text_edge_var.get():
                self._on_text_changed()
        self.text_edge_width_var.trace_add('write', lambda *args: on_edge_width_entry_change())

        # [ADD] エッジ形状選択（角 or 丸）
        shape_frame = tk.Frame(edge_frame)
        shape_frame.pack(anchor='w', pady=(5, 0))
        tk.Label(shape_frame, text='エッジ形状:', font=('Arial', 9)).pack(side='left')
        sharp_rb = tk.Radiobutton(shape_frame, text='角', variable=self.edge_style_var, value='sharp', command=lambda: self._on_text_changed() if self.text_edge_var.get() else None, font=('Arial', 9))
        sharp_rb.pack(side='left', padx=(5, 0))
        round_rb = tk.Radiobutton(shape_frame, text='丸', variable=self.edge_style_var, value='round', command=lambda: self._on_text_changed() if self.text_edge_var.get() else None, font=('Arial', 9))
        round_rb.pack(side='left', padx=(5, 0))
        
        # ===== PNG読込タブ =====
        png_tab = ttk.Frame(tab_container)
        tab_container.add(png_tab, text='PNG読込')
        
        tk.Label(png_tab, text='PNG画像を読み込み:', font=('Arial', 12, 'bold')).pack(pady=(10, 5))
        
        png_btn_frame = tk.Frame(png_tab)
        png_btn_frame.pack(pady=10)
        
        tk.Button(
            png_btn_frame,
            text='📁 PNGファイルを選択',
            command=self._load_png_for_text,
            font=('Arial', 11),
            width=20
        ).pack()
        
        self.png_path_label = tk.Label(png_tab, text='ファイル未選択', font=('Arial', 9), fg='gray')
        self.png_path_label.pack(pady=5)
        
        tk.Label(png_tab, text='※ PNG画像は自動的にグレースケールに変換されます', 
                font=('Arial', 9), fg='gray').pack(pady=10)
        
        png_edge_frame = tk.Frame(png_tab)
        png_edge_frame.pack(fill='x', padx=20, pady=10)
        
        png_edge_check = tk.Checkbutton(
            png_edge_frame, 
            text='白エッジを追加（透過領域を追加）', 
            variable=self.text_edge_var,
            command=self._apply_edge_to_layer,
            font=('Arial', 11)
        )
        png_edge_check.pack(anchor='w')
        
        tk.Label(png_edge_frame, text='エッジ幅:', font=('Arial', 10)).pack(anchor='w', pady=(10, 2))
        
        # [MOD] PNG挿入時のエッジ幅スライダも0〜100まで選択可能にする
        self.png_edge_scale = tk.Scale(
            png_edge_frame,
            from_=0,
            to=100,
            orient='horizontal',
            variable=self.text_edge_width_var,
            command=lambda v: self._apply_edge_to_layer() if self.text_edge_var.get() else None,
            length=200
        )
        self.png_edge_scale.pack(fill='x')

        # [ADD] PNG用エッジ幅数値入力ボックス
        png_width_entry_frame = tk.Frame(png_edge_frame)
        png_width_entry_frame.pack(anchor='w', pady=(2, 2))
        tk.Label(png_width_entry_frame, text='幅入力:', font=('Arial', 9)).pack(side='left')
        self.png_edge_width_entry = tk.Entry(png_width_entry_frame, textvariable=self.text_edge_width_var, width=4)
        self.png_edge_width_entry.pack(side='left')
        # 値変更時にスライダと連動して更新
        def on_png_edge_width_entry_change(*args):
            try:
                val = int(self.text_edge_width_var.get())
            except Exception:
                return
            if val < 0:
                self.text_edge_width_var.set(0)
            elif val > 100:
                self.text_edge_width_var.set(100)
            if self.text_edge_var.get():
                self._apply_edge_to_layer()
        self.text_edge_width_var.trace_add('write', lambda *args: on_png_edge_width_entry_change())

        # [ADD] PNG用エッジ形状選択
        png_shape_frame = tk.Frame(png_edge_frame)
        png_shape_frame.pack(anchor='w', pady=(5, 0))
        tk.Label(png_shape_frame, text='エッジ形状:', font=('Arial', 9)).pack(side='left')
        sharp_png_rb = tk.Radiobutton(png_shape_frame, text='角', variable=self.edge_style_var, value='sharp', command=lambda: self._apply_edge_to_layer() if self.text_edge_var.get() else None, font=('Arial', 9))
        sharp_png_rb.pack(side='left', padx=(5, 0))
        round_png_rb = tk.Radiobutton(png_shape_frame, text='丸', variable=self.edge_style_var, value='round', command=lambda: self._apply_edge_to_layer() if self.text_edge_var.get() else None, font=('Arial', 9))
        round_png_rb.pack(side='left', padx=(5, 0))
        
        # ===== 共通ボタン =====
        btn_frame = tk.Frame(dialog)
        btn_frame.pack(side='bottom', pady=10)
        
        tk.Button(btn_frame, text='✅ 決定', command=self._commit_text_layer, 
                 width=10, font=('Arial', 11, 'bold'), bg='#4CAF50', fg='white').pack(side='left', padx=5)
        tk.Button(btn_frame, text='❌ キャンセル', command=self._cancel_text_input, 
                 width=10, font=('Arial', 11)).pack(side='left', padx=5)
        
        dialog.protocol('WM_DELETE_WINDOW', self._cancel_text_input)
    
    def _load_png_for_text(self) -> None:
        """PNG画像を読み込み"""
        path = filedialog.askopenfilename(
            title='PNG画像を選択',
            filetypes=[
                ('PNG Image', '*.png'),
                ('All Files', '*.*')
            ]
        )
        
        if not path:
            return
        
        try:
            img = Image.open(path)
            
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background.convert('L')
            elif img.mode != 'L':
                img = img.convert('L')
            
            max_size = Config.CANVAS_SIZE // 2
            if img.width > max_size or img.height > max_size:
                ratio = min(max_size / img.width, max_size / img.height)
                new_width = int(img.width * ratio)
                new_height = int(img.height * ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)
            
            self.text_layer = img
            self.text_layer_original = img.copy()
            
            x_pos = (Config.CANVAS_SIZE - img.width) // 2
            y_pos = (Config.CANVAS_SIZE - img.height) // 2
            self.text_layer_pos = (x_pos, y_pos)
            
            if self.text_edge_var.get():
                self._apply_edge_to_layer()
            
            self._update_preview()
            
            self.png_path_label.config(text=os.path.basename(path))
            
        except Exception as e:
            messagebox.showerror('エラー', f'PNG読み込みエラー:\n{e}')
    
    def _apply_edge_to_layer(self) -> None:
        """
        レイヤーにエッジを適用する。

        現在のテキストレイヤー (self.text_layer_original) を基にエッジ領域を算出し、
        エッジが有効な場合はその領域を preview で白色として表示し、決定時には透過となるよう
        自前のエッジマスク (self.text_edge_mask) と描画用レイヤー (self.text_layer) を生成する。
        エッジ幅が 0 またはエッジ表示が無効な場合はマスクを生成せず元画像を使用する。
        """
        # テキストレイヤーが存在しない場合は何もしない
        if not self.text_layer_original:
            return

        # エッジ機能が無効の場合はマスクをクリアして元画像をコピー
        if not self.text_edge_var.get():
            # エッジが無効な場合はマスクをクリア
            self.text_edge_mask = None
            self.text_edge_mask_commit = None
            self.text_layer = self.text_layer_original.copy()
            self._update_preview()
            return

        edge_width = self.text_edge_width_var.get()

        # エッジ幅が0の場合も同様にマスク無しで元画像をそのまま使用
        if edge_width == 0:
            # エッジ幅0の場合もマスクを生成しない
            self.text_edge_mask = None
            self.text_edge_mask_commit = None
            self.text_layer = self.text_layer_original.copy()
            self._update_preview()
            return

        from PIL import ImageFilter

        # テキストレイヤーのコピー（グレースケール）
        base = self.text_layer_original.copy()
        width, height = base.size

        # 元の黒領域マスクを作成：文字部分は0、背景は255
        # 250未満のピクセルを文字とみなす（アンチエイリアス部分も含む）
        mask_original = base.point(lambda p: 0 if p < 250 else 255)

        # Edge style: 'sharp' or 'round'. For 'round', smooth the mask before dilation to round corners
        edge_style = getattr(self, 'edge_style_var', None).get() if hasattr(self, 'edge_style_var') else 'sharp'
        mask_to_dilate = mask_original
        if edge_style == 'round':
            # Apply a slight Gaussian blur to soften corners before dilation. The blur radius of 1
            # provides a smoother contour. Threshold back to binary after blurring.
            blurred = mask_original.filter(ImageFilter.GaussianBlur(1))
            mask_to_dilate = blurred.point(lambda p: 0 if p < 128 else 255)

        # 膨張処理：MinFilterを大きなカーネルで1回適用することで高速化する。
        # MinFilterは黒(0)を外側へ広げるので、サイズは2*edge_width+1とする。
        if edge_width > 0:
            kernel_size = edge_width * 2 + 1
            # pillow の MinFilter はカーネルサイズが奇数である必要がある
            # kernel_size が偶数の場合は次の奇数に調整
            if kernel_size % 2 == 0:
                kernel_size += 1
            dilated = mask_to_dilate.filter(ImageFilter.MinFilter(kernel_size))
        else:
            dilated = mask_to_dilate.copy()

        # 結果となるレイヤーとエッジマスクを初期化
        result = Image.new('L', base.size, 255)
        edge_mask_commit = Image.new('L', base.size, 0)  # コミット用
        edge_mask_preview = Image.new('L', base.size, 0)  # プレビュー用 (sharp=original, round=加工)

        orig_pixels = base.load()
        mask_pixels = mask_original.load()
        dil_pixels = dilated.load()
        res_pixels = result.load()
        edge_pixels_commit = edge_mask_commit.load()
        edge_pixels_preview = edge_mask_preview.load()

        # ピクセル単位で分類
        for y in range(height):
            for x in range(width):
                # 元の文字部分はそのまま（濃度を保持）
                if mask_pixels[x, y] == 0:
                    res_pixels[x, y] = orig_pixels[x, y]
                # 膨張した領域かつ元の文字ではない → エッジ領域
                elif dil_pixels[x, y] == 0 and mask_pixels[x, y] != 0:
                    # レイヤー上では透過（255）とする
                    res_pixels[x, y] = 255
                    # コミット用マスク：エッジ領域は255
                    edge_pixels_commit[x, y] = 255
                    # プレビュー用マスク: 初期状態はsharpと同様
                    edge_pixels_preview[x, y] = 255
                # それ以外は背景のまま

        # エッジ形状が丸の場合、プレビュー用マスクをぼかして角を丸める
        if edge_style == 'round' and edge_width > 0:
            try:
                # プレビュー用マスクをガウシアンブラーで滑らかにし、閾値をかけてバイナリ化
                blur_radius = max(1, int(edge_width / 2))
                blurred = edge_mask_preview.filter(ImageFilter.GaussianBlur(blur_radius))
                # 一旦二値化（少しでも白くなった部分をエッジとする）
                thresholded = blurred.point(lambda p: 255 if p > 0 else 0)
                # 内部侵食を防ぐため、元の文字部分(mask_pixels==0)ではマスクを0に設定する
                preview_pixels = thresholded.load()
                for yy in range(height):
                    for xx in range(width):
                        # mask_pixels[x,y]==0 は元の文字領域
                        if mask_pixels[xx, yy] == 0:
                            preview_pixels[xx, yy] = 0
                edge_mask_preview = thresholded
            except Exception:
                pass

        # テキストレイヤーとエッジマスクを保存
        self.text_layer = result
        # コミット用エッジマスク
        self.text_edge_mask_commit = edge_mask_commit
        # プレビュー用エッジマスク
        self.text_edge_mask = edge_mask_preview
        # プレビュー更新
        self._update_preview()
    
    def _on_text_changed(self, event=None) -> None:
        """テキスト入力変更時"""
        text = self.text_entry.get().strip()
        
        if not text:
            if not hasattr(self, 'text_layer_original') or self.text_layer_original is None:
                self.text_layer = None
                self.text_layer_resized_size = None
                self.text_layer_resized_pos = None
                self._update_preview()
            else:
                self._apply_edge_to_layer()
                self._update_preview()
            return
        
        if not self.project.font_path:
            messagebox.showwarning('警告', 'フォントが読み込まれていません')
            return
        
        try:
            target_size = self.text_layer_resized_size
            target_pos = self.text_layer_resized_pos
            
            font = ImageFont.truetype(self.project.font_path, size=Config.FONT_RENDER_SIZE)
            
            char_img = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 255)
            draw = ImageDraw.Draw(char_img)
            
            bbox = draw.textbbox((0, 0), text, font=font)
            w = bbox[2] - bbox[0]
            h = bbox[3] - bbox[1]
            
            x = (Config.CANVAS_SIZE - w) / 2 - bbox[0]
            y = (Config.CANVAS_SIZE - h) / 2 - bbox[1]
            
            draw.text((x, y), text, fill=0, font=font)
            
            bbox = char_img.getbbox()
            if bbox:
                trimmed = char_img.crop(bbox)
                
                if target_size and target_pos:
                    target_w, target_h = target_size
                    self.text_layer_original = trimmed.resize((target_w, target_h), Image.LANCZOS)
                    self.text_layer_pos = target_pos
                else:
                    self.text_layer_original = trimmed
                    x_pos = (Config.CANVAS_SIZE - trimmed.width) // 2
                    y_pos = (Config.CANVAS_SIZE - trimmed.height) // 2
                    self.text_layer_pos = (x_pos, y_pos)
                
                if self.text_edge_var.get():
                    self._apply_edge_to_layer()
                else:
                    # エッジ無しの場合は元画像をそのまま使用し、エッジマスクをクリア
                    self.text_layer = self.text_layer_original.copy()
                    self.text_edge_mask = None
            else:
                self.text_layer = None
                self.text_layer_original = None
                self.text_layer_resized_size = None
                self.text_layer_resized_pos = None
            
            self._update_preview()
            
        except Exception as e:
            print(f'テキストレンダリングエラー: {e}')
            import traceback
            traceback.print_exc()
    
    def _commit_text_layer(self) -> None:
        """テキストレイヤーをベースに統合 (2025-10-17: 透過を正しく処理)"""  # [FIX]
        if not self.text_layer:
            messagebox.showwarning('警告', 'テキストが入力されていません')
            return
        
        # [FIX] 2025-10-17: 透過(255)はスキップ、黒はmin合成
        # レイヤーが存在しない場合は警告を出して終了
        if not self.text_layer:
            messagebox.showwarning('警告', 'テキストが入力されていません')
            return

        x_pos, y_pos = self.text_layer_pos
        # ラスタライズしたテキストレイヤーをキャンバスサイズの画像に展開
        layer_img = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 255)
        layer_img.paste(self.text_layer, (x_pos, y_pos))

        # 合成処理：255より小さい領域のみ更新（255は透過および背景として扱う）
        mask = layer_img.point(lambda p: 255 if p < 255 else 0)
        # darker関数で元画像とレイヤーの暗い方を採用
        darker = ImageChops.darker(self.edit_bitmap, layer_img)
        # マスクに従ってペースト
        new_bitmap = self.edit_bitmap.copy()
        new_bitmap.paste(darker, mask=mask)

        # [ADD] 2025-10-22: エッジ領域を透過処理
        # テキストエッジマスクが存在する場合は、エッジ領域と重なっているベース画像を透過
        # （白＝255）に置き換える。これにより、決定後に白エッジが透明に変換される。
        # コミット用エッジマスクが存在する場合は、エッジ領域を白に置き換える
        if getattr(self, 'text_edge_mask_commit', None):
            try:
                edge_mask_full = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 0)
                x_pos, y_pos = self.text_layer_pos
                edge_mask_full.paste(self.text_edge_mask_commit, (x_pos, y_pos))
                white_layer = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 255)
                new_bitmap = Image.composite(white_layer, new_bitmap, edge_mask_full)
            except Exception:
                pass
        # 更新
        self.edit_bitmap = new_bitmap

        # テキスト関連データのリセット
        self.text_layer = None
        self.text_layer_original = None
        self.text_layer_resized_size = None
        self.text_layer_resized_pos = None
        self.is_text_mode = False
        # エッジマスク類をリセット
        self.text_edge_mask = None
        self.text_edge_mask_commit = None
        if self.text_input_dialog:
            self.text_input_dialog.destroy()
            self.text_input_dialog = None

        self._save_to_undo()
        self._update_preview()
        messagebox.showinfo('完了', 'テキストを統合しました')
    
    def _cancel_text_input(self) -> None:
        """テキスト入力をキャンセル"""
        self.text_layer = None
        self.text_layer_original = None
        self.text_layer_resized_size = None
        self.text_layer_resized_pos = None
        self.is_text_mode = False
        
        if self.text_input_dialog:
            self.text_input_dialog.destroy()
            self.text_input_dialog = None
        
        self._update_preview()

# ===== [BLOCK5.5-END] =====









# ===== [BLOCK5.6-BEGIN] 編集エディタGUI - 描画メソッドとプレビュー更新 (2025-10-13: 新規作成) =====
# ===== GlyphEditorクラスの続き =====
    
    # ===== 描画メソッド (2025-10-13: 基本描画処理) =====
    
    def _draw_point(self, x: int, y: int) -> None:
        """点を描画"""
        if not (0 <= x < Config.CANVAS_SIZE and 0 <= y < Config.CANVAS_SIZE):
            return
        
        pixels = self.edit_bitmap.load()
        color = 0 if self.current_tool == 'pen' else 255  # ペン=黒、消しゴム=白
        
        # ブラシサイズに応じて円形で描画
        radius = self.brush_size // 2
        
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                if dx * dx + dy * dy <= radius * radius:  # 円形判定
                    px = x + dx
                    py = y + dy
                    
                    if 0 <= px < Config.CANVAS_SIZE and 0 <= py < Config.CANVAS_SIZE:
                        pixels[px, py] = color
    
    def _draw_line(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """線を描画（ブレゼンハムアルゴリズム）"""
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy
        
        x, y = x0, y0
        
        while True:
            self._draw_point(x, y)
            
            if x == x1 and y == y1:
                break
            
            e2 = 2 * err
            
            if e2 > -dy:
                err -= dy
                x += sx
            
            if e2 < dx:
                err += dx
                y += sy
    
    def _flood_fill(self, x: int, y: int) -> None:
        """塗りつぶし（スタック使用）"""
        if not (0 <= x < Config.CANVAS_SIZE and 0 <= y < Config.CANVAS_SIZE):
            return
        
        pixels = self.edit_bitmap.load()
        target_color = pixels[x, y]
        fill_color = 0 if self.current_tool == 'pen' else 255
        
        if target_color == fill_color:
            return  # 既に同じ色
        
        # スタック使用（再帰ではなく）
        stack = [(x, y)]
        visited = set()
        
        while stack:
            cx, cy = stack.pop()
            
            if (cx, cy) in visited:
                continue
            
            if not (0 <= cx < Config.CANVAS_SIZE and 0 <= cy < Config.CANVAS_SIZE):
                continue
            
            if pixels[cx, cy] != target_color:
                continue
            
            pixels[cx, cy] = fill_color
            visited.add((cx, cy))
            
            # 4方向に拡張
            stack.extend([(cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)])
        
        self._save_to_undo()
        self._update_preview()
    
    # ===== プレビュー更新 (2025-10-13: 通常版と高速版) =====
    
    def _update_preview(self) -> None:
        """プレビュー更新（通常版：グリッド・ハンドル含む）"""
        # ズーム適用
        new_width = int(Config.CANVAS_SIZE * self.zoom_level)
        new_height = int(Config.CANVAS_SIZE * self.zoom_level)

        # 背景パターンが未生成の場合は生成
        if self._bg_full is None or self._bg_full.size != (Config.CANVAS_SIZE, Config.CANVAS_SIZE):
            self._create_full_bg()

        # ベースレイヤーとテキストレイヤーを合成（ループを使わず高速化）
        if self.text_layer and not (self.is_moving or self.is_resizing):
            # テキストレイヤーが存在する場合はレイヤーをベース画像に貼り付けた画像を作成し、
            # ImageChops.darkerにより暗い方（0に近い方）を採用することで合成する
            layer_img = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 255)
            x_pos, y_pos = self.text_layer_pos
            layer_img.paste(self.text_layer, (x_pos, y_pos))
            composite = ImageChops.darker(self.edit_bitmap, layer_img)
        else:
            composite = self.edit_bitmap

        # [ADD] 2025-10-22: テキストエッジのプレビュー表示
        # エッジマスクが存在する場合は、プレビュー上で白系の色を合成して視認性を高める
        if self.text_layer and self.text_edge_mask and not (self.is_moving or self.is_resizing):
            try:
                # エッジマスクをキャンバス全体に展開
                edge_full = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 0)
                x_pos, y_pos = self.text_layer_pos
                edge_full.paste(self.text_edge_mask, (x_pos, y_pos))
                # エッジ用の明るいレイヤー（白に近いグレー）
                # エッジを際立たせるため、背景より明るい値(254)で表示
                edge_color = 254
                edge_layer = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), edge_color)
                # マスクの非ゼロ領域にedge_layerを適用 (maskの値255→first image)
                composite = Image.composite(edge_layer, composite, edge_full)
            except Exception:
                # 念のためエラー時はそのまま表示
                pass

        # 背景パターンと合成
        # 255の場所（完全な白）は透過とみなし、背景パターンが表示される。
        # それ以外は黒や薄い灰色をそのまま前景として描画する。
        # まずマスク画像を生成: 255->0, その他->255
        mask = composite.point(lambda p: 0 if p == 255 else 255)
        merged = Image.composite(composite, self._bg_full, mask)

        # ズームリサイズ
        zoomed = merged.resize((new_width, new_height), Image.NEAREST if self.is_moving or self.is_resizing else Image.LANCZOS)

        # キャンバスに描画
        self.photo = ImageTk.PhotoImage(zoomed)
        self.preview_canvas.delete('all')
        self.preview_canvas.create_image(0, 0, anchor='nw', image=self.photo, tags='base')

        # グリッド線描画
        self._draw_grid()

        # 移動/リサイズ中の選択領域プレビュー
        if self.current_tool == 'move' and self.is_moving and self.move_current_pos:
            self._draw_moving_preview()
        if self.current_tool == 'resize' and self.is_resizing and self.resize_preview_rect:
            self._draw_resizing_preview()

        # 選択矩形描画
        if self.current_tool == 'select' and self.selection_start and self.selection_end:
            self._draw_selection_rect()

        # リサイズハンドル描画
        if self.current_tool == 'resize' and self.selection_start and self.selection_end:
            self._draw_resize_handles()

        # 図形プレビュー
        if self.shape_start and self.shape_end:
            self._draw_shape_preview()

        # テキストレイヤーの操作中プレビュー（移動/リサイズ中）
        if self.is_text_mode and self.text_layer:
            self._draw_text_layer_preview()
            if self.current_tool == 'resize':
                self._draw_text_layer_handles()

        # ガイドライン描画
        for guide_type, pos in self.guidelines:
            if guide_type == 'h':
                y_canvas = pos * self.zoom_level
                self.preview_canvas.create_line(0, y_canvas, new_width, y_canvas,
                                               fill='#FF00FF', dash=(5, 5), tags='guide')
            elif guide_type == 'v':
                x_canvas = pos * self.zoom_level
                self.preview_canvas.create_line(x_canvas, 0, x_canvas, new_height,
                                               fill='#FF00FF', dash=(5, 5), tags='guide')

        # ナビゲーション更新
        self._update_nav()

        # スクロール領域更新
        self.preview_canvas.configure(scrollregion=(0, 0, new_width, new_height))
    
    def _update_preview_fast(self) -> None:
        """高速プレビュー更新（ドラッグ中専用：グリッド・ハンドル省略）"""  # [ADD] 2025-10-13
        # ズーム適用
        new_width = int(Config.CANVAS_SIZE * self.zoom_level)
        new_height = int(Config.CANVAS_SIZE * self.zoom_level)

        # 背景パターン生成を確認
        if self._bg_full is None or self._bg_full.size != (Config.CANVAS_SIZE, Config.CANVAS_SIZE):
            self._create_full_bg()

        # グリッド線を削除（高速更新では描画しないため）
        self.preview_canvas.delete('grid')

        # 合成処理（高速バージョン）
        if self.text_layer:
            # レイヤーをベースに貼り付けて暗い方を採用
            layer_img = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 255)
            x_pos, y_pos = self.text_layer_pos
            layer_img.paste(self.text_layer, (x_pos, y_pos))
            composite = ImageChops.darker(self.edit_bitmap, layer_img)
        else:
            composite = self.edit_bitmap

        # [ADD] 2025-10-22: 高速プレビューでもエッジを表示
        if self.text_layer and self.text_edge_mask:
            try:
                edge_full = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 0)
                x_pos, y_pos = self.text_layer_pos
                edge_full.paste(self.text_edge_mask, (x_pos, y_pos))
                # エッジを際立たせるため、背景より明るい値(254)で表示
                edge_color = 254
                edge_layer = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), edge_color)
                composite = Image.composite(edge_layer, composite, edge_full)
            except Exception:
                pass

        # 背景と合成
        mask = composite.point(lambda p: 0 if p == 255 else 255)
        merged = Image.composite(composite, self._bg_full, mask)

        zoomed = merged.resize((new_width, new_height), Image.NEAREST)

        self.photo = ImageTk.PhotoImage(zoomed)
        # ベース画像のみ更新（タグ指定で高速化）
        self.preview_canvas.delete('base')
        self.preview_canvas.create_image(0, 0, anchor='nw', image=self.photo, tags='base')
        
        # テキストレイヤーの枠のみ描画（簡易表示）
        if self.is_text_mode and self.text_layer:
            self.preview_canvas.delete('text_layer_rect')
            
            x_pos, y_pos = self.text_layer_pos
            x_end = x_pos + self.text_layer.width
            y_end = y_pos + self.text_layer.height
            
            cx1 = x_pos * self.zoom_level
            cy1 = y_pos * self.zoom_level
            cx2 = x_end * self.zoom_level
            cy2 = y_end * self.zoom_level
            
            self.preview_canvas.create_rectangle(cx1, cy1, cx2, cy2, 
                                                outline='#00FF00', width=2, 
                                                dash=(5, 5), tags='text_layer_rect')
        
        # ナビゲーション更新（軽量版）
        self._update_nav()
    
    # ===== 描画補助メソッド (2025-10-13: プレビュー用) =====
    
    def _draw_grid(self) -> None:
        """グリッド線を描画"""
        # グリッド表示がオフの場合は描画しない
        if not self.grid_visible_var.get():
            return

        new_width = int(Config.CANVAS_SIZE * self.zoom_level)
        new_height = int(Config.CANVAS_SIZE * self.zoom_level)

        # 縦線
        for x in range(0, Config.CANVAS_SIZE, Config.GRID_SPACING):
            x_canvas = x * self.zoom_level
            color = Config.GRID_CENTER_COLOR if x == Config.CANVAS_SIZE // 2 else Config.GRID_COLOR
            self.preview_canvas.create_line(x_canvas, 0, x_canvas, new_height,
                                           fill=color, tags='grid')

        # 横線
        for y in range(0, Config.CANVAS_SIZE, Config.GRID_SPACING):
            y_canvas = y * self.zoom_level
            color = Config.GRID_CENTER_COLOR if y == Config.CANVAS_SIZE // 2 else Config.GRID_COLOR
            self.preview_canvas.create_line(0, y_canvas, new_width, y_canvas,
                                           fill=color, tags='grid')
    
    def _draw_selection_rect(self) -> None:
        """選択矩形を描画"""
        if not (self.selection_start and self.selection_end):
            return
        
        x1, y1 = self.selection_start
        x2, y2 = self.selection_end
        
        cx1 = x1 * self.zoom_level
        cy1 = y1 * self.zoom_level
        cx2 = x2 * self.zoom_level
        cy2 = y2 * self.zoom_level
        
        self.preview_canvas.create_rectangle(cx1, cy1, cx2, cy2, 
                                            outline='#0000FF', width=2, 
                                            dash=(5, 5), tags='selection')
    
    def _draw_resize_handles(self) -> None:
        """リサイズハンドルを描画"""
        if not (self.selection_start and self.selection_end):
            return
        
        self._normalize_selection()
        x1, y1 = self.selection_start
        x2, y2 = self.selection_end
        
        cx1 = x1 * self.zoom_level
        cy1 = y1 * self.zoom_level
        cx2 = x2 * self.zoom_level
        cy2 = y2 * self.zoom_level
        
        cx_mid = (cx1 + cx2) / 2
        cy_mid = (cy1 + cy2) / 2
        
        handle_size = 6
        
        handles = [
            (cx1, cy1), (cx_mid, cy1), (cx2, cy1),
            (cx1, cy_mid), (cx2, cy_mid),
            (cx1, cy2), (cx_mid, cy2), (cx2, cy2)
        ]
        
        # 既存のハンドルを削除
        for hid in self.resize_handle_ids:
            self.preview_canvas.delete(hid)
        self.resize_handle_ids.clear()
        
        # 新しいハンドルを描画
        for hx, hy in handles:
            hid = self.preview_canvas.create_rectangle(
                hx - handle_size, hy - handle_size,
                hx + handle_size, hy + handle_size,
                fill='white', outline='blue', width=2, tags='handle'
            )
            self.resize_handle_ids.append(hid)
    
    def _draw_text_layer_handles(self) -> None:
        """テキストレイヤーのハンドルを描画"""
        if not self.text_layer:
            return
        
        x_pos, y_pos = self.text_layer_pos
        x_end = x_pos + self.text_layer.width
        y_end = y_pos + self.text_layer.height
        
        cx1 = x_pos * self.zoom_level
        cy1 = y_pos * self.zoom_level
        cx2 = x_end * self.zoom_level
        cy2 = y_end * self.zoom_level
        
        cx_mid = (cx1 + cx2) / 2
        cy_mid = (cy1 + cy2) / 2
        
        handle_size = 6
        
        handles = [
            (cx1, cy1), (cx_mid, cy1), (cx2, cy1),
            (cx1, cy_mid), (cx2, cy_mid),
            (cx1, cy2), (cx_mid, cy2), (cx2, cy2)
        ]
        
        # テキストレイヤーハンドルを描画
        for hx, hy in handles:
            self.preview_canvas.create_rectangle(
                hx - handle_size, hy - handle_size,
                hx + handle_size, hy + handle_size,
                fill='lime', outline='green', width=2, tags='text_handle'
            )
    
    def _draw_moving_preview(self) -> None:
        """移動中のプレビュー描画"""
        if not (self.selected_image and self.move_current_pos):
            return
        
        x, y = self.move_current_pos
        w = self.selected_image.width
        h = self.selected_image.height
        
        cx1 = x * self.zoom_level
        cy1 = y * self.zoom_level
        cx2 = (x + w) * self.zoom_level
        cy2 = (y + h) * self.zoom_level
        
        # 選択範囲の内容をプレビューに描画し、枠線を表示
        # 既存の移動画像を削除
        self.preview_canvas.delete('moving_img')
        # 描画内容を貼り付け
        try:
            # 拡大縮小された選択内容を作成
            preview_sel = self.selected_image.resize((int(w * self.zoom_level), int(h * self.zoom_level)), Image.NEAREST)
            self._move_photo = ImageTk.PhotoImage(preview_sel)
            self.preview_canvas.create_image(cx1, cy1, anchor='nw', image=self._move_photo, tags='moving_img')
        except Exception:
            pass
        # 枠線を描画
        self.preview_canvas.create_rectangle(cx1, cy1, cx2, cy2,
                                            outline='#00FF00', width=2,
                                            dash=(3, 3), tags='moving')
    
    def _draw_resizing_preview(self) -> None:
        """リサイズ中のプレビュー描画"""
        if not self.resize_preview_rect:
            return
        
        x1, y1, x2, y2 = self.resize_preview_rect
        
        cx1 = x1 * self.zoom_level
        cy1 = y1 * self.zoom_level
        cx2 = x2 * self.zoom_level
        cy2 = y2 * self.zoom_level
        
        self.preview_canvas.create_rectangle(cx1, cy1, cx2, cy2, 
                                            outline='#FF00FF', width=2, 
                                            dash=(3, 3), tags='resizing')
    
    def _draw_shape_preview(self) -> None:
        """図形プレビュー描画"""
        if not (self.shape_start and self.shape_end):
            return
        
        x1, y1 = self.shape_start
        x2, y2 = self.shape_end
        
        cx1 = x1 * self.zoom_level
        cy1 = y1 * self.zoom_level
        cx2 = x2 * self.zoom_level
        cy2 = y2 * self.zoom_level
        
        if self.current_tool == 'line':
            self.preview_canvas.create_line(cx1, cy1, cx2, cy2, 
                                           fill='red', width=2, tags='shape_preview')
        elif self.current_tool == 'rect':
            self.preview_canvas.create_rectangle(cx1, cy1, cx2, cy2, 
                                                outline='red', width=2, tags='shape_preview')
        elif self.current_tool == 'ellipse':
            self.preview_canvas.create_oval(cx1, cy1, cx2, cy2, 
                                           outline='red', width=2, tags='shape_preview')
    
    def _draw_text_layer_preview(self) -> None:
        """テキストレイヤーのプレビュー描画"""
        if not self.text_layer:
            return
        
        x_pos, y_pos = self.text_layer_pos
        x_end = x_pos + self.text_layer.width
        y_end = y_pos + self.text_layer.height
        
        cx1 = x_pos * self.zoom_level
        cy1 = y_pos * self.zoom_level
        cx2 = x_end * self.zoom_level
        cy2 = y_end * self.zoom_level
        
        # 緑色の枠で表示
        self.preview_canvas.create_rectangle(cx1, cy1, cx2, cy2, 
                                            outline='#00FF00', width=2, 
                                            dash=(5, 5), tags='text_layer')
    
    # ===== ナビゲーション更新 (2025-10-13) =====
    
    def _update_nav(self) -> None:
        """ナビゲーションウィンドウ更新"""
        # 現在の画像を縮小してナビゲーションに表示
        nav_img = self.edit_bitmap.resize((Config.NAV_SIZE, Config.NAV_SIZE), Image.NEAREST)
        self._nav_photo = ImageTk.PhotoImage(nav_img)
        
        self.nav_canvas.delete('all')
        self.nav_canvas.create_image(0, 0, anchor='nw', image=self._nav_photo)
        
        # 現在の表示範囲を赤枠で表示
        # visible_w/h: 画像上で表示されている幅・高さ
        visible_w = self.preview_canvas.winfo_width() / self.zoom_level
        visible_h = self.preview_canvas.winfo_height() / self.zoom_level
        ratio = Config.NAV_SIZE / Config.CANVAS_SIZE
        nav_w = visible_w * ratio
        nav_h = visible_h * ratio
        # 現在のオフセット（スクロール位置）を取得
        # canvasx/canvasyはズーム後の座標を返すのでズームレベルで割る
        try:
            x0_canvas = self.preview_canvas.canvasx(0)
            y0_canvas = self.preview_canvas.canvasy(0)
        except Exception:
            x0_canvas = 0
            y0_canvas = 0
        img_x0 = x0_canvas / self.zoom_level
        img_y0 = y0_canvas / self.zoom_level
        nav_x = img_x0 * ratio
        nav_y = img_y0 * ratio
        self.nav_canvas.create_rectangle(nav_x, nav_y, nav_x + nav_w, nav_y + nav_h,
                                        outline='red', width=2)

    def _on_nav_click(self, event: tk.Event) -> None:
        """
        ナビゲーションウィンドウのクリック位置に応じて、プレビューキャンバスの表示領域をスクロールする。
        クリックした位置がプレビューの中心になるように移動する。
        """
        # クリック座標を画像の座標系に変換
        ratio = Config.NAV_SIZE / Config.CANVAS_SIZE
        img_x = event.x / ratio
        img_y = event.y / ratio
        # 現在の表示領域のサイズ（画像座標）
        visible_w = self.preview_canvas.winfo_width() / self.zoom_level
        visible_h = self.preview_canvas.winfo_height() / self.zoom_level
        # 左上座標を計算（クリック位置を中心に）
        target_x = img_x - visible_w / 2
        target_y = img_y - visible_h / 2
        max_x = Config.CANVAS_SIZE - visible_w
        max_y = Config.CANVAS_SIZE - visible_h
        target_x = max(0, min(target_x, max_x))
        target_y = max(0, min(target_y, max_y))
        # キャンバス座標に変換
        target_canvas_x = target_x * self.zoom_level
        target_canvas_y = target_y * self.zoom_level
        total_width = Config.CANVAS_SIZE * self.zoom_level
        total_height = Config.CANVAS_SIZE * self.zoom_level
        denom_x = max(1, total_width - self.preview_canvas.winfo_width())
        denom_y = max(1, total_height - self.preview_canvas.winfo_height())
        # スクロール移動
        self.preview_canvas.xview_moveto(target_canvas_x / denom_x)
        self.preview_canvas.yview_moveto(target_canvas_y / denom_y)
        self._update_preview()
    
    # ===== スクロール処理 (2025-10-13) =====
    
    def _on_xscroll(self, *args) -> None:
        """横スクロール"""
        self.preview_canvas.xview(*args)
    
    def _on_yscroll(self, *args) -> None:
        """縦スクロール"""
        self.preview_canvas.yview(*args)

# ===== [BLOCK5.6-END] =====








# ===== [BLOCK5.7-BEGIN] 編集エディタGUI - 選択・変形・操作メソッド (2025-10-17: マウスイベント追加) =====
# ===== GlyphEditorクラスの続き =====
    
    # ===== 選択領域メソッド (2025-10-13: 選択・移動・削除処理) =====
    
    def _finalize_selection(self) -> None:
        """選択範囲を確定"""
        if not (self.selection_start and self.selection_end):
            return
        
        self._normalize_selection()
        x1, y1 = self.selection_start
        x2, y2 = self.selection_end
        
        # 選択領域が小さすぎる場合は無視
        if abs(x2 - x1) < 2 or abs(y2 - y1) < 2:
            self.selection_start = None
            self.selection_end = None
            self.selected_image = None
            return
        
        # 選択領域を切り取り
        try:
            self.selected_image = self.edit_bitmap.crop((x1, y1, x2, y2)).copy()
        except Exception as e:
            print(f'選択エラー: {e}')
            self.selection_start = None
            self.selection_end = None
            self.selected_image = None
    
    def _apply_translation(self) -> None:
        """移動を確定"""
        if not (self.selected_image and self.move_current_pos and self.selection_start):
            return
        
        # 元の領域を白で塗りつぶし
        draw = ImageDraw.Draw(self.edit_bitmap)
        x1, y1 = self.selection_start
        x2, y2 = self.selection_end
        # 元の領域を白で塗りつぶす（排他的範囲）
        draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
        # 新しい位置に貼り付け（透過部分は無視）
        dest_x, dest_y = self.move_current_pos
        mask = self.selected_image.point(lambda p: 255 if p < 255 else 0)
        self.edit_bitmap.paste(self.selected_image, (dest_x, dest_y), mask)
        # 選択状態を更新
        w = self.selected_image.width
        h = self.selected_image.height
        self.selection_start = (dest_x, dest_y)
        self.selection_end = (dest_x + w, dest_y + h)
        self._save_to_undo()
        self._update_preview()
    
    def _commit_shape(self, start: Tuple[int, int], end: Tuple[int, int]) -> None:
        """図形を確定して描画"""
        x1, y1 = start
        x2, y2 = end
        
        draw = ImageDraw.Draw(self.edit_bitmap)
        
        if self.current_tool == 'line':
            # 直線描画
            self._draw_line(x1, y1, x2, y2)
        elif self.current_tool == 'rect':
            # 矩形描画
            draw.rectangle((x1, y1, x2, y2), outline=0, width=self.brush_size)
        elif self.current_tool == 'ellipse':
            # 楕円描画
            draw.ellipse((x1, y1, x2, y2), outline=0, width=self.brush_size)
        
        self._save_to_undo()
        self._update_preview()
    
    def _nudge(self, dx: int, dy: int) -> None:
        """矢印キーで1px移動"""
        if self.is_text_mode and self.text_layer:
            # テキストレイヤーの移動
            x_pos, y_pos = self.text_layer_pos
            new_x = max(0, min(x_pos + dx, Config.CANVAS_SIZE - self.text_layer.width))
            new_y = max(0, min(y_pos + dy, Config.CANVAS_SIZE - self.text_layer.height))
            self.text_layer_pos = (new_x, new_y)
            self._update_preview()
        elif self.selected_image and self.selection_start and self.selection_end:
            # 選択領域の移動
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            new_x1 = max(0, min(x1 + dx, Config.CANVAS_SIZE - (x2 - x1)))
            new_y1 = max(0, min(y1 + dy, Config.CANVAS_SIZE - (y2 - y1)))
            # 元の領域を白で塗りつぶし
            draw = ImageDraw.Draw(self.edit_bitmap)
            # 元の領域を白で塗りつぶす（Pillowのrectangleは終点を含むため-1する）
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
            # マスクを作成し、選択範囲の黒／灰色ピクセルのみを貼り付け
            # 250未満は描画すべき領域、その他は透過扱い
            # 非透過ピクセルをすべて移動対象とする
            mask = self.selected_image.point(lambda p: 255 if p < 255 else 0)
            self.edit_bitmap.paste(self.selected_image, (new_x1, new_y1), mask)
            # 選択状態を更新
            w = x2 - x1
            h = y2 - y1
            self.selection_start = (new_x1, new_y1)
            self.selection_end = (new_x1 + w, new_y1 + h)
            self._save_to_undo()
            self._update_preview()
    
    def _copy_selection(self) -> None:
        """選択領域をコピー"""
        if self.selected_image:
            self.project.clipboard = self.selected_image.copy()
            messagebox.showinfo('コピー', '選択領域をコピーしました')
    
    def _cut_selection(self) -> None:
        """選択領域を切り取り"""
        if self.selected_image and self.selection_start and self.selection_end:
            # クリップボードにコピー
            self.project.clipboard = self.selected_image.copy()
            
            # 選択領域を白で塗りつぶし
            draw = ImageDraw.Draw(self.edit_bitmap)
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            draw.rectangle((x1, y1, x2, y2), fill=255)
            
            # 選択解除
            self.selection_start = None
            self.selection_end = None
            self.selected_image = None
            
            self._save_to_undo()
            self._update_preview()
            
            messagebox.showinfo('切り取り', '選択領域を切り取りました')
    
    def _delete_selection(self) -> None:
        """選択領域を削除"""
        if self.selection_start and self.selection_end:
            draw = ImageDraw.Draw(self.edit_bitmap)
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            draw.rectangle((x1, y1, x2, y2), fill=255)
            
            self.selection_start = None
            self.selection_end = None
            self.selected_image = None
            
            self._save_to_undo()
            self._update_preview()
    
    def _clear_selection(self) -> None:
        """選択を解除"""
        self.selection_start = None
        self.selection_end = None
        self.selected_image = None
        self._update_preview()
    
    # ===== 変形メソッド (2025-10-13: 反転・回転・中央配置) =====
    
    def _flip_horizontal(self) -> None:
        """左右反転"""
        self.edit_bitmap = self.edit_bitmap.transpose(Image.FLIP_LEFT_RIGHT)
        self._save_to_undo()
        self._update_preview()
    
    def _flip_vertical(self) -> None:
        """上下反転"""
        self.edit_bitmap = self.edit_bitmap.transpose(Image.FLIP_TOP_BOTTOM)
        self._save_to_undo()
        self._update_preview()
    
    def _rotate_90(self) -> None:
        """90度回転（反時計回り）"""
        self.edit_bitmap = self.edit_bitmap.transpose(Image.ROTATE_90)
        self._save_to_undo()
        self._update_preview()
    
    def _center_horizontal(self) -> None:
        """左右中央配置。選択中は選択範囲をキャンバス中心に移動し、
        選択していない場合は描画部分全体を中央に配置する。"""
        # 選択領域があればその中心を計算して移動
        if self.selected_image and self.selection_start and self.selection_end:
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            sel_width = x2 - x1
            # キャンバス中心に合わせるターゲット位置
            target_x = (Config.CANVAS_SIZE - sel_width) // 2
            offset_x = target_x - x1
            if offset_x != 0:
                # 元の領域を白で塗りつぶし
                draw = ImageDraw.Draw(self.edit_bitmap)
                # 元の領域を白で塗りつぶす（終点を含まないよう -1）
                draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
                # 新しい位置に貼り付け（透明部分を無視）
                new_x1 = max(0, min(target_x, Config.CANVAS_SIZE - sel_width))
                # 255未満のピクセルを全て貼り付け対象とする
                mask = self.selected_image.point(lambda p: 255 if p < 255 else 0)
                self.edit_bitmap.paste(self.selected_image, (new_x1, y1), mask)
                # 選択状態を更新
                self.selection_start = (new_x1, y1)
                self.selection_end = (new_x1 + sel_width, y2)
                self._save_to_undo()
                self._update_preview()
            return
        # 選択されていない場合はコンテンツ全体を対象
        bbox = self.edit_bitmap.getbbox()
        if not bbox:
            return
        x1, y1, x2, y2 = bbox
        content_width = x2 - x1
        target_x = (Config.CANVAS_SIZE - content_width) // 2
        offset_x = target_x - x1
        if offset_x == 0:
            return
        # 新しい画像を作成し中央配置
        new_img = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 255)
        content = self.edit_bitmap.crop(bbox)
        new_img.paste(content, (target_x, y1))
        self.edit_bitmap = new_img
        self._save_to_undo()
        self._update_preview()
    
    def _center_vertical(self) -> None:
        """上下中央配置。選択範囲があればその中心をキャンバス中央に移動する。"""
        # 選択領域がある場合
        if self.selected_image and self.selection_start and self.selection_end:
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            sel_height = y2 - y1
            target_y = (Config.CANVAS_SIZE - sel_height) // 2
            offset_y = target_y - y1
            if offset_y != 0:
                draw = ImageDraw.Draw(self.edit_bitmap)
                draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
                new_y1 = max(0, min(target_y, Config.CANVAS_SIZE - sel_height))
                # マスクを使用して透明部分を無視
                mask = self.selected_image.point(lambda p: 255 if p < 255 else 0)
                self.edit_bitmap.paste(self.selected_image, (x1, new_y1), mask)
                self.selection_start = (x1, new_y1)
                self.selection_end = (x2, new_y1 + sel_height)
                self._save_to_undo()
                self._update_preview()
            return
        # 選択が無い場合は全体を上下中央に配置
        bbox = self.edit_bitmap.getbbox()
        if not bbox:
            return
        x1, y1, x2, y2 = bbox
        content_height = y2 - y1
        target_y = (Config.CANVAS_SIZE - content_height) // 2
        offset_y = target_y - y1
        if offset_y == 0:
            return
        new_img = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 255)
        content = self.edit_bitmap.crop(bbox)
        new_img.paste(content, (x1, target_y))
        self.edit_bitmap = new_img
        self._save_to_undo()
        self._update_preview()
    
    def _center_both(self) -> None:
        """上下左右中央配置。選択範囲があればその中心をキャンバス中央に移動する。"""
        # 選択範囲がある場合
        if self.selected_image and self.selection_start and self.selection_end:
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            sel_w = x2 - x1
            sel_h = y2 - y1
            target_x = (Config.CANVAS_SIZE - sel_w) // 2
            target_y = (Config.CANVAS_SIZE - sel_h) // 2
            # 塗りつぶして移動
            draw = ImageDraw.Draw(self.edit_bitmap)
            draw.rectangle((x1, y1, x2 - 1, y2 - 1), fill=255)
            new_x1 = max(0, min(target_x, Config.CANVAS_SIZE - sel_w))
            new_y1 = max(0, min(target_y, Config.CANVAS_SIZE - sel_h))
            # マスクを使用して黒い部分のみを貼り付け
            mask = self.selected_image.point(lambda p: 255 if p < 255 else 0)
            self.edit_bitmap.paste(self.selected_image, (new_x1, new_y1), mask)
            self.selection_start = (new_x1, new_y1)
            self.selection_end = (new_x1 + sel_w, new_y1 + sel_h)
            self._save_to_undo()
            self._update_preview()
            return
        # 選択が無ければ全体を中央に配置
        bbox = self.edit_bitmap.getbbox()
        if not bbox:
            return
        x1, y1, x2, y2 = bbox
        content_width = x2 - x1
        content_height = y2 - y1
        target_x = (Config.CANVAS_SIZE - content_width) // 2
        target_y = (Config.CANVAS_SIZE - content_height) // 2
        new_img = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 255)
        content = self.edit_bitmap.crop(bbox)
        new_img.paste(content, (target_x, target_y))
        self.edit_bitmap = new_img
        self._save_to_undo()
        self._update_preview()
    
    # ===== 操作メソッド (2025-10-13: 元に戻す・保存・クリア等) =====
    
    def _undo(self) -> None:
        """元に戻す"""
        if len(self.undo_stack) > 1:
            # 現在の状態をリドゥスタックへ
            self.redo_stack.append(self.undo_stack.pop())
            # 1つ前の状態を復元
            self.edit_bitmap = self.undo_stack[-1].copy()
            self._update_preview()
    
    def _redo(self) -> None:
        """やり直し"""
        if self.redo_stack:
            state = self.redo_stack.pop()
            self.undo_stack.append(state)
            self.edit_bitmap = state.copy()
            self._update_preview()
    
    def _copy(self) -> None:
        """全体をコピー"""
        self.project.clipboard = self.edit_bitmap.copy()
        messagebox.showinfo('コピー', 'グリフ全体をコピーしました')
    
    def _paste(self) -> None:
        """貼り付け"""
        if not self.project.clipboard:
            messagebox.showwarning('警告', 'クリップボードが空です')
            return
        
        # クリップボードのサイズがキャンバスと同じ場合は全体を置き換え
        if self.project.clipboard.size == (Config.CANVAS_SIZE, Config.CANVAS_SIZE):
            self.edit_bitmap = self.project.clipboard.copy()
        else:
            # 中央に貼り付け
            w, h = self.project.clipboard.size
            x = (Config.CANVAS_SIZE - w) // 2
            y = (Config.CANVAS_SIZE - h) // 2
            self.edit_bitmap.paste(self.project.clipboard, (x, y))
        
        self._save_to_undo()
        self._update_preview()
    
    def _clear(self) -> None:
        """全消去"""
        if messagebox.askyesno('確認', '全て消去しますか？'):
            self.edit_bitmap = Image.new('L', (Config.CANVAS_SIZE, Config.CANVAS_SIZE), 255)
            self._save_to_undo()
            self._update_preview()
    
    def _save(self) -> None:
        """保存してプロジェクトに反映"""
        # グリフを更新
        self.project.set_glyph(self.char_code, self.edit_bitmap.copy(), is_edited=True)
        self.project.mark_as_edited(self.char_code)
        
        # コールバック実行
        if self.on_save:
            self.on_save()
        
        messagebox.showinfo('保存', '保存しました')
    
    def _save_png(self) -> None:
        """PNG保存"""
        default_name = f'U+{self.char_code:04X}.png'
        path = filedialog.asksaveasfilename(
            title='PNG保存',
            defaultextension='.png',
            initialfile=default_name,
            filetypes=[('PNG Image', '*.png'), ('All Files', '*.*')]
        )
        
        if path:
            # 透過PNG変換
            rgba_img = Image.new('RGBA', self.edit_bitmap.size, (255, 255, 255, 0))
            pixels_gray = self.edit_bitmap.load()
            pixels_rgba = rgba_img.load()
            
            for y in range(self.edit_bitmap.size[1]):
                for x in range(self.edit_bitmap.size[0]):
                    gray_value = pixels_gray[x, y]
                    alpha = 255 - gray_value
                    pixels_rgba[x, y] = (0, 0, 0, alpha)
            
            rgba_img.save(path, 'PNG')
            messagebox.showinfo('保存完了', f'保存しました:\n{path}')
    
    def _mark_as_empty(self) -> None:
        """空白グリフとしてマーク"""
        if messagebox.askyesno('確認', 'このグリフを空白としてマークしますか？'):
            # 空グリフとして登録
            self.project.glyphs[self.char_code] = GlyphData(self.char_code, None, is_edited=True)
            
            if self.on_save:
                self.on_save()
            
            self.destroy()
    
    def _load_from_other_font(self) -> None:
        """他のフォントから読み込み"""
        path = filedialog.askopenfilename(
            title='フォントファイルを選択',
            filetypes=[
                ('TrueType Font', '*.ttf'),
                ('OpenType Font', '*.otf'),
                ('All Files', '*.*')
            ]
        )
        
        if not path:
            return
        
        try:
            # 該当文字をレンダリング
            font = ImageFont.truetype(path, size=Config.FONT_RENDER_SIZE)
            char = chr(self.char_code)
            
            bitmap = FontRenderer._render_char(char, font)
            
            if bitmap:
                self.edit_bitmap = bitmap
                self._save_to_undo()
                self._update_preview()
                messagebox.showinfo('読込完了', f'フォントから読み込みました:\n{path}')
            else:
                messagebox.showwarning('警告', 'この文字はフォントに存在しません')
        
        except Exception as e:
            messagebox.showerror('エラー', f'フォント読み込み失敗:\n{e}')
    
    def _show_settings_dialog(self) -> None:
        """設定ダイアログ表示"""
        dialog = tk.Toplevel(self)
        dialog.title('設定')
        dialog.geometry('400x300')
        dialog.transient(self)
        
        tk.Label(dialog, text='エディタ設定', font=('Arial', 14, 'bold')).pack(pady=10)
        
        # グリッド表示設定
        tk.Label(dialog, text='グリッド間隔:').pack(pady=5)
        
        grid_var = tk.IntVar(value=Config.GRID_SPACING)
        tk.Scale(
            dialog,
            from_=16,
            to=128,
            orient='horizontal',
            variable=grid_var,
            length=300
        ).pack()
        
        # 適用ボタン
        def apply_settings():
            Config.GRID_SPACING = grid_var.get()
            self._update_preview()
            dialog.destroy()
        
        tk.Button(dialog, text='適用', command=apply_settings, width=10).pack(pady=20)
    
    # ===== [ADD] 2025-10-13: BLOCK10互換メソッド =====
    
    def _draw_rect(self, x0: int, y0: int, x1: int, y1: int, width: int = 1) -> None:
        """矩形描画（BLOCK10互換）"""
        draw = ImageDraw.Draw(self.edit_bitmap)
        draw.rectangle((x0, y0, x1, y1), outline=0, width=max(1, int(width)))
        self._update_preview()
    
    def _draw_ellipse(self, x0: int, y0: int, x1: int, y1: int, width: int = 1) -> None:
        """楕円描画（BLOCK10互換）"""
        draw = ImageDraw.Draw(self.edit_bitmap)
        draw.ellipse((x0, y0, x1, y1), outline=0, width=max(1, int(width)))
        self._update_preview()
    
    def _start_selection(self, x0: int, y0: int, x1: int, y1: int) -> None:
        """選択開始（BLOCK10互換）"""
        try:
            x0 = max(0, min(int(x0), Config.CANVAS_SIZE - 1))
            y0 = max(0, min(int(y0), Config.CANVAS_SIZE - 1))
            x1 = max(0, min(int(x1), Config.CANVAS_SIZE))
            y1 = max(0, min(int(y1), Config.CANVAS_SIZE))
            self.selection_start = (min(x0, x1), min(y0, y1))
            self.selection_end = (max(x0, x1), max(y0, y1))
            self.selected_image = self.edit_bitmap.crop((*self.selection_start, *self.selection_end)).copy()
            self._update_preview()
        except Exception:
            pass
    
    def _on_copy(self) -> None:
        """コピー（BLOCK10互換）"""
        self._copy()
    
    def _on_cut(self) -> None:
        """切り取り（BLOCK10互換）"""
        self._cut_selection()
    
    def _on_paste(self) -> None:
        """貼り付け（BLOCK10互換）"""
        self._paste()
    
    def commit_to_project_without_close(self) -> None:
        """エディタ内容をプロジェクトへ反映（BLOCK10互換）"""
        self.project.glyphs[self.char_code] = GlyphData(
            self.char_code, 
            self.edit_bitmap.copy(), 
            is_edited=True
        )
        self.project.dirty = True
        if self.on_save:
            self.on_save()
    
    def _save_from_editor(self, event: Optional[tk.Event] = None) -> None:
        """⌘S/Ctrl+S: 保存（BLOCK10互換）"""
        self.commit_to_project_without_close()
        if hasattr(self.master, '_save_project_dialog'):
            self.master._save_project_dialog()  # type: ignore
    
    # ===== [ADD] 2025-10-17: 座標変換・ハンドル取得メソッド =====
    
    def _normalize_selection(self) -> None:
        """選択範囲を正規化"""
        if self.selection_start and self.selection_end:
            x1, y1 = self.selection_start
            x2, y2 = self.selection_end
            self.selection_start = (min(x1, x2), min(y1, y2))
            self.selection_end = (max(x1, x2), max(y1, y2))
    
    def _canvas_to_image_coords(self, canvas_x: float, canvas_y: float) -> Tuple[int, int]:
        """キャンバス座標を画像座標に変換"""
        c_x = self.preview_canvas.canvasx(canvas_x)
        c_y = self.preview_canvas.canvasy(canvas_y)
        img_x = int(c_x / self.zoom_level)
        img_y = int(c_y / self.zoom_level)
        img_x = max(0, min(img_x, Config.CANVAS_SIZE - 1))
        img_y = max(0, min(img_y, Config.CANVAS_SIZE - 1))
        return img_x, img_y
    
    def _get_handle_at(self, canvas_x: float, canvas_y: float) -> Optional[str]:
        """ハンドルを取得"""
        if not (self.selection_start and self.selection_end):
            return None
        
        self._normalize_selection()
        x1, y1 = self.selection_start
        x2, y2 = self.selection_end
        
        cx1 = x1 * self.zoom_level
        cy1 = y1 * self.zoom_level
        cx2 = x2 * self.zoom_level
        cy2 = y2 * self.zoom_level
        
        cx_mid = (cx1 + cx2) / 2
        cy_mid = (cy1 + cy2) / 2
        
        threshold = 8
        
        handles = {
            'nw': (cx1, cy1),
            'n': (cx_mid, cy1),
            'ne': (cx2, cy1),
            'e': (cx2, cy_mid),
            'se': (cx2, cy2),
            's': (cx_mid, cy2),
            'sw': (cx1, cy2),
            'w': (cx1, cy_mid)
        }
        
        scroll_x = self.preview_canvas.canvasx(canvas_x)
        scroll_y = self.preview_canvas.canvasy(canvas_y)
        
        for handle, (hx, hy) in handles.items():
            if abs(scroll_x - hx) <= threshold and abs(scroll_y - hy) <= threshold:
                return handle
        
        return None
    
    def _get_text_layer_handle_at(self, canvas_x: float, canvas_y: float) -> Optional[str]:
        """テキストレイヤーのハンドルを取得"""
        if not self.text_layer:
            return None
        
        x_pos, y_pos = self.text_layer_pos
        x_end = x_pos + self.text_layer.width
        y_end = y_pos + self.text_layer.height
        
        cx1 = x_pos * self.zoom_level
        cy1 = y_pos * self.zoom_level
        cx2 = x_end * self.zoom_level
        cy2 = y_end * self.zoom_level
        
        cx_mid = (cx1 + cx2) / 2
        cy_mid = (cy1 + cy2) / 2
        
        threshold = 8
        
        handles = {
            'nw': (cx1, cy1),
            'n': (cx_mid, cy1),
            'ne': (cx2, cy1),
            'e': (cx2, cy_mid),
            'se': (cx2, cy2),
            's': (cx_mid, cy2),
            'sw': (cx1, cy2),
            'w': (cx1, cy_mid)
        }
        
        scroll_x = self.preview_canvas.canvasx(canvas_x)
        scroll_y = self.preview_canvas.canvasy(canvas_y)
        
        for handle, (hx, hy) in handles.items():
            if abs(scroll_x - hx) <= threshold and abs(scroll_y - hy) <= threshold:
                return handle
        
        return None
    
    def _resize_by_handle(self, x: int, y: int) -> None:
        """選択領域をハンドルでリサイズ"""
        if not (self.resize_origin and self.resize_handle):
            return
        
        (x1, y1), (x2, y2) = self.resize_origin
        handle = self.resize_handle
        
        new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
        
        if 'n' in handle:
            new_y1 = min(y, y2 - 1)
        if 's' in handle:
            new_y2 = max(y, y1 + 1)
        if 'w' in handle:
            new_x1 = min(x, x2 - 1)
        if 'e' in handle:
            new_x2 = max(x, x1 + 1)
        
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(Config.CANVAS_SIZE, new_x2)
        new_y2 = min(Config.CANVAS_SIZE, new_y2)
        
        self.resize_preview_rect = (new_x1, new_y1, new_x2, new_y2)
    
    def _apply_resize_by_handle(self) -> None:
        """選択領域のリサイズを確定"""
        if not (self.selected_image and self.resize_preview_rect):
            return
        
        x1, y1, x2, y2 = self.resize_preview_rect
        new_w = x2 - x1
        new_h = y2 - y1
        
        if new_w <= 0 or new_h <= 0:
            return
        
        resized = self.selected_image.resize((new_w, new_h), Image.LANCZOS)
        
        draw = ImageDraw.Draw(self.edit_bitmap)
        old_x1, old_y1 = self.selection_start
        old_x2, old_y2 = self.selection_end
        draw.rectangle((old_x1, old_y1, old_x2, old_y2), fill=255)
        
        self.edit_bitmap.paste(resized, (x1, y1))
        
        self.selection_start = (x1, y1)
        self.selection_end = (x2, y2)
        self.selected_image = resized
        
        self._save_to_undo()
        self._update_preview()
    
    def _resize_text_layer_by_handle(self, x: int, y: int) -> None:
        """テキストレイヤーをハンドルでリサイズ"""
        if not (self.resize_origin and self.resize_handle):
            return
        
        (x1, y1), (x2, y2) = self.resize_origin
        handle = self.resize_handle
        
        new_x1, new_y1, new_x2, new_y2 = x1, y1, x2, y2
        
        if 'n' in handle:
            new_y1 = min(y, y2 - 1)
        if 's' in handle:
            new_y2 = max(y, y1 + 1)
        if 'w' in handle:
            new_x1 = min(x, x2 - 1)
        if 'e' in handle:
            new_x2 = max(x, x1 + 1)
        
        new_x1 = max(0, new_x1)
        new_y1 = max(0, new_y1)
        new_x2 = min(Config.CANVAS_SIZE, new_x2)
        new_y2 = min(Config.CANVAS_SIZE, new_y2)
        
        self.resize_preview_rect = (new_x1, new_y1, new_x2, new_y2)
    
    def _apply_text_layer_resize(self) -> None:
        """テキストレイヤーのリサイズを確定"""
        if not (self.text_layer and self.resize_preview_rect):
            return
        
        x1, y1, x2, y2 = self.resize_preview_rect
        new_w = x2 - x1
        new_h = y2 - y1
        
        if new_w <= 0 or new_h <= 0:
            return
        
        if self.text_layer_original:
            resized = self.text_layer_original.resize((new_w, new_h), Image.LANCZOS)
        else:
            resized = self.text_layer.resize((new_w, new_h), Image.LANCZOS)
        
        self.text_layer = resized
        self.text_layer_pos = (x1, y1)
        self.text_layer_resized_size = (new_w, new_h)
        self.text_layer_resized_pos = (x1, y1)

        # [ADD] 2025-10-23: テキストエッジマスクもリサイズに追従させる
        # 既存のエッジマスクが存在する場合、現在のサイズに合わせて拡大縮小する。
        # NEARESTを使うことでマスクのバイナリ性を保つ。
        try:
            if getattr(self, 'text_edge_mask', None):
                self.text_edge_mask = self.text_edge_mask.resize((new_w, new_h), Image.NEAREST)
            if getattr(self, 'text_edge_mask_commit', None):
                self.text_edge_mask_commit = self.text_edge_mask_commit.resize((new_w, new_h), Image.NEAREST)
        except Exception:
            pass
        
        self.resize_preview_rect = None
        self._update_preview()
    
    # ===== [ADD] 2025-10-17: ズーム機能 =====
    
    def _on_space_press(self, event: tk.Event) -> None:
        """スペースキー押下でパンモード"""
        if not self.is_panning:
            self.is_panning = True
            self.preview_canvas.config(cursor='hand2')
    
    def _on_space_release(self, event: tk.Event) -> None:
        """スペースキー解放でパンモード解除"""
        if self.is_panning:
            self.is_panning = False
            self.preview_canvas.config(cursor='')
    
    def _zoom_in(self) -> None:
        """ズームイン"""
        """
        ズームイン：現在の倍率より大きい最も近い倍率に切り替える。定義されている範囲よりも大きい場合は2倍にする。
        """
        # 既存レベルから次に大きい倍率を取得
        sorted_levels = sorted(set(self.zoom_levels + [self.zoom_level]))
        try:
            idx = sorted_levels.index(self.zoom_level)
        except ValueError:
            # 万が一現在の倍率がリストにない場合は標準倍率1.0として扱う
            idx = sorted_levels.index(1.0) if 1.0 in sorted_levels else 0
        # 次の倍率が存在すればそれを採用
        if idx < len(sorted_levels) - 1:
            new_zoom = sorted_levels[idx + 1]
        else:
            # 最大値より大きくする場合は倍にする
            new_zoom = self.zoom_level * 2
        self.zoom_level = new_zoom
        self.zoom_label.config(text=f'{int(self.zoom_level * 100)}%')
        self._update_preview()
    
    def _zoom_out(self) -> None:
        """ズームアウト"""
        """
        ズームアウト：現在の倍率より小さい最も近い倍率に切り替える。定義されている範囲よりも小さい場合は半分にする。
        """
        # 既存レベルから次に小さい倍率を取得
        sorted_levels = sorted(set(self.zoom_levels + [self.zoom_level]))
        try:
            idx = sorted_levels.index(self.zoom_level)
        except ValueError:
            idx = sorted_levels.index(1.0) if 1.0 in sorted_levels else 0
        if idx > 0:
            new_zoom = sorted_levels[idx - 1]
        else:
            new_zoom = max(self.zoom_level / 2, 0.01)
        self.zoom_level = new_zoom
        self.zoom_label.config(text=f'{int(self.zoom_level * 100)}%')
        self._update_preview()
    
    def _reset_zoom(self) -> None:
        """ズームリセット"""
        self.zoom_level = 1.0
        self.zoom_label.config(text='100%')
        self.pan_offset = [0, 0]
        self._update_preview()
    
    # ===== [ADD] 2025-10-17: マウスイベントハンドラ =====
    
    def _on_mouse_down(self, event: tk.Event) -> None:
        """マウスボタン押下"""
        self.drag_start = (event.x, event.y)
        if self.is_panning:
            self.preview_canvas.scan_mark(event.x, event.y)
            return
        
        x, y = self._canvas_to_image_coords(event.x, event.y)
        
        # テキストレイヤーのモード
        if self.is_text_mode and self.text_layer:
            if self.current_tool == 'move':
                x_pos, y_pos = self.text_layer_pos
                x_end = x_pos + self.text_layer.width
                y_end = y_pos + self.text_layer.height
                if x_pos <= x < x_end and y_pos <= y < y_end:
                    self.is_moving = True
                    self.move_start_offset = (x - x_pos, y - y_pos)
                    return
            elif self.current_tool == 'resize':
                handle = self._get_text_layer_handle_at(event.x, event.y)
                if handle:
                    self.is_resizing = True
                    self.resize_handle = handle
                    x_pos, y_pos = self.text_layer_pos
                    x_end = x_pos + self.text_layer.width
                    y_end = y_pos + self.text_layer.height
                    self.resize_origin = ((x_pos, y_pos), (x_end, y_end))
                    self.resize_start_point = (x, y)
                    return
        
        # 通常モード
        if self.current_tool == 'select':
            # [ADD] 2025-10-23: 選択モードでも既存選択の移動・リサイズを可能にする
            if self.selected_image and self.selection_start and self.selection_end:
                self._normalize_selection()
                # まずリサイズハンドルをクリックしているかチェック
                handle = self._get_handle_at(event.x, event.y)
                if handle:
                    # リサイズ開始
                    self.is_resizing = True
                    self.resize_handle = handle
                    self.resize_origin = (self.selection_start, self.selection_end)
                    self.resize_start_point = (x, y)
                    return
                # ハンドル以外で選択領域内をクリックした場合は移動
                x0, y0 = self.selection_start
                x1, y1 = self.selection_end
                if x0 <= x < x1 and y0 <= y < y1:
                    self.is_moving = True
                    self.move_start_offset = (x - x0, y - y0)
                    self.move_current_pos = (x0, y0)
                    return
            # 既存選択を無視して新規選択を開始
            self.selection_start = (x, y)
            self.selection_end = (x, y)
            self.selected_image = None
            self.is_moving = False
            self.is_resizing = False
            self.shape_start = None
            self.shape_end = None
        elif self.current_tool in ['pen', 'eraser']:
            if self.is_text_mode:
                messagebox.showinfo('情報', 'テキスト入力モード中は描画できません\n「決定」または「キャンセル」してください')
                return
            self.is_drawing = True
            self.last_x = x
            self.last_y = y
            self._draw_point(x, y)
        elif self.current_tool == 'fill':
            if self.is_text_mode:
                messagebox.showinfo('情報', 'テキスト入力モード中は塗りつぶしできません')
                return
            self._flood_fill(x, y)
        elif self.current_tool == 'move':
            if self.selected_image and self.selection_start and self.selection_end:
                self._normalize_selection()
                x0, y0 = self.selection_start
                x1, y1 = self.selection_end
                if x0 <= x < x1 and y0 <= y < y1:
                    self.is_moving = True
                    self.move_start_offset = (x - x0, y - y0)
                    self.move_current_pos = (x0, y0)
                    return
        elif self.current_tool == 'resize':
            if self.selected_image and self.selection_start and self.selection_end:
                self._normalize_selection()
                handle = self._get_handle_at(event.x, event.y)
                if handle:
                    self.is_resizing = True
                    self.resize_handle = handle
                    self.resize_origin = (self.selection_start, self.selection_end)
                    self.resize_start_point = (x, y)
                    return
        elif self.current_tool in ['line', 'rect', 'ellipse']:
            if self.is_text_mode:
                messagebox.showinfo('情報', 'テキスト入力モード中は図形描画できません')
                return
            self.shape_start = (x, y)
            self.shape_end = (x, y)
        elif self.current_tool == 'guide':
            self.guidelines.append(('h', y))
            self.guidelines.append(('v', x))
            self._update_preview()
    
    def _on_mouse_drag(self, event: tk.Event) -> None:
        """マウスドラッグ"""
        if self.is_panning:
            self.preview_canvas.scan_dragto(event.x, event.y, gain=1)
            self._update_preview()
            return
        
        x, y = self._canvas_to_image_coords(event.x, event.y)
        
        # テキストレイヤー移動
        if self.is_text_mode and self.is_moving and self.text_layer:
            if self.move_start_offset:
                offset_x, offset_y = self.move_start_offset
                dest_x = x - offset_x
                dest_y = y - offset_y
                
                w = self.text_layer.width
                h = self.text_layer.height
                dest_x = max(-w + 10, min(dest_x, Config.CANVAS_SIZE - 10))
                dest_y = max(-h + 10, min(dest_y, Config.CANVAS_SIZE - 10))
                
                self.text_layer_pos = (dest_x, dest_y)
                self._update_preview_fast()
            return
        
        # テキストレイヤーリサイズ
        if self.is_text_mode and self.is_resizing and self.text_layer:
            if self.resize_origin and self.resize_handle:
                self._resize_text_layer_by_handle(x, y)
                self._update_preview_fast()
            return
        
        # 通常の操作
        if self.current_tool == 'select':
            # [ADD] 2025-10-23: 選択モードでも移動・リサイズを可能にする
            if self.is_moving and self.move_start_offset and self.selected_image:
                offset_x, offset_y = self.move_start_offset
                dest_x = x - offset_x
                dest_y = y - offset_y
                w = self.selected_image.width
                h = self.selected_image.height
                dest_x = max(-w + 10, min(dest_x, Config.CANVAS_SIZE - 10))
                dest_y = max(-h + 10, min(dest_y, Config.CANVAS_SIZE - 10))
                self.move_current_pos = (dest_x, dest_y)
                self._update_preview_fast()
                return
            if self.is_resizing and self.selected_image and self.resize_origin and self.resize_handle:
                self._resize_by_handle(x, y)
                self._update_preview_fast()
                return
            # 移動・リサイズでない場合は通常の選択範囲更新
            self.selection_end = (x, y)
            self._update_preview()
        elif self.is_drawing and self.current_tool in ['pen', 'eraser']:
            if self.last_x is not None and self.last_y is not None:
                self._draw_line(self.last_x, self.last_y, x, y)
            self.last_x = x
            self.last_y = y
            self._update_preview()
        elif self.current_tool == 'move' and self.is_moving:
            if self.move_start_offset:
                offset_x, offset_y = self.move_start_offset
                dest_x = x - offset_x
                dest_y = y - offset_y
                w = self.selected_image.width if self.selected_image else 0
                h = self.selected_image.height if self.selected_image else 0
                dest_x = max(-w + 10, min(dest_x, Config.CANVAS_SIZE - 10))
                dest_y = max(-h + 10, min(dest_y, Config.CANVAS_SIZE - 10))
                self.move_current_pos = (dest_x, dest_y)
                self._update_preview()
        elif self.current_tool == 'resize' and self.is_resizing:
            if self.resize_origin and self.resize_handle:
                self._resize_by_handle(x, y)
                self._update_preview()
        elif self.current_tool in ['line', 'rect', 'ellipse'] and self.shape_start:
            self.shape_end = (x, y)
            self._update_preview()
    
    def _on_mouse_up(self, event: tk.Event) -> None:
        """マウスボタン解放"""
        if self.is_panning:
            self.is_panning = False
            self.pan_start = None
            return

        # テキストレイヤー移動終了
        if self.is_text_mode and self.is_moving:
            self.is_moving = False
            self.move_start_offset = None
            self.move_current_pos = None
            self._update_preview()
            return
        
        # テキストレイヤーリサイズ終了
        if self.is_text_mode and self.is_resizing:
            self.is_resizing = False
            self._apply_text_layer_resize()
            self.resize_origin = None
            self.resize_handle = None
            self.resize_start_point = None
            self.resize_preview_rect = None
            return

        x, y = self._canvas_to_image_coords(event.x, event.y)

        # ペン・消しゴム終了
        if self.is_drawing and self.current_tool in ['pen', 'eraser']:
            self.is_drawing = False
            self._save_to_undo()
            self._update_preview()
            return


        # 選択モードでの移動・リサイズ・選択確定
        if self.current_tool == 'select' and self.selection_start:
            # 移動確定
            if self.is_moving:
                self.is_moving = False
                self._apply_translation()
                self.move_start_offset = None
                self.move_current_pos = None
                return
            # リサイズ確定
            if self.is_resizing:
                self.is_resizing = False
                self._apply_resize_by_handle()
                self.resize_origin = None
                self.resize_handle = None
                self.resize_start_point = None
                self.resize_preview_rect = None
                return
            # 新規選択の確定
            self.selection_end = (x, y)
            self._finalize_selection()
            self._update_preview()
            return

        # 移動確定
        if self.current_tool == 'move' and self.is_moving:
            self.is_moving = False
            self._apply_translation()
            self.move_start_offset = None
            self.move_current_pos = None
            return

        # リサイズ確定
        if self.current_tool == 'resize' and self.is_resizing:
            self.is_resizing = False
            self._apply_resize_by_handle()
            self.resize_origin = None
            self.resize_handle = None
            self.resize_start_point = None
            self.resize_preview_rect = None
            return

        # 図形描画確定
        if self.current_tool in ['line', 'rect', 'ellipse'] and self.shape_start:
            self.shape_end = (x, y)
            self._commit_shape(self.shape_start, self.shape_end)
            self.shape_start = None
            self.shape_end = None
            return

        self._update_preview()
    
    def _on_mouse_move(self, event: tk.Event) -> None:
        """マウス移動"""
        # リサイズツール選択時、ハンドルにカーソルを合わせたらカーソル変更
        if self.current_tool == 'resize' and not self.is_resizing:
            if self.is_text_mode and self.text_layer:
                handle = self._get_text_layer_handle_at(event.x, event.y)
            else:
                handle = self._get_handle_at(event.x, event.y)
            
            if handle:
                cursor_map = {
                    'nw': 'size_nw_se', 'se': 'size_nw_se',
                    'ne': 'size_ne_sw', 'sw': 'size_ne_sw',
                    'n': 'size_ns', 's': 'size_ns',
                    'e': 'size_we', 'w': 'size_we'
                }
                self.preview_canvas.config(cursor=cursor_map.get(handle, ''))
            else:
                self.preview_canvas.config(cursor='')
        
        # ペン・消しゴムツール選択時、ブラシカーソル表示
        if self.current_tool in ['pen', 'eraser']:
            if self.brush_cursor:
                self.preview_canvas.delete(self.brush_cursor)
            
            radius = int(self.brush_size * self.zoom_level / 2)
            
            self.brush_cursor = self.preview_canvas.create_oval(
                event.x - radius, event.y - radius,
                event.x + radius, event.y + radius,
                outline='red' if self.current_tool == 'pen' else 'blue',
                width=1,
                dash=(2, 2)
            )

    # ===== [BLOCK5.7-END] =====

    # [ADD] 2025-10-23: 偏旁パーツ貼り付けメソッド
    def insert_part_image(self, part_image: Image.Image, scale_hint: float = 1.0, offset_hint: Tuple[float, float] = (0.0, 0.0)) -> None:
        """
        偏旁パーツを編集キャンバスに貼り付ける。
        part_image: グレースケールImage（255は透過扱い）
        scale_hint: 貼り付け時の倍率（1.0=原寸）
        offset_hint: キャンバス中心からの相対オフセット (x,y)（-1.0〜1.0程度）
        """
        try:
            # 型チェック
            from PIL import Image as PILImage  # 遅延インポート
            if not isinstance(part_image, PILImage.Image):
                raise ValueError('無効な画像')
            # グレースケールに変換
            if part_image.mode != 'L':
                part_image = part_image.convert('L')
            # 貼り付けサイズ計算
            orig_w, orig_h = part_image.size
            new_w = max(1, int(orig_w * scale_hint))
            new_h = max(1, int(orig_h * scale_hint))
            if (new_w, new_h) != (orig_w, orig_h):
                resized = part_image.resize((new_w, new_h), Image.Resampling.NEAREST)
            else:
                resized = part_image.copy()
            # オフセット計算 (キャンバス中心 + offset_hint * canvas_size)
            x = (Config.CANVAS_SIZE - new_w) // 2 + int(offset_hint[0] * Config.CANVAS_SIZE)
            y = (Config.CANVAS_SIZE - new_h) // 2 + int(offset_hint[1] * Config.CANVAS_SIZE)
            # 範囲内に収める
            x = max(0, min(x, Config.CANVAS_SIZE - new_w))
            y = max(0, min(y, Config.CANVAS_SIZE - new_h))
            # 透過用マスク: 255→0(透明), それ以外→255(不透明)
            mask = resized.point(lambda p: 0 if p == 255 else 255)
            # ビットマップに貼り付け
            self.edit_bitmap.paste(resized, (x, y), mask)
            # Undo履歴追加
            self._save_to_undo()
            # プレビュー更新
            self._update_preview()
            # ツールを移動モードに変更してすぐに調整できるようにする
            self._set_tool('move')
        except Exception as e:
            messagebox.showerror('エラー', f'パーツ貼り付けに失敗しました: {e}')

# ===== [BLOCK5.7-END] =====











# ===== [BLOCK6-BEGIN] メインアプリケーション (2025-10-11: 型ヒント追加、エラーハンドリング改善) =====

class FontEditorApp(tk.Tk):
    """メインアプリケーション"""
    
    def __init__(self) -> None:
        self._open_editors: List[GlyphEditor] = []  # 開いているエディタ追跡
        super().__init__()
        
        self.title('フォントエディタ - ハイブリッド方式(全機能版)')
        self.geometry(f'{Config.WINDOW_WIDTH}x{Config.WINDOW_HEIGHT}')
        
        self.project: FontProject = FontProject()
        self.bg_loader: Optional['BackgroundLoader'] = None  # バックグラウンドローダー (2025-10-03)
        self.current_filter: str = 'all'  # 現在のフィルタ (2025-10-03)
        
        self._setup_ui()
        
        # キーボードショートカット (2025-10-03)
        self.bind('<Control-z>', lambda e: self.grid_view.winfo_children() and None)
        self.bind('<Control-o>', lambda e: self._open_font())
        self.bind('<Control-f>', lambda e: self._show_filter_dialog())
        self.bind('<Control-p>', lambda e: self._show_text_preview())
        
        # バックグラウンド読み込み結果チェック用タイマー (2025-10-03)
        self.after(100, self._check_bg_loader)
    
    def _setup_ui(self) -> None:
        """UI構築"""
        # メニューバー
        menubar = tk.Menu(self)
        self.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='ファイル', menu=file_menu)
        file_menu.add_command(label='フォントを開く...', command=self._open_font)
        file_menu.add_command(label='プロジェクトを保存...', command=self._save_project_dialog)
        file_menu.add_command(label='プロジェクトを開く...', command=self._open_project_dialog)
        file_menu.add_separator()
        file_menu.add_command(label='バックグラウンド読み込み停止', command=self._stop_bg_loading)
        file_menu.add_separator()
        file_menu.add_command(label='終了', command=self.quit)
        
        # 表示メニュー (2025-10-03)
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='表示', menu=view_menu)
        view_menu.add_command(label='グリフフィルタ...', command=self._show_filter_dialog)
        view_menu.add_command(label='テキストプレビュー...', command=self._show_text_preview)
        
        # エクスポートメニュー (2025-10-03)
        export_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label='エクスポート', menu=export_menu)
        export_menu.add_command(label='BDF形式で保存...', command=self._export_bdf)
        export_menu.add_command(label='TTF形式で保存... (高品質アウトライン)', command=self._export_ttf)
        export_menu.add_separator()
        export_menu.add_command(label='PNG一括書き出し...', command=self._export_png_batch)
        
        # ツールバー
        toolbar = tk.Frame(self, bg=Config.COLOR_BG)
        toolbar.pack(side='top', fill='x', padx=5, pady=5)
        
        tk.Button(toolbar, text='📂 フォントを開く', command=self._open_font).pack(side='left', padx=2)
        tk.Button(toolbar, text='🔍 フィルタ', command=self._show_filter_dialog).pack(side='left', padx=2)
        tk.Button(toolbar, text='👁️ プレビュー', command=self._show_text_preview).pack(side='left', padx=2)
        # [ADD] 2025-10-23: 部首パレットを開くボタン
        tk.Button(toolbar, text='部首', command=self._open_parts_palette).pack(side='left', padx=2)
        
        # 範囲選択ドロップダウン (2025-10-03)
        tk.Label(toolbar, text='文字範囲:', bg=Config.COLOR_BG).pack(side='left', padx=(20, 5))
        self.range_var = tk.StringVar(value=Config.DEFAULT_RANGE)
        range_combo = ttk.Combobox(
            toolbar,
            textvariable=self.range_var,
            values=list(Config.CHAR_RANGES.keys()),
            state='readonly',
            width=30
        )
        range_combo.pack(side='left', padx=5)
        range_combo.bind('<<ComboboxSelected>>', self._on_range_changed)
        
        # グリッドビュー
        self.grid_view = GridView(self, self.project, self._on_edit_char)
        self.grid_view.pack(fill='both', expand=True)
        
        # ステータスバー
        self.status_label = tk.Label(self, text='ファイル: なし', anchor='w', relief='sunken')
        self.status_label.pack(side='bottom', fill='x')
    
    def _open_font(self) -> None:
        """フォント読み込み (2025-10-11: エラーハンドリング改善、2025-11-10: TTC対応)"""
        path = filedialog.askopenfilename(
            title='フォントファイルを選択',
            filetypes=[
                ('Font Files', '*.ttf;*.otf;*.ttc'),
                ('TrueType Font', '*.ttf'),
                ('TrueType Collection', '*.ttc'),
                ('OpenType Font', '*.otf'),
                ('All Files', '*.*')
            ]
        )

        if not path:
            return

        # TTCファイルの場合はフォントインデックスを選択 (2025-11-10)
        font_index = 0
        if path.lower().endswith('.ttc'):
            font_index = self._select_ttc_font_index(path)

        # プロジェクト初期化
        self.project.font_path = path

        # 現在の範囲の文字コード取得
        char_codes = self.project.get_char_codes()

        # プログレスウィンドウ作成
        progress_win = tk.Toplevel(self)
        progress_win.title('読み込み中...')
        progress_win.geometry('500x150')
        progress_win.transient(self)
        progress_win.grab_set()

        tk.Label(
            progress_win,
            text='フォントを読み込んでいます...',
            font=('Arial', 12)
        ).pack(pady=10)

        progress_var = tk.IntVar(value=0)
        progress_bar = ttk.Progressbar(
            progress_win,
            maximum=len(char_codes),
            variable=progress_var,
            length=400
        )
        progress_bar.pack(pady=10)

        progress_label = tk.Label(
            progress_win,
            text='0 / 0 文字',
            font=('Arial', 10)
        )
        progress_label.pack()

        # プログレスバー更新用コールバック
        def progress_callback(current: int, total: int) -> None:
            """プログレス更新"""
            progress_var.set(current)
            progress_label.config(text=f'{current} / {total} 文字')
            progress_win.update()

        # 同期読み込み実行 (2025-11-10: font_index追加)
        success = FontRenderer.load_font(
            path,
            char_codes,
            self.project,
            progress_callback,
            font_index
        )
        
        if not success:
            progress_win.destroy()
            return
        
        # ★★★ 重要: 範囲を読み込み済みとしてマーク ★★★
        self.project.mark_range_loaded(self.project.char_range)
        
        # プログレスウィンドウ閉じる
        progress_win.destroy()
        
        # グリッド表示更新
        self.grid_view.refresh()
        self._update_status()
        
        # 統計情報取得 (2025-10-05 22:00: エラー修正 - defined と empty を正しく計算)
        total = len(char_codes)
        empty = self.project.get_empty_count()
        defined = total - empty
        
        # 読み込み完了メッセージ
        messagebox.showinfo(
            '読込完了',
            f'フォント読込完了\n\n'
            f'定義済み: {defined} / 空白: {empty}\n\n'
            f'バックグラウンドで他の範囲も読み込みます\n'
            f'※編集画面をクリックすると表示されます'
        )
        
        # ===== バックグラウンドで残りの範囲を読み込み =====
        self._start_background_loading(path, font_index)

    def _get_ttc_font_count(self, ttc_path: str) -> int:
        """TTCファイルに含まれるフォント数を取得 (2025-11-10: 新規追加)"""
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

    def _select_ttc_font_index(self, ttc_path: str) -> int:
        """TTCファイルからフォントインデックスを選択 (2025-11-10: 新規追加)"""
        # TTCファイルに含まれるフォント数を調べる
        font_count = self._get_ttc_font_count(ttc_path)

        if font_count <= 1:
            return 0

        # ダイアログでフォントインデックスを選択
        dialog = tk.Toplevel(self)
        dialog.title('TTCフォント選択')
        dialog.geometry('400x250')
        dialog.transient(self)
        dialog.grab_set()

        tk.Label(
            dialog,
            text=f'このTTCファイルには{font_count}個のフォントが含まれています。\n使用するフォントを選択してください:',
            font=('Arial', 10)
        ).pack(pady=10)

        # プレビューフレーム
        preview_frame = tk.Frame(dialog)
        preview_frame.pack(pady=10, fill='both', expand=True)

        # スクロールバー付きリストボックス
        scrollbar = tk.Scrollbar(preview_frame)
        scrollbar.pack(side='right', fill='y')

        listbox = tk.Listbox(preview_frame, yscrollcommand=scrollbar.set, height=8)
        listbox.pack(side='left', fill='both', expand=True, padx=10)
        scrollbar.config(command=listbox.yview)

        for i in range(font_count):
            listbox.insert('end', f'フォント {i}')

        listbox.selection_set(0)

        selected_index = tk.IntVar(value=0)

        def on_ok():
            sel = listbox.curselection()
            if sel:
                selected_index.set(sel[0])
            dialog.destroy()

        # OKボタン
        tk.Button(dialog, text='OK', command=on_ok, width=10).pack(pady=10)

        dialog.wait_window()
        return selected_index.get()

    def _start_background_loading(self, font_path: str, font_index: int = 0) -> None:
        """バックグラウンド読み込み開始 (2025-11-10: TTC対応)"""
        # 既存のローダー停止
        if self.bg_loader:
            self.bg_loader.stop()

        # 新規ローダー作成
        self.bg_loader = BackgroundLoader(self.project, self._on_bg_status_update)
        self.bg_loader.start_background_load(font_path, self.project.char_range, font_index)
        
        # ステータス更新
        self.status_label.config(
            text=f'{Path(font_path).name} - バックグラウンド読み込み開始...'
        )
    
    def _check_bg_loader(self) -> None:
        """バックグラウンドローダーの結果チェック"""
        if self.bg_loader:
            self.bg_loader.check_results()
        
        # 定期的に実行
        self.after(100, self._check_bg_loader)
    
    def _on_bg_status_update(self, result: Dict[str, Any]) -> None:
        """バックグラウンド読み込みステータス更新"""
        result_type = result.get('type')
        message = result.get('message', '')
        
        if result_type == 'status':
            # 進行中
            if self.project.font_path:
                self.status_label.config(
                    text=f'{Path(self.project.font_path).name} - {message}'
                )
                
        elif result_type == 'complete':
            # 完了
            if self.project.font_path:
                self.status_label.config(
                    text=f'{Path(self.project.font_path).name} - バックグラウンド読み込み完了'
                )
                
        elif result_type == 'error':
            # エラー
            self.status_label.config(text=f'エラー: {message}')
            messagebox.showerror('エラー', message)
    
    def _on_range_changed(self, event: tk.Event) -> None:
        """文字範囲変更時の処理"""
        range_name = self.range_var.get()
        self.project.set_range(range_name)
        self.grid_view.refresh()
        self._update_status()
    
    def _on_edit_char(self, char_code: int) -> None:
        """文字編集ウィンドウを開く"""
        def on_save() -> None:
            self.grid_view.refresh()
            self._update_status()
        
        editor = GlyphEditor(self, self.project, char_code, on_save)
        self._open_editors.append(editor)
    
    def _update_status(self) -> None:
        """ステータス更新"""
        if self.project.font_path:
            total = len(self.project.get_char_codes())
            empty = self.project.get_empty_count()
            defined = total - empty
            
            range_name = self.range_var.get()
            
            self.status_label.config(
                text=f'{Path(self.project.font_path).name} | {range_name} | '
                     f'定義済み: {defined} / 空白: {empty}'
            )
        else:
            self.status_label.config(text='ファイル: なし')
    
    def _stop_bg_loading(self) -> None:
        """バックグラウンド読み込み停止"""
        if self.bg_loader:
            self.bg_loader.stop()
            self.status_label.config(text='バックグラウンド読み込み停止')
            messagebox.showinfo('停止', 'バックグラウンド読み込みを停止しました')
    
    def _show_filter_dialog(self) -> None:
        """フィルタダイアログを表示"""
        dialog = GlyphFilterDialog(self, self.project, self.current_filter)
        self.wait_window(dialog)
        
        if hasattr(dialog, 'result'):
            self.current_filter = dialog.result
            self.grid_view.set_filter(self.current_filter)
    
    def _show_text_preview(self) -> None:
        """テキストプレビューダイアログを表示"""
        if not self.project.glyphs:
            messagebox.showwarning('警告', 'フォントが読み込まれていません')
            return
        
        TextPreviewDialog(self, self.project)
    
    def _export_bdf(self) -> None:
        """BDF書き出し"""
        if not self.project.glyphs:
            messagebox.showwarning('警告', 'フォントが読み込まれていません')
            return
        
        path = filedialog.asksaveasfilename(
            title='BDF保存',
            defaultextension='.bdf',
            filetypes=[('BDF Font', '*.bdf'), ('All Files', '*.*')]
        )
        
        if path:
            if FontExporter.export_bdf(self.project, path):
                messagebox.showinfo('書き出し完了', f'BDF書き出し完了:\n{path}')
    
    def _export_ttf(self) -> None:
        """TTF書き出し"""
        if not self.project.glyphs:
            messagebox.showwarning('警告', 'フォントが読み込まれていません')
            return
        
        if not self.project.original_ttf_path:
            messagebox.showwarning('警告', '元のTTFファイルが必要です')
            return
        
        path = filedialog.asksaveasfilename(
            title='TTF保存',
            defaultextension='.ttf',
            filetypes=[('TrueType Font', '*.ttf'), ('All Files', '*.*')]
        )
        
        if path:
            if TTFExporter.export_ttf(self.project, path):
                messagebox.showinfo('書き出し完了', f'TTF書き出し完了:\n{path}')
    
    def _export_png_batch(self) -> None:
        """PNG一括書き出し"""
        if not self.project.glyphs:
            messagebox.showwarning('警告', 'フォントが読み込まれていません')
            return
        
        folder = filedialog.askdirectory(title='PNGを保存するフォルダを選択')
        
        if folder:
            count = FontExporter.export_png_batch(self.project, folder)
            messagebox.showinfo('書き出し完了', f'{count}個のPNGを書き出しました:\n{folder}')
    
    def _commit_all_open_editors(self) -> None:
        """開いているエディタを閉じずに全てプロジェクトへ反映（BLOCK9互換）"""
        for ed in list(self._open_editors):
            try:
                if hasattr(ed, 'commit_to_project_without_close'):
                    ed.commit_to_project_without_close()
            except Exception:
                pass
    
    def _confirm_unsaved_changes(self) -> bool:
        """未保存確認（BLOCK9互換）"""
        if not self.project.dirty:
            return True

        ans = messagebox.askyesnocancel(
            '未保存の変更',
            '編集中の変更があります。プロジェクトとして保存しますか?\n\n'
            'はい: 保存して続行\nいいえ: 保存せず続行\nキャンセル: 中止'
        )
        if ans is None:
            return False
        if ans:
            return self._save_project_dialog()
        return True
    
    def _save_project_dialog(self) -> bool:
        """プロジェクト保存ダイアログ（BLOCK9で実装）"""
        # BLOCK9で実装される
        pass  # type: ignore
    
    def _open_project_dialog(self) -> None:
        """プロジェクト読込ダイアログ（BLOCK9で実装）"""
        # BLOCK9で実装される
        pass

    # [ADD] 2025-10-23: 偏旁パレットを開く
    def _open_parts_palette(self) -> None:
        """偏旁パレットウィンドウを開く。プロジェクトにパーツがない場合は警告を表示。"""
        # 既存のパレットがあれば前面に表示
        try:
            if getattr(self, 'parts_palette', None) and self.parts_palette.winfo_exists():
                self.parts_palette.lift()
                return
        except Exception:
            pass
        # パーツデータが存在しなければ警告
        if not getattr(self.project, 'parts', None):
            messagebox.showinfo('情報', '偏旁パーツがありません。偏旁エディタでパーツを作成してください。')
            return
        # パレットを新規作成
        self.parts_palette = PartsPalette(self, self.project, self._insert_part_to_active_editor)

    # [ADD] 2025-10-23: アクティブなエディタにパーツを貼り付け
    def _insert_part_to_active_editor(self, img: Image.Image, scale_hint: float, offset_hint: Tuple[float, float]) -> None:
        """アクティブなグリフエディタに偏旁パーツを貼り付ける。"""
        # エディタが開かれていない場合
        if not getattr(self, '_open_editors', None):
            messagebox.showwarning('警告', 'グリフエディタが開かれていません')
            return
        editor = self._open_editors[-1] if self._open_editors else None
        if not editor:
            messagebox.showwarning('警告', 'グリフエディタが開かれていません')
            return
        try:
            editor.insert_part_image(img, scale_hint, offset_hint)
            editor.lift()
        except Exception as e:
            messagebox.showerror('エラー', f'パーツ貼り付けに失敗しました: {e}')

# [ADD] 2025-10-23: 偏旁パレットウィンドウ定義
class PartsPalette(tk.Toplevel):
    """偏旁パーツ一覧パレット。カテゴリ別に偏旁パーツを表示し、クリックで指定コールバックを呼び出す。"""

    def __init__(self, parent: tk.Widget, project: FontProject, callback: Callable[[Image.Image, float, Tuple[float, float]], None]) -> None:
        super().__init__(parent)
        self.title('部首パレット')
        self.project: FontProject = project
        self.callback = callback
        self.transient(parent)
        # パレットウィンドウサイズ設定
        self.geometry('600x500')
        self.minsize(400, 300)
        # カテゴリ一覧（日本語表示）
        self.categories: List[str] = ['偏', '旁', '冠', '脚', '構', '垂', '繞', 'その他']
        # パーツ分類
        self.parts_by_category: Dict[str, List[str]] = self._group_parts_by_category()
        # UI構築
        self._setup_ui()

    def _setup_ui(self) -> None:
        """パレットUIを構築。左にカテゴリリスト、右にパーツグリッドを表示する。"""
        # 左側カテゴリリスト
        left_frame = tk.Frame(self)
        left_frame.pack(side='left', fill='y', padx=5, pady=5)
        tk.Label(left_frame, text='カテゴリ', anchor='w').pack(anchor='nw', pady=(0, 5))
        self.cat_list = tk.Listbox(left_frame, height=len(self.categories))
        for cat in self.categories:
            self.cat_list.insert('end', cat)
        self.cat_list.pack(fill='both', expand=True)
        self.cat_list.bind('<<ListboxSelect>>', lambda e: self._on_category_selected())
        # 先頭を選択
        if self.categories:
            self.cat_list.selection_set(0)
        # 右側パーツ表示エリア
        right_frame = tk.Frame(self)
        right_frame.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        self.parts_canvas = tk.Canvas(right_frame, bg=Config.COLOR_BG)
        vbar = ttk.Scrollbar(right_frame, orient='vertical', command=self.parts_canvas.yview)
        hbar = ttk.Scrollbar(right_frame, orient='horizontal', command=self.parts_canvas.xview)
        self.parts_canvas.configure(yscrollcommand=vbar.set, xscrollcommand=hbar.set)
        self.parts_canvas.grid(row=0, column=0, sticky='nsew')
        vbar.grid(row=0, column=1, sticky='ns')
        hbar.grid(row=1, column=0, sticky='ew')
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)
        # パーツ一覧を格納するフレーム
        self.parts_container = tk.Frame(self.parts_canvas, bg=Config.COLOR_BG)
        self.parts_canvas.create_window((0, 0), window=self.parts_container, anchor='nw')
        # サムネイルキャッシュと参照リスト
        self._thumb_cache: Dict[str, ImageTk.PhotoImage] = {}
        self._photos: List[ImageTk.PhotoImage] = []
        # 初期カテゴリのパーツを表示
        if self.categories:
            self._populate_parts_grid(self.categories[0])
        # スクロール範囲更新
        self.parts_container.bind('<Configure>', lambda e: self.parts_canvas.configure(scrollregion=self.parts_canvas.bbox('all')))

    def _group_parts_by_category(self) -> Dict[str, List[str]]:
        """プロジェクトのパーツをカテゴリ毎に分類する。メタデータの type または category キーを参照する。"""
        groups: Dict[str, List[str]] = {c: [] for c in self.categories}
        # 英語カテゴリから日本語へのマッピング
        type_map = {
            'hen': '偏',
            'tsukuri': '旁',
            'kanmuri': '冠',
            'ashi': '脚',
            'kamae': '構',
            'tare': '垂',
            'nyou': '繞',
        }
        for name, info in self.project.parts.items():
            meta = info.get('meta', {}) or {}
            t = meta.get('type') or meta.get('category')
            # 英語カテゴリを日本語に変換
            if t in type_map:
                cat = type_map[t]
            elif t in self.categories:
                cat = t
            else:
                cat = 'その他'
            groups.setdefault(cat, []).append(name)
        # 名前順にソート
        for cat_list in groups.values():
            cat_list.sort()
        return groups

    def _on_category_selected(self) -> None:
        """カテゴリリスト選択時に表示を更新。"""
        sel = self.cat_list.curselection()
        if not sel:
            return
        index = sel[0]
        category = self.cat_list.get(index)
        self._populate_parts_grid(category)

    def _populate_parts_grid(self, category: str) -> None:
        """指定カテゴリのパーツをグリッド表示。"""
        # 既存ウィジェット削除
        for child in self.parts_container.winfo_children():
            child.destroy()
        self._photos.clear()
        part_names = self.parts_by_category.get(category, [])
        # グリッド設定
        cols = 4
        size = 80  # サムネイルサイズ
        row = 0
        col = 0
        for name in part_names:
            info = self.project.parts.get(name)
            if not info:
                continue
            img_obj: Image.Image = info.get('image')
            if img_obj is None:
                continue
            # サムネイル生成
            if name not in self._thumb_cache:
                try:
                    # 元画像がグレースケールならアルファ付きRGBAに変換
                    im = img_obj
                    if im.mode != 'L':
                        im = im.convert('L')
                    rgba = Image.new('RGBA', im.size, (255, 255, 255, 0))
                    pix_l = im.load()
                    pix_rgba = rgba.load()
                    for y in range(im.size[1]):
                        for x in range(im.size[0]):
                            val = pix_l[x, y]
                            alpha = 255 - val
                            pix_rgba[x, y] = (0, 0, 0, alpha)
                    thumb = rgba.resize((size, size), Image.Resampling.LANCZOS)
                    ph = ImageTk.PhotoImage(thumb)
                    self._thumb_cache[name] = ph
                except Exception:
                    # フォールバック: 単純にグレースケールを縮小
                    small = img_obj.convert('L').resize((size, size), Image.Resampling.LANCZOS)
                    ph = ImageTk.PhotoImage(small)
                    self._thumb_cache[name] = ph
            ph = self._thumb_cache[name]
            # セルフレーム
            cell = tk.Frame(self.parts_container, bd=1, relief='solid', bg='white')
            cell.grid(row=row, column=col, padx=5, pady=5)
            # 画像ラベル
            img_label = tk.Label(cell, image=ph, bg='white')
            img_label.pack()
            # テキストラベル
            lbl = tk.Label(cell, text=name, font=('Arial', 8), wraplength=size, justify='center')
            lbl.pack(fill='x')
            # クリックイベント
            def on_click(evt, pname=name) -> None:
                self._on_part_clicked(pname)
            cell.bind('<Button-1>', on_click)
            img_label.bind('<Button-1>', on_click)
            lbl.bind('<Button-1>', on_click)
            # 参照保持
            self._photos.append(ph)
            col += 1
            if col >= cols:
                col = 0
                row += 1
        # スクロール範囲更新
        self.parts_canvas.update_idletasks()
        self.parts_canvas.configure(scrollregion=self.parts_canvas.bbox('all'))

    def _on_part_clicked(self, part_name: str) -> None:
        """パーツがクリックされた際の処理。選択した画像とメタデータをコールバックに渡す。"""
        info = self.project.parts.get(part_name)
        if not info:
            messagebox.showerror('エラー', f'パーツ {part_name} が見つかりません')
            return
        img = info.get('image')
        meta = info.get('meta', {}) or {}
        # デフォルト値
        scale_hint: float = 1.0
        offset_hint: Tuple[float, float] = (0.0, 0.0)
        try:
            if 'scale_hint' in meta:
                scale_hint = float(meta.get('scale_hint', 1.0))
            if 'offset_hint' in meta and isinstance(meta['offset_hint'], (list, tuple)) and len(meta['offset_hint']) == 2:
                offset_hint = (float(meta['offset_hint'][0]), float(meta['offset_hint'][1]))
        except Exception:
            pass
        # 画像型チェック
        if not isinstance(img, Image.Image):
            messagebox.showerror('エラー', '画像データが不正です')
            return
        # コールバック呼び出し
        try:
            self.callback(img.copy(), scale_hint, offset_hint)
        except Exception as e:
            messagebox.showerror('エラー', f'パーツ挿入処理でエラーが発生しました: {e}')

# ===== [BLOCK6-END] =====











# ===== [BLOCK7-BEGIN] ファイル入出力ユーティリティ (2025-01-17: PNG書き出しサイズ選択対応) =====

class FileUtils:
    """ファイル入出力ユーティリティ"""
    
    @staticmethod
    def save_glyph_png(bitmap: Image.Image, char_code: int, parent: Optional[tk.Widget] = None) -> None:
        """単一グリフをPNG保存 (2025-01-17: サイズ選択対応)"""  # [ADD]
        if not bitmap:
            if parent:
                messagebox.showwarning('警告', '保存する画像がありません', parent=parent)
            return
        
        # サイズ選択ダイアログ
        if parent:
            dialog = tk.Toplevel(parent)
            dialog.title('PNG書き出しサイズ')
            dialog.geometry('350x200')
            dialog.transient(parent)
            dialog.grab_set()
            
            tk.Label(dialog, text='書き出しサイズを選択:', font=('Arial', 12, 'bold')).pack(pady=10)
            
            size_var = tk.StringVar(value='768')
            
            # サイズオプション
            sizes = [
                ('768 x 768 (標準)', '768'),
                ('1024 x 1024 (高品質)', '1024'),
                ('1536 x 1536 (超高品質)', '1536'),
                ('2048 x 2048 (最高品質)', '2048'),
                ('カスタムサイズ...', 'custom')
            ]
            
            for text, value in sizes:
                tk.Radiobutton(dialog, text=text, variable=size_var, value=value).pack(anchor='w', padx=20)
            
            custom_frame = tk.Frame(dialog)
            custom_frame.pack(pady=5)
            tk.Label(custom_frame, text='カスタム:').pack(side='left')
            custom_entry = tk.Entry(custom_frame, width=10)
            custom_entry.pack(side='left', padx=5)
            custom_entry.insert(0, '768')
            
            export_size = [768]  # デフォルト
            
            def confirm():
                selected = size_var.get()
                if selected == 'custom':
                    try:
                        size = int(custom_entry.get())
                        if size < 256 or size > 4096:
                            messagebox.showwarning('警告', 'サイズは256〜4096の範囲で指定してください')
                            return
                        export_size[0] = size
                    except ValueError:
                        messagebox.showwarning('警告', '正しい数値を入力してください')
                        return
                else:
                    export_size[0] = int(selected)
                dialog.destroy()
            
            tk.Button(dialog, text='OK', command=confirm, width=10).pack(pady=10)
            dialog.wait_window()
        else:
            export_size = [768]  # デフォルト
        
        # ファイル保存ダイアログ
        try:
            char_str = chr(char_code) if char_code < 0x10000 else ''
        except ValueError:
            char_str = ''
        
        default_name = f'U+{char_code:04X}{char_str}.png' if char_str else f'U+{char_code:04X}.png'
        
        path = filedialog.asksaveasfilename(
            title='PNG保存',
            defaultextension='.png',
            initialfile=default_name,
            filetypes=[('PNG Image', '*.png'), ('All Files', '*.*')]
        )
        
        if path:
            # サイズ変更して保存
            target_size = export_size[0]
            if bitmap.size != (target_size, target_size):
                resized = bitmap.resize((target_size, target_size), Image.LANCZOS)
                resized.save(path)
            else:
                bitmap.save(path)
            
            if parent:
                messagebox.showinfo('保存完了', f'サイズ: {target_size}x{target_size}\n保存先: {path}', parent=parent)
    
    @staticmethod
    def save_all_glyphs_png(project: FontProject, parent: Optional[tk.Widget] = None) -> None:
        """全グリフをPNG一括保存 (2025-01-17: サイズ選択対応)"""  # [ADD]
        folder = filedialog.askdirectory(title='保存先フォルダを選択')
        if not folder:
            return
        
        # サイズ選択ダイアログ
        if parent:
            dialog = tk.Toplevel(parent)
            dialog.title('一括書き出しサイズ')
            dialog.geometry('350x200')
            dialog.transient(parent)
            dialog.grab_set()
            
            tk.Label(dialog, text='書き出しサイズを選択:', font=('Arial', 12, 'bold')).pack(pady=10)
            
            size_var = tk.StringVar(value='768')
            
            sizes = [
                ('768 x 768 (標準)', '768'),
                ('1024 x 1024 (高品質)', '1024'),
                ('1536 x 1536 (超高品質)', '1536'),
                ('2048 x 2048 (最高品質)', '2048')
            ]
            
            for text, value in sizes:
                tk.Radiobutton(dialog, text=text, variable=size_var, value=value).pack(anchor='w', padx=20)
            
            export_size = [768]
            
            def confirm():
                export_size[0] = int(size_var.get())
                dialog.destroy()
            
            tk.Button(dialog, text='OK', command=confirm, width=10).pack(pady=10)
            dialog.wait_window()
        else:
            export_size = [768]
        
        # 保存処理
        count = 0
        target_size = export_size[0]
        
        for code, glyph in project.glyphs.items():
            if not glyph.is_empty:
                try:
                    char_str = chr(code) if code < 0x10000 else ''
                except ValueError:
                    char_str = ''
                
                filename = f'U+{code:04X}{char_str}.png' if char_str else f'U+{code:04X}.png'
                path = os.path.join(folder, filename)
                
                # サイズ変更して保存
                if glyph.bitmap.size != (target_size, target_size):
                    resized = glyph.bitmap.resize((target_size, target_size), Image.LANCZOS)
                    resized.save(path)
                else:
                    glyph.bitmap.save(path)
                
                count += 1
        
        if parent:
            messagebox.showinfo('完了', f'{count}個のグリフをPNG保存しました\nサイズ: {target_size}x{target_size}\n保存先: {folder}', parent=parent)
    
    @staticmethod
    def export_font(project: FontProject, parent: Optional[tk.Widget] = None) -> None:
        """TTFファイルとしてエクスポート"""
        path = filedialog.asksaveasfilename(
            title='フォントをエクスポート',
            defaultextension='.ttf',
            filetypes=[('TrueType Font', '*.ttf'), ('All Files', '*.*')]
        )
        
        if not path:
            return
        
        try:
            import fontforge
            
            # 一時ディレクトリ作成
            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                # FontForge用スクリプト作成
                font = fontforge.font()
                font.familyname = "CustomFont"
                font.fullname = "Custom Bitmap Font"
                font.fontname = "CustomFont-Regular"
                
                for code, glyph in project.glyphs.items():
                    if not glyph.is_empty:
                        # グリフを一時ファイルとして保存
                        temp_path = os.path.join(tmpdir, f'{code}.png')
                        glyph.bitmap.save(temp_path)
                        
                        # FontForgeでグリフ作成
                        g = font.createChar(code)
                        g.importOutlines(temp_path)
                
                # TTF保存
                font.generate(path)
                
                if parent:
                    messagebox.showinfo('完了', f'フォントをエクスポートしました:\n{path}', parent=parent)
                    
        except ImportError:
            if parent:
                messagebox.showerror('エラー', 'FontForgeがインストールされていません\nTTFエクスポートにはFontForgeが必要です', parent=parent)
        except Exception as e:
            if parent:
                messagebox.showerror('エラー', f'エクスポートエラー:\n{e}', parent=parent)

# ===== [BLOCK7-END] =====











# ===== [BLOCK8-BEGIN] TTF高品質書き出し (2025-10-11: 型ヒント追加、定数使用、エラーハンドリング改善) =====

class TTFExporter:
    """TTF形式書き出し（アウトライン変換版）"""
    
    @staticmethod
    def check_dependencies() -> Tuple[bool, str]:
        """必要な依存関係をチェック"""
        errors = []
        
        # potraceチェック
        try:
            result = subprocess.run(['potrace', '--version'], capture_output=True, text=True, timeout=5)
            if result.returncode != 0:
                errors.append('potrace がインストールされていません')
        except FileNotFoundError:
            errors.append('potrace がインストールされていません\n\nインストール方法:\n- Mac: brew install potrace\n- Linux: apt-get install potrace\n- Windows: http://potrace.sourceforge.net/')
        except subprocess.TimeoutExpired:
            errors.append('potrace の起動がタイムアウトしました')
        
        # fontToolsチェック
        try:
            import fontTools  # type: ignore  # noqa: F401
        except ImportError:
            errors.append('fontTools がインストールされていません\n\nインストール方法:\npip install fonttools')
        
        if errors:
            return False, '\n\n'.join(errors)
        
        return True, ''
    
    @staticmethod
    def export_ttf(
        project: FontProject, 
        output_path: str, 
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> bool:
        """TTF形式で書き出し（ハイブリッドマージ方式） (2025-10-11: 型ヒント追加、定数使用)"""
        
        # 依存関係チェック
        deps_ok, error_msg = TTFExporter.check_dependencies()
        if not deps_ok:
            messagebox.showerror('依存関係エラー', error_msg)
            return False
        
        try:
            from fontTools.fontBuilder import FontBuilder  # type: ignore
            from fontTools.pens.ttGlyphPen import TTGlyphPen  # type: ignore
            from fontTools.ttLib import TTFont  # type: ignore
            
            # 元のTTFを読み込み（存在する場合）
            original_font: Optional[Any] = None
            if project.original_ttf_path and os.path.exists(project.original_ttf_path):
                try:
                    original_font = TTFont(project.original_ttf_path)
                    print(f'元のTTFを読み込み: {project.original_ttf_path}')
                except Exception as e:
                    print(f'元のTTF読み込み失敗: {e}')
            
            # 編集済みグリフと全グリフを取得
            edited_glyphs = project.get_edited_glyphs()
            with project._lock:  # (2025-10-11: スレッドセーフ化)
                all_valid_glyphs = [(code, glyph) for code, glyph in project.glyphs.items() if not glyph.is_empty]
            
            if not all_valid_glyphs:
                messagebox.showwarning('警告', '書き出すグリフがありません')
                return False
            
            total = len(all_valid_glyphs)
            edited_count = len(edited_glyphs)
            
            print(f'全グリフ: {total}, 編集済み: {edited_count}')
            
            # フォントビルダー作成
            fb = FontBuilder(Config.CANVAS_SIZE, isTTF=True)
            fb.setupGlyphOrder(['.notdef'] + [f'uni{code:04X}' for code, _ in all_valid_glyphs])
            
            # フォント情報設定
            fb.setupPost()
            fb.setupHead(unitsPerEm=Config.CANVAS_SIZE)
            
            # (2025-10-11: 定数使用)
            ascent = int(Config.CANVAS_SIZE * Config.ASCENT_RATIO)
            descent = int(Config.CANVAS_SIZE * Config.DESCENT_RATIO)
            fb.setupHhea(ascent=ascent, descent=descent)
            
            # フォント名設定
            fb.setupNameTable({
                'familyName': 'CustomFont',
                'styleName': 'Regular',
            })
            
            # OS/2テーブル
            fb.setupOS2()
            
            # グリフ処理
            glyphs: Dict[str, Any] = {}
            metrics: Dict[str, Tuple[int, int]] = {}
            
            # .notdef グリフ
            glyphs['.notdef'] = TTFExporter._create_notdef_glyph()
            metrics['.notdef'] = (Config.CANVAS_SIZE, 0)
            
            # 編集済み文字コードのセット
            edited_codes = {code for code, _ in edited_glyphs}
            
            for idx, (code, glyph) in enumerate(all_valid_glyphs):
                glyph_name = f'uni{code:04X}'
                
                # プログレス更新
                if progress_callback:
                    progress_callback(idx + 1, total)
                
                # 編集済みかどうかで処理を分岐
                if code in edited_codes:
                    # 編集済み：ビットマップからアウトライン変換
                    print(f'  変換: {glyph_name} (編集済み)')
                    outline = TTFExporter._bitmap_to_outline(glyph.bitmap)
                    
                    if outline:
                        glyphs[glyph_name] = outline
                        metrics[glyph_name] = (Config.CANVAS_SIZE, 0)
                        
                else:
                    # 未編集：元のTTFから取得を試みる
                    if original_font and glyph_name in original_font['glyf']:
                        print(f'  流用: {glyph_name} (元データ)')
                        # 元のグリフデータを取得
                        original_glyph = original_font['glyf'][glyph_name]
                        glyphs[glyph_name] = original_glyph
                        
                        # メトリクス情報も取得
                        if glyph_name in original_font['hmtx'].metrics:
                            metrics[glyph_name] = original_font['hmtx'].metrics[glyph_name]
                        else:
                            metrics[glyph_name] = (Config.CANVAS_SIZE, 0)
                    else:
                        # 元のTTFにない場合はアウトライン変換
                        print(f'  変換: {glyph_name} (元データなし)')
                        outline = TTFExporter._bitmap_to_outline(glyph.bitmap)
                        
                        if outline:
                            glyphs[glyph_name] = outline
                            metrics[glyph_name] = (Config.CANVAS_SIZE, 0)
            
            # グリフとメトリクスを設定
            fb.setupGlyf(glyphs)
            fb.setupHorizontalMetrics(metrics)
            
            # cmapテーブル（文字コード→グリフマッピング）
            cmap = {code: f'uni{code:04X}' for code, _ in all_valid_glyphs}
            fb.setupCharacterMap(cmap)
            
            # 保存
            fb.save(output_path)
            
            print(f'\nTTF書き出し完了')
            print(f'  全グリフ: {total}')
            print(f'  変換済み: {edited_count}')
            print(f'  流用: {total - edited_count}')
            
            return True
            
        except ImportError as e:
            messagebox.showerror('書き出しエラー', f'必要なライブラリが見つかりません:\n{e}')
            return False
        except Exception as e:
            messagebox.showerror('書き出しエラー', f'TTF書き出し失敗:\n{e}')
            import traceback
            traceback.print_exc()
            return False
    
    @staticmethod
    def _create_notdef_glyph() -> Any:
        """".notdef" グリフを作成（空の四角） (2025-10-11: 定数使用)"""
        from fontTools.pens.ttGlyphPen import TTGlyphPen  # type: ignore
        
        pen = TTGlyphPen(None)
        
        # 外枠 (2025-10-11: 定数使用)
        size = Config.CANVAS_SIZE
        margin = int(size * Config.NOTDEF_MARGIN_RATIO)
        
        pen.moveTo((margin, margin))
        pen.lineTo((size - margin, margin))
        pen.lineTo((size - margin, size - margin))
        pen.lineTo((margin, size - margin))
        pen.closePath()
        
        # 内枠（カウンター） (2025-10-11: 定数使用)
        inner_margin = int(size * Config.NOTDEF_INNER_MARGIN_RATIO)
        pen.moveTo((inner_margin, inner_margin))
        pen.lineTo((inner_margin, size - inner_margin))
        pen.lineTo((size - inner_margin, size - inner_margin))
        pen.lineTo((size - inner_margin, inner_margin))
        pen.closePath()
        
        return pen.glyph()
    
    @staticmethod
    @contextmanager
    def _temp_files() -> Any:
        """一時ファイル管理用コンテキストマネージャ (2025-10-11: 安全な一時ファイル管理)"""
        bitmap_path = None
        svg_path = None
        try:
            with tempfile.NamedTemporaryFile(suffix='.bmp', delete=False) as tmp_bmp:
                bitmap_path = tmp_bmp.name
            with tempfile.NamedTemporaryFile(suffix='.svg', delete=False) as tmp_svg:
                svg_path = tmp_svg.name
            yield bitmap_path, svg_path
        finally:
            if bitmap_path and os.path.exists(bitmap_path):
                try:
                    os.unlink(bitmap_path)
                except OSError:
                    pass
            if svg_path and os.path.exists(svg_path):
                try:
                    os.unlink(svg_path)
                except OSError:
                    pass
    
    @staticmethod
    def _bitmap_to_outline(bitmap: Optional[Image.Image]) -> Optional[Any]:
        """ビットマップをアウトライン（TTFグリフ）に変換 (2025-10-11: 安全な一時ファイル管理)"""
        if bitmap is None:
            return None
        
        from fontTools.pens.ttGlyphPen import TTGlyphPen  # type: ignore
        
        try:
            with TTFExporter._temp_files() as (bitmap_path, svg_path):
                # 白黒反転（potraceは黒を輪郭として認識）
                inverted = Image.eval(bitmap, lambda x: 255 - x)
                inverted.save(bitmap_path)
                
                # potrace実行
                result = subprocess.run(
                    [
                        'potrace',
                        bitmap_path,
                        '-s',  # SVG出力
                        '-o', svg_path,
                        '--turdsize', '2',  # ノイズ除去
                        '--alphamax', '1.0',  # 角の鋭さ
                        '--opttolerance', '0.2'  # 最適化
                    ],
                    capture_output=True,
                    text=True,
                    timeout=30
                )
                
                if result.returncode != 0:
                    return None
                
                # SVGパスを読み込み
                with open(svg_path, 'r') as f:
                    svg_content = f.read()
                
                # パスデータを抽出
                path_data = TTFExporter._extract_svg_paths(svg_content)
                
                if not path_data:
                    return None
                
                # TTGlyphPenで描画
                pen = TTGlyphPen(None)
                
                # SVGパスをTTFグリフに変換
                TTFExporter._svg_path_to_ttglyph(path_data, pen, Config.CANVAS_SIZE)
                
                return pen.glyph()
            
        except subprocess.TimeoutExpired:
            print('potrace タイムアウト')
            return None
        except Exception as e:
            print(f'アウトライン変換エラー: {e}')
            return None
    
    @staticmethod
    def _extract_svg_paths(svg_content: str) -> List[str]:
        """SVGからパスデータを抽出"""
        paths = []
        
        # <path d="..." /> を抽出
        path_pattern = r'<path[^>]*d="([^"]*)"'
        matches = re.findall(path_pattern, svg_content)
        
        for match in matches:
            paths.append(match)
        
        return paths
    
    @staticmethod
    def _svg_path_to_ttglyph(path_data_list: List[str], pen: Any, size: int) -> None:
        """SVGパスデータをTTGlyphPenに描画"""
        
        for path_data in path_data_list:
            # SVGパスをパース
            commands = TTFExporter._parse_svg_path(path_data)
            
            for cmd, args in commands:
                if cmd == 'M':  # MoveTo
                    x, y = args[0], args[1]
                    pen.moveTo((x, size - y))  # Y軸反転
                    
                elif cmd == 'L':  # LineTo
                    x, y = args[0], args[1]
                    pen.lineTo((x, size - y))
                    
                elif cmd == 'C':  # CurveTo（3次ベジエ）
                    x1, y1, x2, y2, x, y = args
                    pen.curveTo(
                        (x1, size - y1),
                        (x2, size - y2),
                        (x, size - y)
                    )
                    
                elif cmd == 'Z':  # ClosePath
                    pen.closePath()
    
    @staticmethod
    def _parse_svg_path(path_data: str) -> List[Tuple[str, List[float]]]:
        """SVGパスデータをパース"""
        commands = []
        
        # 簡易パーサー（M, L, C, Zに対応）
        tokens = re.findall(r'[MLCZ]|[-+]?[0-9]*\.?[0-9]+', path_data)
        
        i = 0
        while i < len(tokens):
            token = tokens[i]
            
            if token == 'M':
                x, y = float(tokens[i+1]), float(tokens[i+2])
                commands.append(('M', [x, y]))
                i += 3
                
            elif token == 'L':
                x, y = float(tokens[i+1]), float(tokens[i+2])
                commands.append(('L', [x, y]))
                i += 3
                
            elif token == 'C':
                args = [float(tokens[i+j]) for j in range(1, 7)]
                commands.append(('C', args))
                i += 7
                
            elif token == 'Z' or token == 'z':
                commands.append(('Z', []))
                i += 1
                
            else:
                i += 1
        
        return commands

# ===== [BLOCK8-END] =====











# ===== [BLOCK9-BEGIN] バックグラウンド読み込み (2025-10-11: 型ヒント追加、スレッド安全性改善) =====

class BackgroundLoader:
    """バックグラウンドでフォントを読み込むクラス"""
    
    def __init__(self, project: FontProject, status_callback: Callable[[Dict[str, Any]], None]) -> None:
        self.project: FontProject = project
        self.status_callback: Callable[[Dict[str, Any]], None] = status_callback  # ステータス更新用コールバック
        self.thread: Optional[threading.Thread] = None
        self.is_loading: bool = False
        self.stop_flag: bool = False
        self.result_queue: queue.Queue = queue.Queue()  # 結果受け渡し用
    
    def start_background_load(self, font_path: str, initial_range: Tuple[int, int], font_index: int = 0) -> None:
        """バックグラウンド読み込み開始 (2025-11-10: TTC対応)"""
        if self.is_loading:
            return  # 既に読み込み中

        self.stop_flag = False
        self.is_loading = True

        # スレッド開始
        self.thread = threading.Thread(
            target=self._background_load_worker,
            args=(font_path, initial_range, font_index),
            daemon=True
        )
        self.thread.start()

    def stop(self) -> None:
        """読み込み停止"""
        self.stop_flag = True
        self.is_loading = False

    def _background_load_worker(self, font_path: str, initial_range: Tuple[int, int], font_index: int = 0) -> None:
        """バックグラウンド読み込みワーカー (2025-10-11: スレッド安全性改善、定数使用、2025-11-10: TTC対応)"""
        try:
            # フォント読み込み (2025-10-11: 定数使用、2025-11-10: TTC対応)
            pil_font = ImageFont.truetype(font_path, size=Config.FONT_RENDER_SIZE, index=font_index)
            
            # 全範囲を取得（初期範囲以外）
            all_ranges = list(Config.CHAR_RANGES.values())
            
            # 初期範囲を先頭に移動（既に読み込み済みなのでスキップ）
            if initial_range in all_ranges:
                all_ranges.remove(initial_range)
            
            total_ranges = len(all_ranges)
            
            for idx, range_tuple in enumerate(all_ranges):
                if self.stop_flag:
                    break  # 停止要求
                
                # 既に読み込み済みならスキップ
                if self.project.is_range_loaded(range_tuple):
                    continue
                
                # ステータス更新
                range_name = self._get_range_name(range_tuple)
                self.result_queue.put({
                    'type': 'status',
                    'message': f'バックグラウンド読み込み中: {range_name} ({idx+1}/{total_ranges})'
                })
                
                # 範囲の文字コード取得
                start, end = range_tuple
                char_codes = list(range(start, end + 1))
                
                # 読み込み実行
                for code in char_codes:
                    if self.stop_flag:
                        break
                    
                    # 既存グリフはスキップ (2025-10-11: スレッドセーフ化)
                    with self.project._lock:
                        if code in self.project.glyphs and not self.project.glyphs[code].is_empty:
                            continue
                    
                    try:
                        char = chr(code)
                        bitmap = FontRenderer._render_char(char, pil_font)
                        
                        if bitmap:
                            self.project.set_glyph(code, bitmap, is_edited=False)  # スレッドセーフなメソッドを使用
                        else:
                            with self.project._lock:
                                if code not in self.project.glyphs:
                                    self.project.glyphs[code] = GlyphData(code, None, False)
                                
                    except (ValueError, OSError):
                        with self.project._lock:
                            if code not in self.project.glyphs:
                                self.project.glyphs[code] = GlyphData(code, None, False)
                
                # 範囲を読み込み済みとしてマーク
                self.project.mark_range_loaded(range_tuple)
            
            # 完了通知
            if not self.stop_flag:
                self.result_queue.put({
                    'type': 'complete',
                    'message': 'バックグラウンド読み込み完了'
                })
            
        except Exception as e:
            self.result_queue.put({
                'type': 'error',
                'message': f'バックグラウンド読み込みエラー: {e}'
            })
        
        finally:
            self.is_loading = False
    
    def _get_range_name(self, range_tuple: Tuple[int, int]) -> str:
        """範囲名を取得"""
        for name, r in Config.CHAR_RANGES.items():
            if r == range_tuple:
                return name
        return f'{range_tuple[0]:04X}-{range_tuple[1]:04X}'
    
    def check_results(self) -> None:
        """結果キューをチェック（メインスレッドから呼び出す）"""
        try:
            while True:
                result = self.result_queue.get_nowait()
                self.status_callback(result)
        except queue.Empty:
            pass

# ===== [BLOCK9-END] =====














# ===== [BLOCK10-BEGIN] 補助機能（安全保存＋未保存確認＋単一ファイルI/O＋GlyphEditor互換パッチ） (2025-10-11: 型ヒント追加、統合版) =====

# --- 未保存確認（Yes=保存 / No=保存せず続行 / Cancel=中止） ---
def _confirm_unsaved_changes(self: FontEditorApp) -> bool:
    """未保存の変更確認"""
    if not self.project.dirty:
        return True

    ans = messagebox.askyesnocancel(
        '未保存の変更',
        '編集中の変更があります。プロジェクトとして保存しますか？\n\n'
        'はい: 保存して続行\nいいえ: 保存せず続行\nキャンセル: 中止'
    )
    if ans is None:
        return False
    if ans:
        return self._save_project_dialog()
    return True

FontEditorApp._confirm_unsaved_changes = _confirm_unsaved_changes  # type: ignore


# --- .fproj 保存：辞書スナップショットで安全保存 ---
def _save_project_dialog_impl(self: FontEditorApp) -> bool:
    """プロジェクト保存ダイアログ (2025-10-11: 型ヒント追加)"""
    try:
        if hasattr(self, '_stop_bg_loading'):
            self._stop_bg_loading()
    except Exception:
        pass
    try:
        if hasattr(self, '_commit_all_open_editors'):
            self._commit_all_open_editors()
    except Exception:
        pass

    path = filedialog.asksaveasfilename(
        title='プロジェクトを保存',
        defaultextension='.fproj',
        filetypes=[('Font Project', '*.fproj'), ('Single File Project', '*.fprojz')]
    )
    if not path:
        return False

    # --- 単一ファイル保存 ---
    if path.endswith('.fprojz'):
        return _export_project_singlefile_impl(self, path)

    # --- 通常フォルダ保存 ---
    if not path.endswith('.fproj'):
        path += '.fproj'

    # プログレスウィンドウ作成 (2025-11-10)
    progress_win = tk.Toplevel(self)
    progress_win.title('保存中...')
    progress_win.geometry('500x150')
    progress_win.transient(self)
    progress_win.grab_set()

    tk.Label(
        progress_win,
        text='プロジェクトを保存しています...',
        font=('Arial', 12)
    ).pack(pady=10)

    progress_var = tk.IntVar(value=0)
    progress_bar = ttk.Progressbar(
        progress_win,
        maximum=100,
        variable=progress_var,
        length=400
    )
    progress_bar.pack(pady=10)

    progress_label = tk.Label(
        progress_win,
        text='準備中...',
        font=('Arial', 10)
    )
    progress_label.pack()

    # プログレスバー更新用コールバック
    def progress_callback(current: int, total: int, message: str) -> None:
        """プログレス更新"""
        if total > 0:
            progress_var.set(int((current / total) * 100))
            progress_bar.config(maximum=total, value=current)
        progress_label.config(text=message)
        progress_win.update()

    try:
        with self.project._lock:  # (2025-10-11: スレッドセーフ化)
            orig_glyphs = self.project.glyphs
            snapshot = dict(orig_glyphs)
            self.project.glyphs = snapshot

        try:
            self.project.save_project(path, progress_callback)
            self.project.dirty = False
            progress_win.destroy()
            messagebox.showinfo('保存完了', f'プロジェクトを保存しました:\n{path}')
            return True
        except OSError as e:
            progress_win.destroy()
            messagebox.showerror('保存エラー', f'保存に失敗しました:\n{e}')
            return False
        except Exception as e:
            progress_win.destroy()
            messagebox.showerror('保存エラー', f'予期しないエラー:\n{e}')
            return False
        finally:
            with self.project._lock:
                self.project.glyphs = orig_glyphs
    except Exception as e:
        if progress_win.winfo_exists():
            progress_win.destroy()
        messagebox.showerror('保存エラー', f'保存処理中にエラーが発生しました:\n{e}')
        return False


# --- .fproj 読込 ---
def _open_project_dialog_impl(self: FontEditorApp) -> None:
    """プロジェクト読込ダイアログ"""
    if not self._confirm_unsaved_changes():
        return
    folder = filedialog.askdirectory(title='プロジェクトを開く（*.fproj フォルダを選択）')
    if not folder:
        return
    try:
        self.project.load_project(folder)
        self.project.dirty = False
        if hasattr(self, 'grid_view'):
            self.grid_view.refresh()
        if hasattr(self, '_update_status'):
            self._update_status()
        messagebox.showinfo('読込完了', f'プロジェクトを読み込みました:\n{folder}')
    except OSError as e:
        messagebox.showerror('読込エラー', f'プロジェクト読込に失敗しました:\n{e}')
    except Exception as e:
        messagebox.showerror('読込エラー', f'予期しないエラー:\n{e}')


# --- .fprojz へ単一ファイル書出し ---
def _export_project_singlefile_impl(self: FontEditorApp, dest: Optional[str] = None) -> bool:
    """単一ファイル書出し (2025-10-11: 型ヒント追加、安全な一時ファイル管理)"""
    try:
        if hasattr(self, '_stop_bg_loading'):
            self._stop_bg_loading()
    except Exception:
        pass
    try:
        if hasattr(self, '_commit_all_open_editors'):
            self._commit_all_open_editors()
    except Exception:
        pass

    if not dest:
        dest = filedialog.asksaveasfilename(
            title='単一ファイルに書き出し',
            defaultextension='.fprojz',
            filetypes=[('Font Project (Single File)', '*.fprojz')]
        )
    if not dest:
        return False
    if not dest.endswith('.fprojz'):
        dest += '.fprojz'

    tmpdir = tempfile.mkdtemp(prefix='fproj_tmp_')
    try:
        with self.project._lock:  # (2025-10-11: スレッドセーフ化)
            orig_glyphs = self.project.glyphs
            snapshot = dict(orig_glyphs)
            self.project.glyphs = snapshot

        temp_folder = os.path.join(tmpdir, 'project.fproj')
        
        try:
            self.project.save_project(temp_folder)

            with zipfile.ZipFile(dest, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                for root, dirs, files in os.walk(temp_folder):
                    for name in files:
                        abspath = os.path.join(root, name)
                        arcname = os.path.relpath(abspath, start=tmpdir)
                        zf.write(abspath, arcname)

            self.project.dirty = False
            messagebox.showinfo('書き出し完了', f'単一ファイルへ書き出しました:\n{dest}')
            return True
        except OSError as e:
            messagebox.showerror('書き出しエラー', f'単一ファイル書き出しに失敗しました:\n{e}')
            return False
        except Exception as e:
            messagebox.showerror('書き出しエラー', f'予期しないエラー:\n{e}')
            return False
        finally:
            with self.project._lock:
                self.project.glyphs = orig_glyphs
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- .fprojz 読込 ---
def _open_project_singlefile_impl(self: FontEditorApp) -> None:
    """単一ファイルプロジェクト読込"""
    if not self._confirm_unsaved_changes():
        return
    src = filedialog.askopenfilename(
        title='単一ファイルのプロジェクトを開く',
        filetypes=[('Font Project (Single File)', '*.fprojz'), ('All Files', '*.*')]
    )
    if not src:
        return
    tmpdir = tempfile.mkdtemp(prefix='fproj_open_')
    try:
        with zipfile.ZipFile(src, 'r') as zf:
            zf.extractall(tmpdir)
        target = None
        for entry in os.listdir(tmpdir):
            p = os.path.join(tmpdir, entry)
            if os.path.isdir(p) and entry.endswith('.fproj'):
                target = p
                break
        if not target:
            raise RuntimeError('アーカイブ内に .fproj フォルダが見つかりません')

        self.project.load_project(target)
        self.project.dirty = False
        if hasattr(self, 'grid_view'):
            self.grid_view.refresh()
        if hasattr(self, '_update_status'):
            self._update_status()
        messagebox.showinfo('読込完了', f'単一ファイルのプロジェクトを読み込みました:\n{src}')
    except OSError as e:
        messagebox.showerror('読込エラー', f'単一ファイル読込に失敗しました:\n{e}')
    except Exception as e:
        messagebox.showerror('読込エラー', f'予期しないエラー:\n{e}')
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- FontEditorApp へバインド ---
def _wrap(f: Callable) -> Callable:
    """関数ラッパー"""
    f.__wrapped__ = f  # type: ignore
    return f

FontEditorApp._save_project_dialog = _wrap(_save_project_dialog_impl)  # type: ignore
FontEditorApp._open_project_dialog = _wrap(_open_project_dialog_impl)  # type: ignore
FontEditorApp._export_project_singlefile = _wrap(_export_project_singlefile_impl)  # type: ignore
FontEditorApp._open_project_singlefile = _wrap(_open_project_singlefile_impl)  # type: ignore


# =========================
# GlyphEditor 互換パッチ群
# =========================
# UI/レイアウトは一切変更せず、足りないAPIだけ後付け
#  - ⌘S/Ctrl+Sで「閉じずにプロジェクトへ反映→保存ダイアログ」
#  - _on_copy/_on_cut/_on_paste/_draw_rect/_draw_ellipse/_start_selection を補完

def _ge_commit(self: GlyphEditor) -> None:
    """エディタ内容をプロジェクトへ反映（BLOCK9互換）"""
    self.project.glyphs[self.char_code] = GlyphData(self.char_code, self.edit_bitmap.copy(), is_edited=True)
    self.project.dirty = True
    if callable(getattr(self, 'on_commit', None)):
        self.on_commit(self.char_code)  # type: ignore

def _ge_save_from_editor(self: GlyphEditor, event: Optional[tk.Event] = None) -> None:
    """⌘S/Ctrl+S: 保存（BLOCK9互換）"""
    if not hasattr(self, 'commit_to_project_without_close'):
        self.commit_to_project_without_close = MethodType(_ge_commit, self)  # type: ignore
    self.commit_to_project_without_close()  # type: ignore
    if hasattr(self.master, '_save_project_dialog'):
        self.master._save_project_dialog()  # type: ignore

def _ge_on_copy(self: GlyphEditor) -> None:
    """コピー（BLOCK9互換）"""
    if hasattr(self, '_copy'):
        self._copy()

def _ge_on_cut(self: GlyphEditor) -> None:
    """切り取り（BLOCK9互換）"""
    if hasattr(self, '_cut'):
        self._cut()

def _ge_on_paste(self: GlyphEditor) -> None:
    """貼り付け（BLOCK9互換）"""
    if hasattr(self, '_paste'):
        self._paste()

def _ge_start_selection(self: GlyphEditor, x0: int, y0: int, x1: int, y1: int) -> None:
    """選択開始（BLOCK9互換）"""
    try:
        x0 = max(0, min(int(x0), self.edit_bitmap.width - 1))
        y0 = max(0, min(int(y0), self.edit_bitmap.height - 1))
        x1 = max(0, min(int(x1), self.edit_bitmap.width))
        y1 = max(0, min(int(y1), self.edit_bitmap.height))
        self.selection_start = (min(x0, x1), min(y0, y1))
        self.selection_end = (max(x0, x1), max(y0, y1))
        self.selected_image = self.edit_bitmap.crop((*self.selection_start, *self.selection_end)).copy()
        if hasattr(self, '_update_preview'):
            self._update_preview()
    except Exception:
        pass

def _ge_draw_rect(self: GlyphEditor, x0: int, y0: int, x1: int, y1: int, width: int = 1) -> None:
    """矩形描画（BLOCK9互換）"""
    dr = ImageDraw.Draw(self.edit_bitmap)
    dr.rectangle((x0, y0, x1, y1), outline=0, width=max(1, int(width)))
    if hasattr(self, '_update_preview'):
        self._update_preview()

def _ge_draw_ellipse(self: GlyphEditor, x0: int, y0: int, x1: int, y1: int, width: int = 1) -> None:
    """楕円描画（BLOCK9互換）"""
    dr = ImageDraw.Draw(self.edit_bitmap)
    dr.ellipse((x0, y0, x1, y1), outline=0, width=max(1, int(width)))
    if hasattr(self, '_update_preview'):
        self._update_preview()

# __init__ をラップしてショートカット/メソッドを付与
try:
    _GE_init_orig = GlyphEditor.__init__
    def _GE_init_patched(self: GlyphEditor, *a: Any, **k: Any) -> None:
        _GE_init_orig(self, *a, **k)  # type: ignore
        if not hasattr(self, 'commit_to_project_without_close'):
            self.commit_to_project_without_close = MethodType(_ge_commit, self)  # type: ignore
        if not hasattr(self, '_save_from_editor'):
            self._save_from_editor = MethodType(_ge_save_from_editor, self)  # type: ignore
        for name, func in (
            ('_on_copy', _ge_on_copy),
            ('_on_cut', _ge_on_cut),
            ('_on_paste', _ge_on_paste),
            ('_start_selection', _ge_start_selection),
            ('_draw_rect', _ge_draw_rect),
            ('_draw_ellipse', _ge_draw_ellipse),
        ):
            if not hasattr(self, name):
                setattr(self, name, MethodType(func, self))  # type: ignore
        try:
            self.unbind_all('<Command-s>')
            self.unbind_all('<Control-s>')
        except Exception:
            pass
        try:
            self.bind_all('<Command-s>', self._save_from_editor)  # type: ignore
            self.bind_all('<Control-s>', self._save_from_editor)  # type: ignore
        except Exception:
            pass
    GlyphEditor.__init__ = _GE_init_patched  # type: ignore
except Exception:
    pass

# ===== [BLOCK10-END] =====













# ===== [BLOCK11-BEGIN] グリフフィルタダイアログ (2025-10-11: 型ヒント追加) =====

class GlyphFilterDialog(tk.Toplevel):
    """グリフフィルタ選択ダイアログ"""
    
    def __init__(self, parent: tk.Widget, project: FontProject, current_filter: str) -> None:
        super().__init__(parent)
        self.project: FontProject = project
        self.result: str = current_filter
        
        self.title('グリフフィルタ')
        self.geometry('400x350')
        self.transient(parent)
        self.grab_set()
        
        # 説明ラベル
        tk.Label(
            self,
            text='表示するグリフの種類を選択してください',
            font=('Arial', 12, 'bold'),
            pady=10
        ).pack()
        
        # フィルタオプション
        self.filter_var = tk.StringVar(value=current_filter)
        
        filter_options = [
            ('all', 'すべて表示'),
            ('edited', '編集済みのみ'),
            ('unedited', '未編集のみ'),
            ('defined', '定義済みのみ'),
            ('empty', '空白のみ')
        ]
        
        for value, label in filter_options:
            tk.Radiobutton(
                self,
                text=label,
                variable=self.filter_var,
                value=value,
                font=('Arial', 11),
                anchor='w'
            ).pack(fill='x', padx=20, pady=5)
        
        # 統計情報表示 (2025-10-08)
        stats_frame = tk.Frame(self, relief='sunken', borderwidth=1)
        stats_frame.pack(fill='x', padx=20, pady=20)
        
        tk.Label(
            stats_frame,
            text='現在の範囲の統計:',
            font=('Arial', 10, 'bold')
        ).pack(pady=5)
        
        # 統計計算 (2025-10-11: スレッドセーフ化)
        char_codes = project.get_char_codes()
        total = len(char_codes)
        
        with project._lock:
            edited = sum(1 for code in char_codes
                        if code in project.glyphs and not project.glyphs[code].is_empty and project.glyphs[code].is_edited)
            unedited = sum(1 for code in char_codes
                          if code in project.glyphs and not project.glyphs[code].is_empty and not project.glyphs[code].is_edited)
            empty = sum(1 for code in char_codes
                       if code not in project.glyphs or project.glyphs[code].is_empty)
        
        stats_text = f'全体: {total}  |  編集済み: {edited}  |  未編集: {unedited}  |  空白: {empty}'
        
        tk.Label(
            stats_frame,
            text=stats_text,
            font=('Arial', 9)
        ).pack(pady=5)
        
        # ボタンフレーム
        button_frame = tk.Frame(self)
        button_frame.pack(pady=10)
        
        tk.Button(
            button_frame,
            text='適用',
            command=self._apply,
            width=10
        ).pack(side='left', padx=5)
        
        tk.Button(
            button_frame,
            text='キャンセル',
            command=self.destroy,
            width=10
        ).pack(side='left', padx=5)
        
        # Enterキーで適用
        self.bind('<Return>', lambda e: self._apply())
    
    def _apply(self) -> None:
        """適用ボタン押下"""
        self.result = self.filter_var.get()
        self.destroy()

# ===== [BLOCK11-END] =====











# ===== [BLOCK12-BEGIN] テキストプレビューダイアログ (2025-10-11: 型ヒント追加、PhotoImage参照保持改善) =====

class TextPreviewDialog(tk.Toplevel):
    """テキストプレビューダイアログ"""
    
    def __init__(self, parent: tk.Widget, project: FontProject) -> None:
        super().__init__(parent)
        self.project: FontProject = project
        
        self.title('テキストプレビュー')
        self.geometry('800x600')
        self.transient(parent)
        
        # PhotoImage参照保持 (2025-10-11: GC対策)
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        
        self._setup_ui()
    
    def _setup_ui(self) -> None:
        """UI構築"""
        # ツールバー
        toolbar = tk.Frame(self)
        toolbar.pack(side='top', fill='x', padx=5, pady=5)
        
        tk.Label(toolbar, text='サイズ:').pack(side='left', padx=5)
        
        self.size_var = tk.IntVar(value=48)
        size_scale = tk.Scale(
            toolbar,
            from_=12,
            to=200,
            orient='horizontal',
            variable=self.size_var,
            command=lambda v: self._update_preview()
        )
        size_scale.pack(side='left', padx=5)
        
        tk.Button(
            toolbar,
            text='PNG保存',
            command=self._save_png
        ).pack(side='left', padx=5)
        
        # 入力エリア
        input_frame = tk.Frame(self)
        input_frame.pack(fill='x', padx=5, pady=5)
        
        tk.Label(input_frame, text='プレビューテキスト:').pack(anchor='w')
        
        self.text_entry = tk.Entry(input_frame, font=('Arial', 12))
        self.text_entry.pack(fill='x', pady=5)
        self.text_entry.insert(0, 'ABCDEFGあいうえお')
        self.text_entry.bind('<KeyRelease>', lambda e: self._update_preview())
        
        # プレビューキャンバス
        preview_frame = tk.Frame(self, relief='sunken', borderwidth=1)
        preview_frame.pack(fill='both', expand=True, padx=5, pady=5)
        
        # スクロール対応
        self.preview_canvas = tk.Canvas(preview_frame, bg='white')
        v_scroll = ttk.Scrollbar(preview_frame, orient='vertical', command=self.preview_canvas.yview)
        h_scroll = ttk.Scrollbar(preview_frame, orient='horizontal', command=self.preview_canvas.xview)
        
        self.preview_canvas.configure(
            yscrollcommand=v_scroll.set,
            xscrollcommand=h_scroll.set
        )
        
        self.preview_canvas.grid(row=0, column=0, sticky='nsew')
        v_scroll.grid(row=0, column=1, sticky='ns')
        h_scroll.grid(row=1, column=0, sticky='ew')
        
        preview_frame.grid_rowconfigure(0, weight=1)
        preview_frame.grid_columnconfigure(0, weight=1)
        
        # 初期プレビュー
        self._update_preview()
    
    def _update_preview(self) -> None:
        """プレビュー更新 (2025-10-11: スレッドセーフ化、PhotoImage参照保持改善)"""
        text = self.text_entry.get()
        size = self.size_var.get()
        
        if not text:
            return
        
        # 各文字のビットマップを取得して並べる
        char_images: List[Image.Image] = []
        
        with self.project._lock:  # (2025-10-11: スレッドセーフ化)
            for char in text:
                try:
                    code = ord(char)
                    glyph = self.project.glyphs.get(code)
                    
                    if glyph and not glyph.is_empty and glyph.bitmap:
                        # グリフをリサイズ
                        resized = glyph.bitmap.resize((size, size), Image.Resampling.LANCZOS)
                        char_images.append(resized)
                    else:
                        # 空白グリフは空白スペース
                        blank = Image.new('L', (size, size), 255)
                        char_images.append(blank)
                        
                except (ValueError, KeyError):
                    # エラー時は空白
                    blank = Image.new('L', (size, size), 255)
                    char_images.append(blank)
        
        if not char_images:
            return
        
        # 横に並べて1枚の画像を作成
        total_width = sum(img.width for img in char_images)
        max_height = max(img.height for img in char_images)
        
        combined = Image.new('L', (total_width, max_height), 255)
        
        x_offset = 0
        for img in char_images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
        
        # キャンバスに表示 (2025-10-11: PhotoImage参照保持改善)
        self.preview_photo = ImageTk.PhotoImage(combined)
        self.preview_canvas.delete('all')
        self.preview_canvas.create_image(0, 0, anchor='nw', image=self.preview_photo)
        
        # スクロール領域更新
        self.preview_canvas.configure(scrollregion=(0, 0, total_width, max_height))
    
    def _save_png(self) -> None:
        """プレビュー画像をPNG保存 (2025-10-11: スレッドセーフ化)"""
        text = self.text_entry.get()
        size = self.size_var.get()
        
        if not text:
            messagebox.showwarning('警告', 'テキストを入力してください')
            return
        
        path = filedialog.asksaveasfilename(
            title='プレビュー画像を保存',
            defaultextension='.png',
            filetypes=[('PNG Image', '*.png'), ('All Files', '*.*')]
        )
        
        if not path:
            return
        
        # 画像を再生成して保存
        char_images: List[Image.Image] = []
        
        with self.project._lock:  # (2025-10-11: スレッドセーフ化)
            for char in text:
                try:
                    code = ord(char)
                    glyph = self.project.glyphs.get(code)
                    
                    if glyph and not glyph.is_empty and glyph.bitmap:
                        resized = glyph.bitmap.resize((size, size), Image.Resampling.LANCZOS)
                        char_images.append(resized)
                    else:
                        blank = Image.new('L', (size, size), 255)
                        char_images.append(blank)
                except (ValueError, KeyError):
                    blank = Image.new('L', (size, size), 255)
                    char_images.append(blank)
        
        if not char_images:
            messagebox.showwarning('警告', '画像を生成できませんでした')
            return
        
        total_width = sum(img.width for img in char_images)
        max_height = max(img.height for img in char_images)
        
        combined = Image.new('L', (total_width, max_height), 255)
        
        x_offset = 0
        for img in char_images:
            combined.paste(img, (x_offset, 0))
            x_offset += img.width
        
        # 透過PNG変換
        rgba_img = Image.new('RGBA', combined.size, (255, 255, 255, 0))
        pixels_gray = combined.load()
        pixels_rgba = rgba_img.load()
        
        for y in range(combined.size[1]):
            for x in range(combined.size[0]):
                gray_value = pixels_gray[x, y]
                alpha = 255 - gray_value
                pixels_rgba[x, y] = (0, 0, 0, alpha)
        
        try:
            rgba_img.save(path, 'PNG')
            messagebox.showinfo('保存完了', f'保存しました:\n{path}')
        except OSError as e:
            messagebox.showerror('保存エラー', f'保存に失敗しました:\n{e}')

# ===== [BLOCK12-END] =====









# ===== [MAIN] メイン実行 =====

if __name__ == '__main__':
    app = FontEditorApp()
    app.mainloop()
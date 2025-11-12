# AI フォント生成システム設計書

## 1. プロジェクト概要

### 目的
機械学習を用いて、既存フォントから学習し、新しいスタイルのフォントを自動生成するシステムの構築

### 主要機能
1. 複数フォントからの学習
2. 文字の骨格（構造）分析
3. 文字の個性（スタイル）分析
4. スタイル指定による新規文字生成
5. インタラクティブな修正機能
6. 完全なフォントファイル出力

---

## 2. システムアーキテクチャ

```
┌─────────────────────────────────────────────────────────┐
│              フロントエンド (GUI/CLI)                    │
│  - フォント読み込み                                      │
│  - 学習パラメータ設定                                    │
│  - 生成プレビュー                                        │
│  - インタラクティブ編集                                  │
└────────────────┬────────────────────────────────────────┘
                 │
┌────────────────▼────────────────────────────────────────┐
│              コアエンジン                                │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │ フォントパーサー  │  │ データ前処理      │            │
│  │ - TTF/OTF読込    │  │ - 画像変換       │            │
│  │ - グリフ抽出     │  │ - 正規化         │            │
│  │ - ベクトル化     │  │ - 拡張           │            │
│  └──────────────────┘  └──────────────────┘            │
│                                                          │
│  ┌──────────────────────────────────────────┐          │
│  │         AI モデル (PyTorch/TensorFlow)    │          │
│  │                                            │          │
│  │  ┌─────────────────┐  ┌────────────────┐ │          │
│  │  │ 骨格抽出モデル   │  │ スタイル抽出   │ │          │
│  │  │ (Encoder)       │  │ (Style Encoder)│ │          │
│  │  └─────────────────┘  └────────────────┘ │          │
│  │                                            │          │
│  │  ┌─────────────────────────────────────┐  │          │
│  │  │      文字生成モデル                  │  │          │
│  │  │      (Decoder/Generator)            │  │          │
│  │  │  - 条件付き生成                      │  │          │
│  │  │  - スタイル転送                      │  │          │
│  │  └─────────────────────────────────────┘  │          │
│  └──────────────────────────────────────────┘          │
│                                                          │
│  ┌──────────────────┐  ┌──────────────────┐            │
│  │ 後処理モジュール  │  │ フォント生成     │            │
│  │ - スムージング   │  │ - TTF/OTF出力    │            │
│  │ - ベクトル変換   │  │ - メトリクス設定 │            │
│  └──────────────────┘  └──────────────────┘            │
└─────────────────────────────────────────────────────────┘
```

---

## 3. 技術スタック

### 3.1 フロントエンド
- **GUI**: Tkinter / PyQt5
- **可視化**: Matplotlib, Pillow

### 3.2 バックエンド
- **言語**: Python 3.10+
- **フォント処理**:
  - fontTools (TTF/OTF読み書き)
  - freetype-py (グリフレンダリング)
  - Pillow (画像処理)

### 3.3 機械学習
- **フレームワーク**: PyTorch 2.0+
- **モデルアーキテクチャ**:
  - **オプション1**: VAE (Variational Autoencoder)
    - 骨格とスタイルの分離学習に適している
    - 潜在空間での補間が可能

  - **オプション2**: GAN (Generative Adversarial Network)
    - より高品質な生成が可能
    - StyleGAN2/3ベースの実装

  - **オプション3**: Diffusion Model
    - 最新の生成技術
    - 安定した学習

  - **推奨**: VAE + Style Transfer の組み合わせ
    - 実装が比較的容易
    - 制御性が高い

### 3.4 事前学習モデル (オプション)
- **GPT-4 Vision / GPT-4o**:
  - 文字の特徴分析
  - スタイル記述の生成
- **CLIP**:
  - テキスト→画像のマッチング
  - スタイル類似度計算

---

## 4. データフロー

### 4.1 学習フェーズ

```
入力フォント (TTF/OTF)
    ↓
グリフ抽出 (各文字をSVG/画像化)
    ↓
前処理 (正規化、拡張、ノイズ除去)
    ↓
┌─────────────────────────────┐
│  同時学習                    │
│  ┌──────────┐  ┌──────────┐ │
│  │骨格抽出   │  │スタイル   │ │
│  │Encoder   │  │Encoder   │ │
│  └──────────┘  └──────────┘ │
│         ↓           ↓        │
│  ┌──────────────────────┐   │
│  │  潜在空間表現         │   │
│  │  z = [z_content,     │   │
│  │       z_style]       │   │
│  └──────────────────────┘   │
│         ↓                    │
│  ┌──────────────────────┐   │
│  │  Decoder/Generator   │   │
│  └──────────────────────┘   │
└─────────────────────────────┘
    ↓
再構成画像 + 損失計算
    ↓
モデル更新
```

### 4.2 生成フェーズ

```
ユーザー入力
  - 生成したい文字: "あ", "い", "う", ...
  - スタイル指定: 既存フォント or 手書き見本
    ↓
骨格抽出
  - ベースとなる文字構造を取得
  - または、標準骨格データベースから取得
    ↓
スタイルエンコーディング
  - 参照フォント/手書きからスタイル特徴抽出
    ↓
条件付き生成
  - z_content (骨格) + z_style (スタイル) → Generator
    ↓
生成画像 (ラスター)
    ↓
ベクトル化
  - Potrace / 自前アルゴリズム
    ↓
後処理
  - スムージング、アンカーポイント最適化
    ↓
プレビュー表示
    ↓
ユーザー修正 (オプション)
    ↓
フォントファイル生成 (TTF/OTF)
```

---

## 5. モジュール設計

### 5.1 ディレクトリ構成

```
fontgen-ai/
├── README.md
├── requirements.txt
├── setup.py
│
├── config/
│   ├── model_config.yaml      # モデル設定
│   ├── training_config.yaml   # 学習設定
│   └── generation_config.yaml # 生成設定
│
├── data/
│   ├── fonts/                 # 学習用フォント
│   ├── processed/             # 前処理済みデータ
│   └── skeleton_db/           # 標準骨格データベース
│
├── models/
│   ├── __init__.py
│   ├── vae.py                 # VAEモデル
│   ├── skeleton_encoder.py   # 骨格抽出
│   ├── style_encoder.py      # スタイル抽出
│   ├── decoder.py             # 生成モデル
│   └── pretrained/            # 事前学習済みモデル
│
├── src/
│   ├── __init__.py
│   ├── font_parser.py         # フォント読み込み
│   ├── preprocessing.py       # データ前処理
│   ├── trainer.py             # 学習ループ
│   ├── generator.py           # 文字生成
│   ├── vectorizer.py          # ベクトル化
│   ├── postprocessing.py      # 後処理
│   └── font_builder.py        # フォントファイル生成
│
├── gui/
│   ├── __init__.py
│   ├── main_window.py         # メインGUI
│   ├── training_panel.py      # 学習パネル
│   ├── generation_panel.py    # 生成パネル
│   └── editor_panel.py        # 編集パネル
│
├── cli/
│   ├── train.py               # 学習CLI
│   ├── generate.py            # 生成CLI
│   └── batch_process.py       # バッチ処理
│
├── tests/
│   ├── test_font_parser.py
│   ├── test_models.py
│   └── test_generator.py
│
└── notebooks/
    ├── 01_data_exploration.ipynb
    ├── 02_model_training.ipynb
    └── 03_generation_demo.ipynb
```

### 5.2 主要クラス設計

```python
# font_parser.py
class FontParser:
    """フォントファイルの読み込みとグリフ抽出"""
    def load_font(self, font_path: str) -> Font
    def extract_glyphs(self, characters: List[str]) -> List[Glyph]
    def render_glyph(self, glyph: Glyph, size: int) -> np.ndarray

# preprocessing.py
class Preprocessor:
    """データ前処理"""
    def normalize_image(self, image: np.ndarray) -> np.ndarray
    def augment_data(self, images: List[np.ndarray]) -> List[np.ndarray]
    def create_dataset(self, font_dirs: List[str]) -> Dataset

# models/vae.py
class SkeletonEncoder(nn.Module):
    """文字の骨格(構造)を抽出"""
    def forward(self, x: Tensor) -> Tensor  # → z_content

class StyleEncoder(nn.Module):
    """文字のスタイル(個性)を抽出"""
    def forward(self, x: Tensor) -> Tensor  # → z_style

class FontDecoder(nn.Module):
    """骨格とスタイルから文字を生成"""
    def forward(self, z_content: Tensor, z_style: Tensor) -> Tensor

class FontVAE(nn.Module):
    """フォント生成VAE全体"""
    def __init__(self):
        self.skeleton_encoder = SkeletonEncoder()
        self.style_encoder = StyleEncoder()
        self.decoder = FontDecoder()

    def encode(self, x: Tensor) -> Tuple[Tensor, Tensor]
    def decode(self, z_content: Tensor, z_style: Tensor) -> Tensor
    def forward(self, x: Tensor) -> Tensor

# generator.py
class FontGenerator:
    """文字生成の高レベルインターフェース"""
    def __init__(self, model: FontVAE):
        self.model = model

    def generate_character(
        self,
        char: str,
        style_reference: Union[str, np.ndarray],
        options: Dict
    ) -> np.ndarray

    def generate_batch(
        self,
        chars: List[str],
        style_reference: Union[str, np.ndarray]
    ) -> List[np.ndarray]

# vectorizer.py
class Vectorizer:
    """ラスター画像をベクトル化"""
    def raster_to_vector(self, image: np.ndarray) -> SVGPath
    def optimize_bezier(self, path: SVGPath) -> SVGPath

# font_builder.py
class FontBuilder:
    """フォントファイルの構築"""
    def create_font(self, name: str) -> Font
    def add_glyph(self, char: str, svg_path: SVGPath)
    def set_metrics(self, metrics: FontMetrics)
    def export_ttf(self, output_path: str)
    def export_otf(self, output_path: str)
```

---

## 6. 学習データ設計

### 6.1 データセット構成

```
学習データセット:
  - 日本語: ひらがな (83文字)
  - 日本語: カタカナ (86文字)
  - 日本語: 常用漢字 (2,136文字)
  - 英語: A-Z, a-z (52文字)
  - 数字: 0-9 (10文字)
  - 記号: 主要記号 (約50文字)

総計: 約2,500文字

フォント数: 最低20フォント (多いほど良い)
```

### 6.2 データ拡張

- 回転: ±5度
- スケール: 0.95〜1.05
- 平行移動: ±5%
- ノイズ付加: Gaussian noise
- 太さ変更: Morphological operations

---

## 7. モデル詳細設計

### 7.1 ネットワーク構造 (VAEベース)

```python
# 推奨構成

SkeletonEncoder:
  Input: 128x128 grayscale image
  Conv2d(1, 64, 4, 2, 1)  → 64x64
  Conv2d(64, 128, 4, 2, 1) → 32x32
  Conv2d(128, 256, 4, 2, 1) → 16x16
  Conv2d(256, 512, 4, 2, 1) → 8x8
  Flatten → FC(512*8*8, 512) → FC(512, z_content_dim)

  Output: z_content (dim=128)

StyleEncoder:
  Input: 128x128 grayscale image
  Conv2d(1, 64, 4, 2, 1)  → 64x64
  Conv2d(64, 128, 4, 2, 1) → 32x32
  Conv2d(128, 256, 4, 2, 1) → 16x16
  AdaptiveAvgPool2d((4, 4)) → 256*16
  FC(256*16, 256) → FC(256, z_style_dim)

  Output: z_style (dim=64)

Decoder:
  Input: z = concat(z_content, z_style) (dim=192)
  FC(192, 512) → FC(512, 512*8*8) → Reshape(512, 8, 8)
  ConvTranspose2d(512, 256, 4, 2, 1) → 16x16
  ConvTranspose2d(256, 128, 4, 2, 1) → 32x32
  ConvTranspose2d(128, 64, 4, 2, 1)  → 64x64
  ConvTranspose2d(64, 1, 4, 2, 1)    → 128x128
  Sigmoid()

  Output: 128x128 grayscale image
```

### 7.2 損失関数

```python
# 総合損失
total_loss = reconstruction_loss +
             kl_divergence_loss +
             style_consistency_loss +
             skeleton_consistency_loss

# 再構成損失
reconstruction_loss = MSE(generated, original) +
                      perceptual_loss(generated, original)

# KLダイバージェンス (VAE正則化)
kl_loss = KL(q(z|x) || p(z))

# スタイル一貫性損失
# 同じフォントの異なる文字は同じスタイル特徴を持つべき
style_consistency_loss = MSE(z_style_A, z_style_B)
                         # A, B は同じフォントの異なる文字

# 骨格一貫性損失
# 同じ文字の異なるフォントは同じ骨格を持つべき
skeleton_consistency_loss = MSE(z_content_A, z_content_B)
                            # A, B は異なるフォントの同じ文字
```

### 7.3 学習ハイパーパラメータ

```yaml
# training_config.yaml
model:
  z_content_dim: 128
  z_style_dim: 64
  image_size: 128

training:
  batch_size: 64
  learning_rate: 0.0001
  num_epochs: 200
  optimizer: Adam
  scheduler: ReduceLROnPlateau

  loss_weights:
    reconstruction: 1.0
    kl_divergence: 0.001
    style_consistency: 0.5
    skeleton_consistency: 0.5

  early_stopping:
    patience: 20
    min_delta: 0.001

data:
  train_split: 0.8
  val_split: 0.1
  test_split: 0.1
  augmentation: true
```

---

## 8. 生成プロセス詳細

### 8.1 スタイル指定方法

```
方法1: 既存フォントから
  - フォントファイルを読み込み
  - 数文字をサンプリング
  - スタイルエンコーダーでz_styleを抽出

方法2: 手書き見本から
  - ユーザーが3〜10文字を手書き
  - 画像をアップロード
  - スタイルエンコーダーでz_styleを抽出
  - 複数サンプルの平均を取る

方法3: スタイル混合
  - 複数のz_styleを重み付き平均
  - z_style = w1*z_style1 + w2*z_style2 + ...

方法4: 潜在空間での補間
  - 2つのスタイル間を補間
  - z_style = lerp(z_style_A, z_style_B, t)
```

### 8.2 生成ワークフロー

```
ステップ1: プロジェクト作成
  - プロジェクト名入力
  - 対象文字セット選択 (ひらがな、漢字、etc.)
  - スタイル参照設定

ステップ2: サンプル生成 (100文字)
  - 選択した文字セットから100文字を自動生成
  - グリッド表示でプレビュー
  - 品質チェック

ステップ3: 確認・調整
  - 生成結果を確認
  - 気に入らない文字を個別に再生成
  - スタイルパラメータ微調整
    - 太さ調整
    - 傾き調整
    - 間隔調整

ステップ4: 個別編集 (オプション)
  - 特定の文字をGUIエディタで修正
  - アンカーポイントの調整
  - ベジェ曲線の編集

ステップ5: 全文字生成
  - 確認後、残りの全文字を生成
  - バッチ処理で高速化
  - 進捗バー表示

ステップ6: フォントファイル出力
  - メタデータ設定 (フォント名、作者、等)
  - メトリクス調整 (行間、字間、等)
  - TTF/OTF形式で出力
```

---

## 9. GUI設計

### 9.1 メイン画面構成

```
┌─────────────────────────────────────────────────────────┐
│  AI フォント生成システム                         [_][□][×]│
├─────────────────────────────────────────────────────────┤
│ [学習] [生成] [編集] [エクスポート]                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  [タブコンテンツエリア]                                   │
│                                                          │
│                                                          │
│                                                          │
│                                                          │
│                                                          │
├─────────────────────────────────────────────────────────┤
│  ステータス: 準備完了                              v1.0.0 │
└─────────────────────────────────────────────────────────┘
```

### 9.2 学習タブ

```
┌─────────────────────────────────────────────────────────┐
│  学習データ設定                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ フォントディレクトリ:                            │   │
│  │ /path/to/fonts                      [参照...]  │   │
│  │                                                  │   │
│  │ 検出されたフォント: 25個                         │   │
│  │ 学習文字数: 2,500文字                            │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  学習パラメータ                                          │
│  ┌─────────────────────────────────────────────────┐   │
│  │ エポック数: [200   ]                            │   │
│  │ バッチサイズ: [64    ]                          │   │
│  │ 学習率: [0.0001]                                │   │
│  │                                                  │   │
│  │ [詳細設定...]                                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  [学習開始]  [中断]  [学習済みモデル読込]               │
│                                                          │
│  進捗:                                                   │
│  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 45% (90/200 epochs)    │
│                                                          │
│  損失グラフ:                                             │
│  [リアルタイム更新グラフ]                                │
└─────────────────────────────────────────────────────────┘
```

### 9.3 生成タブ

```
┌─────────────────────────────────────────────────────────┐
│  プロジェクト設定                                        │
│  ┌─────────────────────────────────────────────────┐   │
│  │ プロジェクト名: [新しいフォント]                │   │
│  │ 対象文字: [✓]ひらがな [✓]カタカナ [✓]漢字      │   │
│  │           [✓]英数字   [ ]記号                   │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  スタイル設定                                            │
│  ┌─────────────────────────────────────────────────┐   │
│  │ スタイル参照: ● フォントファイル                │   │
│  │              ○ 手書き見本                       │   │
│  │              ○ スタイル混合                     │   │
│  │                                                  │   │
│  │ 参照フォント: [example.ttf]  [参照...]         │   │
│  │                                                  │   │
│  │ スタイル強度: ━━●━━━━━━━━ 70%                  │   │
│  └─────────────────────────────────────────────────┘   │
│                                                          │
│  [サンプル生成(100文字)]  [全文字生成]                  │
│                                                          │
│  生成結果プレビュー                                      │
│  ┌─────────────────────────────────────────────────┐   │
│  │ あ い う え お か き く け こ さ し す せ そ   │   │
│  │ た ち つ て と な に ぬ ね の は ひ ふ へ ほ   │   │
│  │ ...                                              │   │
│  │                                                  │   │
│  │ [再生成]  [個別編集]  [エクスポート]            │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

---

## 10. CLI設計

```bash
# 学習
fontgen-ai train \
  --data-dir ./fonts \
  --output-dir ./models/my_model \
  --config ./config/training_config.yaml \
  --epochs 200

# 生成
fontgen-ai generate \
  --model ./models/my_model \
  --style-ref ./reference_font.ttf \
  --characters "あいうえお" \
  --output ./generated/

# バッチ生成
fontgen-ai batch-generate \
  --model ./models/my_model \
  --style-ref ./reference_font.ttf \
  --charset hiragana,katakana,kanji \
  --output-font ./output/new_font.ttf

# フォント構築
fontgen-ai build-font \
  --glyphs-dir ./generated/ \
  --font-name "My New Font" \
  --output ./output/my_font.ttf
```

---

## 11. 評価指標

### 11.1 定量的評価

- **再構成精度**: MSE, SSIM
- **スタイル一貫性**: 同じスタイルの文字間の特徴量距離
- **骨格正確性**: 元の文字との構造的類似度
- **生成速度**: 1文字あたりの生成時間

### 11.2 定性的評価

- **視覚的品質**: 人間による評価
- **可読性**: 実際のテキストでの評価
- **スタイル再現性**: 参照フォントとの類似度

---

## 12. 拡張機能 (将来)

### 12.1 高度なスタイル制御

- テキストプロンプトからスタイル生成
  - 例: "手書き風の丸みのある明るいフォント"
  - GPT-4/CLIPを使用した条件付き生成

### 12.2 インタラクティブ編集

- リアルタイムスタイル調整
- スライダーによる潜在空間ナビゲーション
- 部分的な修正の学習

### 12.3 マルチモーダル学習

- 音声からフォントスタイル生成
- 画像からフォントスタイル抽出

### 12.4 Web版

- ブラウザベースの生成システム
- クラウドでの学習
- オンラインギャラリー

---

## 13. パフォーマンス要件

### 13.1 学習

- GPU必須: NVIDIA GTX 1660以上推奨
- メモリ: 16GB以上
- 学習時間: 20フォント、2,500文字で約6-12時間

### 13.2 生成

- GPU推奨 (CPUでも可能だが遅い)
- 1文字生成: 0.1〜1秒 (GPU)
- バッチ生成: 100文字で10〜60秒 (GPU)

### 13.3 ストレージ

- 学習データ: 約5GB (20フォント)
- モデルサイズ: 約100〜500MB
- 生成データ: 1フォントあたり約50MB

---

## 14. ライセンスと配布

### 14.1 プロジェクトライセンス

- MIT License または Apache 2.0推奨

### 14.2 生成フォントのライセンス

- 学習に使用したフォントのライセンスに注意
- 商用利用可能なフォントのみで学習を推奨
- 生成フォントの権利関係を明確化

---

## 15. リスクと制約

### 15.1 技術的リスク

- 複雑な漢字の生成品質が低い可能性
- スタイル転送の失敗
- ベクトル化の品質問題

### 15.2 法的リスク

- フォントの著作権問題
- 学習データの使用権

### 15.3 対策

- 多様なテストケースでの検証
- 段階的な開発とプロトタイピング
- 法務相談の実施

---

## 参考文献

1. **zi2zi**: Chinese Character Style Transfer
   - https://github.com/kaonashi-tyc/zi2zi

2. **FontVAE**: Disentangled Font Representation
   - https://github.com/hologerry/FontVAE

3. **DeepFont**: Font Recognition and Style Transfer
   - https://arxiv.org/abs/1507.03196

4. **AttnGAN**: Fine-Grained Text to Image Generation
   - https://github.com/taoxugit/AttnGAN

5. **StyleGAN3**: Alias-Free Generative Adversarial Networks
   - https://github.com/NVlabs/stylegan3

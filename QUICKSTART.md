# クイックスタートガイド

このガイドでは、AIフォント生成システムの開発を最短で始める方法を説明します。

---

## 前提条件

### 必須
- Python 3.10以上
- GPU (NVIDIA推奨、CUDA対応)
- 16GB以上のメモリ
- 50GB以上のストレージ空き容量

### 推奨
- NVIDIA RTX 3060以上のGPU
- 32GB以上のメモリ
- SSD

---

## セットアップ (15分)

### 1. リポジトリのクローン

```bash
git clone https://github.com/yourusername/fontgen-ai.git
cd fontgen-ai
```

### 2. 仮想環境の作成

```bash
# Python仮想環境作成
python -m venv venv

# 仮想環境の有効化
# Linux/Mac:
source venv/bin/activate

# Windows:
venv\Scripts\activate
```

### 3. 依存関係のインストール

```bash
# PyTorchのインストール (CUDA 11.8の場合)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# その他の依存関係
pip install -r requirements.txt

# 開発用ツール (オプション)
pip install -r requirements-dev.txt
```

### 4. ディレクトリ構造の作成

```bash
# 自動作成スクリプト
python scripts/setup_dirs.py

# または手動で
mkdir -p data/fonts data/processed data/skeleton_db
mkdir -p models/pretrained
mkdir -p output/samples output/fonts
```

---

## データ準備 (30分)

### 1. フォントの収集

最低20個の商用利用可能なフォントを `data/fonts/` に配置します。

#### 推奨フォント (無料・商用可)

**Google Fonts (https://fonts.google.com)**
- Noto Sans JP (Regular, Bold)
- Noto Serif JP (Regular, Bold)
- M PLUS 1p
- M PLUS Rounded 1c
- Sawarabi Gothic
- Sawarabi Mincho
- Kosugi
- Kosugi Maru

**その他**
- IPAフォント (https://moji.or.jp/ipafont/)
- 源ノ角ゴシック/Source Han Sans
- 源ノ明朝/Source Han Serif

```bash
# Google Fontsから自動ダウンロード (スクリプト使用)
python scripts/download_fonts.py --source google-fonts --output data/fonts/

# 手動でダウンロードした場合
# TTFファイルを data/fonts/ にコピー
cp ~/Downloads/*.ttf data/fonts/
```

### 2. データセットの構築

```bash
# 全フォントから文字画像を抽出
python cli/prepare_data.py \
  --font-dir ./data/fonts \
  --output-dir ./data/processed \
  --characters hiragana,katakana \
  --image-size 128 \
  --num-workers 4

# 処理時間: 約10-20分 (20フォント、約200文字の場合)
```

**出力:**
- `data/processed/train/` - 学習データ
- `data/processed/val/` - 検証データ
- `data/processed/test/` - テストデータ
- `data/processed/metadata.json` - メタデータ

### 3. データ確認

```bash
# データセットの統計情報を表示
python cli/data_info.py --data-dir ./data/processed

# サンプル画像を可視化
python scripts/visualize_data.py --data-dir ./data/processed --output ./output/data_samples.png
```

---

## 最小限の学習実験 (1-2時間)

まずは小規模なデータで動作確認を行います。

### 1. 設定ファイルの調整

```bash
# 最小構成の設定をコピー
cp config/training_config.minimal.yaml config/my_first_training.yaml
```

`config/my_first_training.yaml` の例:
```yaml
model:
  z_content_dim: 64      # 小さめ (デフォルト: 128)
  z_style_dim: 32        # 小さめ (デフォルト: 64)
  image_size: 128

training:
  batch_size: 32         # 小さめ
  learning_rate: 0.001
  num_epochs: 50         # 短めに設定
  optimizer: Adam

data:
  characters: hiragana   # ひらがなのみ (83文字)
  train_split: 0.8
  val_split: 0.2
```

### 2. 学習の開始

```bash
# 学習開始
python cli/train.py \
  --config config/my_first_training.yaml \
  --data-dir ./data/processed \
  --output-dir ./models/first_experiment \
  --gpu 0

# CPUのみの場合 (遅い)
python cli/train.py \
  --config config/my_first_training.yaml \
  --data-dir ./data/processed \
  --output-dir ./models/first_experiment \
  --device cpu
```

**進捗:**
- Epoch 1/50 ━━━━━━━━━━━━━━━━━━━━━━━━━━ 100% | Loss: 0.324
- 約1-2時間で完了 (GPU使用時)

### 3. 学習結果の確認

```bash
# TensorBoardで学習曲線を確認
tensorboard --logdir ./models/first_experiment/logs

# ブラウザで http://localhost:6006 を開く
```

**確認項目:**
- 再構成損失が減少しているか
- 検証損失が過学習していないか
- サンプル画像の品質

---

## 最初の文字生成 (5分)

### 1. スタイル参照の準備

学習データに含まれていたフォントを参照として使用します。

```bash
# 使用可能なフォント一覧
ls data/fonts/
```

### 2. 文字生成

```bash
# サンプル生成 (ひらがな10文字)
python cli/generate.py \
  --model ./models/first_experiment/checkpoints/best_model.pth \
  --style-ref ./data/fonts/NotoSansJP-Regular.ttf \
  --characters "あいうえおかきくけこ" \
  --output ./output/samples/first_generation/
```

**出力:**
- `output/samples/first_generation/あ.png`
- `output/samples/first_generation/い.png`
- ...

### 3. 結果の確認

```bash
# 生成画像を表示
python scripts/view_images.py --dir ./output/samples/first_generation/

# グリッド表示で保存
python scripts/make_grid.py \
  --input ./output/samples/first_generation/ \
  --output ./output/grid_first_generation.png \
  --cols 5
```

---

## フォントファイルの作成 (10分)

### 1. 全文字生成

```bash
# ひらがな全文字を生成
python cli/generate.py \
  --model ./models/first_experiment/checkpoints/best_model.pth \
  --style-ref ./data/fonts/NotoSansJP-Regular.ttf \
  --charset hiragana \
  --output ./output/glyphs/my_first_font/
```

### 2. フォントファイル構築

```bash
# TTFフォントを作成
python cli/build_font.py \
  --glyphs-dir ./output/glyphs/my_first_font/ \
  --font-name "My First AI Font" \
  --output ./output/fonts/my_first_font.ttf \
  --author "Your Name"
```

**出力:**
- `output/fonts/my_first_font.ttf`

### 3. フォントのインストールとテスト

#### Windows
```powershell
# フォントファイルをダブルクリックしてインストール
# または
explorer output\fonts\my_first_font.ttf
```

#### Mac
```bash
# フォントブックで開く
open output/fonts/my_first_font.ttf
```

#### Linux
```bash
# フォントディレクトリにコピー
mkdir -p ~/.fonts
cp output/fonts/my_first_font.ttf ~/.fonts/
fc-cache -fv
```

#### テスト

任意のテキストエディタで新しいフォントを選択して、ひらがなを入力してテストします。

---

## 次のステップ

### 1. より長時間の学習

```bash
# 本格的な学習 (200エポック、6-12時間)
python cli/train.py \
  --config config/training_config.yaml \
  --data-dir ./data/processed \
  --output-dir ./models/production_model \
  --gpu 0
```

### 2. 複数文字種の対応

```yaml
# config/training_config.yaml の修正
data:
  characters: hiragana,katakana,kanji  # 漢字追加
```

常用漢字を含める場合、データ準備から再実行が必要です:

```bash
python cli/prepare_data.py \
  --font-dir ./data/fonts \
  --output-dir ./data/processed \
  --characters hiragana,katakana,kanji_joyo \
  --image-size 128
```

### 3. GUIの起動

```bash
# GUIアプリケーション起動
python gui/main.py

# または
fontgen-ai gui
```

### 4. スタイル実験

```bash
# 手書き風フォントから学習
python cli/generate.py \
  --model ./models/production_model/best_model.pth \
  --style-ref ./path/to/handwriting_font.ttf \
  --charset hiragana,katakana \
  --output-font ./output/fonts/handwriting_style.ttf

# スタイル混合
python cli/generate.py \
  --model ./models/production_model/best_model.pth \
  --style-blend font1.ttf:0.6,font2.ttf:0.4 \
  --charset hiragana \
  --output-font ./output/fonts/blended_style.ttf
```

---

## トラブルシューティング

### CUDA/GPU関連

**エラー: CUDA out of memory**
```bash
# バッチサイズを減らす
# config/training_config.yaml:
training:
  batch_size: 16  # 32 → 16 に変更
```

**エラー: CUDA not available**
```bash
# PyTorchのCUDA対応を確認
python -c "import torch; print(torch.cuda.is_available())"

# CUDAバージョン確認
nvidia-smi

# PyTorchを適切なCUDAバージョンで再インストール
# https://pytorch.org/get-started/locally/
```

### データ関連

**エラー: Font file not found**
```bash
# フォントファイルのパスを確認
ls data/fonts/

# 絶対パスを使用
python cli/prepare_data.py \
  --font-dir /full/path/to/data/fonts \
  ...
```

**エラー: No characters found in font**
```bash
# フォントに含まれる文字を確認
python scripts/check_font.py --font-path data/fonts/font.ttf

# 一部のフォントは特定の文字種のみ対応
```

### 生成品質が低い

**対策:**
1. より長時間学習する (200エポック以上)
2. より多くのフォントでデータを増やす (30-50フォント)
3. 画像サイズを大きくする (256x256)
4. モデルのキャパシティを増やす (z_content_dim, z_style_dim)

```yaml
# config/training_config.yaml:
model:
  z_content_dim: 256  # 128 → 256
  z_style_dim: 128    # 64 → 128
  image_size: 256     # 128 → 256

training:
  num_epochs: 300
```

### メモリ不足

**対策:**
```yaml
# config/training_config.yaml:
training:
  batch_size: 16      # 小さくする
  num_workers: 2      # データローダーのワーカー数を減らす
```

または、データを段階的に学習:
```bash
# ひらがなのみで学習
python cli/train.py ... --characters hiragana

# その後、カタカナを追加して fine-tune
python cli/train.py ... --characters hiragana,katakana --resume models/.../checkpoint.pth
```

---

## 参考資料

### ドキュメント
- [完全な設計書](DESIGN.md)
- [詳細なロードマップ](ROADMAP.md)
- [ユーザーガイド](docs/USER_GUIDE.md)
- [API リファレンス](docs/API_REFERENCE.md)

### チュートリアル
- [Jupyter Notebooks](notebooks/)
  - `01_data_exploration.ipynb` - データ探索
  - `02_model_training.ipynb` - モデル学習
  - `03_generation_demo.ipynb` - 生成デモ

### コミュニティ
- GitHub Issues: バグ報告・機能要望
- Discussions: 質問・議論

---

## 最小限の例 (コピペ用)

完全な初回実行コマンド集:

```bash
# 1. セットアップ
git clone https://github.com/yourusername/fontgen-ai.git
cd fontgen-ai
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
python scripts/setup_dirs.py

# 2. データ準備
python scripts/download_fonts.py --source google-fonts --output data/fonts/
python cli/prepare_data.py \
  --font-dir ./data/fonts \
  --output-dir ./data/processed \
  --characters hiragana \
  --image-size 128

# 3. 学習 (1-2時間)
python cli/train.py \
  --config config/training_config.minimal.yaml \
  --data-dir ./data/processed \
  --output-dir ./models/first_experiment \
  --gpu 0

# 4. 生成
python cli/generate.py \
  --model ./models/first_experiment/checkpoints/best_model.pth \
  --style-ref ./data/fonts/NotoSansJP-Regular.ttf \
  --charset hiragana \
  --output-font ./output/fonts/my_first_font.ttf \
  --font-name "My First AI Font"

# 5. フォントインストール
# Windows: explorer output\fonts\my_first_font.ttf
# Mac: open output/fonts/my_first_font.ttf
# Linux: cp output/fonts/my_first_font.ttf ~/.fonts/ && fc-cache -fv
```

---

**これで始める準備が整いました！**

何か問題があれば、GitHub Issuesで報告してください。

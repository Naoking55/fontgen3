# AI フォント生成システム (fontgen-ai)

機械学習を用いて、既存フォントから学習し、新しいスタイルのフォントを自動生成するシステム

## 主要機能

1. **複数フォントからの学習** - 様々なフォントスタイルを学習
2. **骨格とスタイルの分離** - 文字の構造と個性を独立して抽出
3. **スタイル指定による生成** - 既存フォントや手書き見本からスタイルを転送
4. **インタラクティブ編集** - 生成結果を細かく調整
5. **フォントファイル出力** - TTF/OTF形式で完全なフォントを生成

## セットアップ

### 前提条件
- Python 3.10以上
- CUDA対応GPU (推奨)
- 16GB以上のメモリ

### インストール

```bash
# リポジトリをクローン
git clone https://github.com/yourusername/fontgen-ai.git
cd fontgen-ai

# 仮想環境を作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# PyTorchをインストール (CUDA 11.8の場合)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 依存関係をインストール
pip install -r requirements.txt

# ディレクトリを作成
python scripts/setup_dirs.py
```

## クイックスタート

詳細は [QUICKSTART.md](../QUICKSTART.md) を参照してください。

### 1. データ準備

```bash
# フォントをdata/fonts/に配置
cp path/to/fonts/*.ttf data/fonts/

# データセットを構築
python cli/prepare_data.py \
  --font-dir ./data/fonts \
  --output-dir ./data/processed \
  --characters hiragana,katakana
```

### 2. モデル学習

```bash
python cli/train.py \
  --config config/training_config.yaml \
  --data-dir ./data/processed \
  --output-dir ./models/my_model
```

### 3. フォント生成

```bash
python cli/generate.py \
  --model ./models/my_model/best_model.pth \
  --style-ref ./reference_font.ttf \
  --charset hiragana,katakana \
  --output-font ./output/my_font.ttf
```

## プロジェクト構造

```
fontgen-ai/
├── config/          # 設定ファイル
├── data/            # データ
│   ├── fonts/       # 学習用フォント
│   ├── processed/   # 前処理済みデータ
│   └── skeleton_db/ # 骨格データベース
├── models/          # モデル定義
├── src/             # コアロジック
├── gui/             # GUIアプリケーション
├── cli/             # CLIツール
├── tests/           # テスト
└── notebooks/       # Jupyter notebooks
```

## ドキュメント

- [設計書](../DESIGN.md) - システム全体の設計
- [ロードマップ](../ROADMAP.md) - 開発計画
- [クイックスタート](../QUICKSTART.md) - 最速で始める方法

## 技術スタック

- **機械学習**: PyTorch 2.0+
- **モデル**: VAE (Variational Autoencoder)
- **フォント処理**: fontTools, freetype-py, Pillow

## 開発状況

- [x] Phase 0: プロジェクトセットアップ
- [ ] Phase 1: データパイプライン構築
- [ ] Phase 2: モデル実装
- [ ] Phase 3: 生成機能
- [ ] Phase 4: GUI実装
- [ ] Phase 5: 高度な機能
- [ ] Phase 6: 最適化と公開

## ライセンス

MIT License

## 貢献

プルリクエスト歓迎！

## 著者

Your Name

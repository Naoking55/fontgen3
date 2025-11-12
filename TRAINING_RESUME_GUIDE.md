# フォント生成AI 学習再開ガイド

このガイドでは、fontgen3プロジェクトでフォント生成AIの学習を再開する手順を説明します。

## 🚨 重要な注意事項

現在のWeb環境には以下の制限があります：
- rcloneのインストールが困難
- 大量のデータダウンロードに制限
- GPU/CUDAが利用できない可能性

**推奨：ローカルマシンまたはGPU環境で実行してください**

## 📋 前提条件

- Python 3.10以上
- CUDA対応GPU（推奨、CPUでも可能だが遅い）
- 16GB以上のメモリ
- Google Driveへのアクセス

## 🔧 セットアップ手順

### 1. リポジトリのクローン

```bash
git clone <repository-url>
cd fontgen3
git checkout claude/resume-font-generation-training-011CV4SEYdUdqcp9XpyD73Ms
```

### 2. Rcloneのセットアップ

**方法A: Rcloneを使用（推奨）**

```bash
# Rcloneをインストール
curl https://rclone.org/install.sh | sudo bash

# Google Driveを設定
rclone config
# 詳細は RCLONE_SETUP.md を参照

# 設定を確認
rclone listremotes
# 出力: gdrive:
```

**方法B: 手動ダウンロード（Rcloneが使えない場合）**

1. Google Driveにアクセス
2. `fontgen-ai`フォルダを探す
3. 以下をダウンロード：
   - `checkpoints/` - 学習チェックポイント
   - `hiragana_kanji/` - データセット全体（または）
   - `processed_hiragana_kanji/` - 前処理済みデータ

### 3. データとチェックポイントのダウンロード

**方法A: Rcloneを使用**

```bash
# チェックポイントをダウンロード
./rclone_sync.sh download-checkpoints

# データセット全体をダウンロード
./rclone_sync.sh download-all

# または、手動で：
rclone copy gdrive:fontgen-ai/checkpoints/ outputs/hiragana_kanji/checkpoints/ -P
rclone copy gdrive:fontgen-ai/hiragana_kanji/ outputs/hiragana_kanji/ -P
```

**方法B: 手動配置**

ダウンロードしたファイルを以下のように配置：

```
fontgen3/
├── fontgen-ai/
│   └── data/
│       └── processed_hiragana_kanji/
│           ├── train/
│           ├── val/
│           └── metadata.json
└── outputs/
    └── hiragana_kanji/
        └── checkpoints/
            ├── model_final.pt
            ├── model_best.pt
            └── checkpoint_epoch_*.pt
```

### 4. Python環境のセットアップ

```bash
cd fontgen-ai

# 仮想環境を作成（オプションだが推奨）
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# PyTorchをインストール（CUDA 11.8の場合）
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CPUのみの場合：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 依存関係をインストール
pip install -r requirements.txt
```

### 5. ディレクトリ構造の確認

```bash
# 必要なディレクトリを作成
python scripts/setup_dirs.py

# データを確認
python scripts/data_info.py --data-dir ./data/processed_hiragana_kanji
```

## 🎓 学習の再開

### 最新チェックポイントから再開

```bash
cd fontgen-ai

python cli/train.py \
  --config config/training_config.hiragana_kanji.yaml \
  --data-dir ./data/processed_hiragana_kanji \
  --output-dir ../outputs/hiragana_kanji \
  --resume ../outputs/hiragana_kanji/checkpoints/model_final.pt \
  --device auto
```

### 特定のエポックから再開

```bash
python cli/train.py \
  --config config/training_config.hiragana_kanji.yaml \
  --data-dir ./data/processed_hiragana_kanji \
  --output-dir ../outputs/hiragana_kanji \
  --resume ../outputs/hiragana_kanji/checkpoints/checkpoint_epoch_45.pt \
  --device auto
```

### CPUで実行する場合

```bash
python cli/train.py \
  --config config/training_config.hiragana_kanji.yaml \
  --data-dir ./data/processed_hiragana_kanji \
  --output-dir ../outputs/hiragana_kanji \
  --resume ../outputs/hiragana_kanji/checkpoints/model_final.pt \
  --device cpu
```

## 💾 学習後のアップロード

学習が完了したら、新しいチェックポイントをGoogle Driveにアップロードします：

```bash
cd ..

# チェックポイントのみアップロード
./rclone_sync.sh upload-checkpoints

# 全データをアップロード（サンプル画像、ログなど含む）
./rclone_sync.sh upload-all
```

## 📊 学習の監視

### TensorBoardを使用

```bash
# 別のターミナルで実行
tensorboard --logdir outputs/hiragana_kanji/logs

# ブラウザで http://localhost:6006 にアクセス
```

### 学習状況の確認

```bash
# 最新のチェックポイントを確認
ls -lth outputs/hiragana_kanji/checkpoints/

# ログを確認
tail -f outputs/hiragana_kanji/logs/training.log

# サンプル画像を確認
ls outputs/hiragana_kanji/samples/
```

## ⚙️ 設定のカスタマイズ

設定ファイル `config/training_config.hiragana_kanji.yaml` を編集できます：

```yaml
model:
  z_content_dim: 192      # コンテンツ潜在次元
  z_style_dim: 96         # スタイル潜在次元
  image_size: 128

training:
  batch_size: 32          # バッチサイズ（メモリに応じて調整）
  num_epochs: 80          # エポック数
  learning_rate: 0.0005   # 学習率
```

## 🔍 トラブルシューティング

### メモリ不足エラー

```bash
# バッチサイズを減らす
# config/training_config.hiragana_kanji.yaml の batch_size を 16 または 8 に変更
```

### GPUが認識されない

```bash
# CUDA確認
python -c "import torch; print(torch.cuda.is_available())"

# デバイス情報確認
python -c "import torch; print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

### データが見つからない

```bash
# データディレクトリを確認
ls -la fontgen-ai/data/processed_hiragana_kanji/

# メタデータを確認
cat fontgen-ai/data/processed_hiragana_kanji/metadata.json

# データを再生成（フォントがある場合）
cd fontgen-ai
python cli/prepare_data.py \
  --font-dir ./data/fonts \
  --output-dir ./data/processed_hiragana_kanji \
  --characters hiragana,kanji_joyo \
  --image-size 128
```

## 📚 データセット情報

現在のデータセット（`processed_hiragana_kanji`）：
- **文字数**: 258文字（ひらがな81 + 常用漢字197）
- **フォント数**: 5種類
  - 学習用: 3フォント (MS Gothic, FGGyoshoLC-M, Ro GSan Serif Std U)
  - 検証用: 1フォント (VD-LogoMaru-Medium-G)
  - テスト用: 1フォント (YDW バナナスリップplus plus)
- **画像サイズ**: 128x128
- **データ分割**: Train 75% / Val 25%

## 📖 参考ドキュメント

- [RCLONE_SETUP.md](RCLONE_SETUP.md) - Rclone詳細設定
- [fontgen-ai/README.md](fontgen-ai/README.md) - AIシステム概要
- [fontgen-ai/TRAINING_REPORT.md](fontgen-ai/TRAINING_REPORT.md) - 学習レポート

## 🎯 クイックスタート（すべてが揃っている場合）

```bash
# 1. データとチェックポイントをダウンロード
./rclone_sync.sh download-all

# 2. 依存関係をインストール
cd fontgen-ai
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt

# 3. 学習を再開
python cli/train.py \
  --config config/training_config.hiragana_kanji.yaml \
  --data-dir ./data/processed_hiragana_kanji \
  --output-dir ../outputs/hiragana_kanji \
  --resume ../outputs/hiragana_kanji/checkpoints/model_final.pt

# 4. 完了後、アップロード
cd ..
./rclone_sync.sh upload-checkpoints
```

---

**注意**: Web環境（Claude Code）では、リソース制限のため学習の実行が困難です。ローカルマシンまたはクラウドGPU環境での実行を強く推奨します。

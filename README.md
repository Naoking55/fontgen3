# fontgen3

AI Font Generation Project

## プロジェクト概要

このプロジェクトは、AIを使用してフォント生成を行うツールです。

## ファイル構成

- `font_editor1.50.py` - フォント編集ツール
- `font_parts_extractor_full07.06.py` - フォント部品抽出ツール

## Google Drive連携（Rclone）

モデルのチェックポイントやデータをGoogle Driveと同期できます。

### クイックスタート

```bash
# 1. Rcloneをインストール（初回のみ）
curl https://rclone.org/install.sh | sudo bash

# 2. Google Driveを設定（初回のみ、約5分）
rclone config

# 3. 便利なスクリプトを使用
./rclone_sync.sh upload-checkpoints    # チェックポイントをアップロード
./rclone_sync.sh download-checkpoints  # チェックポイントをダウンロード
./rclone_sync.sh sync-to-cloud         # 全データを同期
```

### 詳細な設定方法

詳しい設定手順は [RCLONE_SETUP.md](RCLONE_SETUP.md) を参照してください。

### 手動コマンド

```bash
# アップロード
rclone copy outputs/hiragana_kanji/checkpoints/ gdrive:fontgen-ai/checkpoints/

# ダウンロード
rclone copy gdrive:fontgen-ai/checkpoints/ outputs/hiragana_kanji/checkpoints/

# 差分同期（最速）
rclone sync outputs/hiragana_kanji/ gdrive:fontgen-ai/hiragana_kanji/
```

## 使用方法

TBD
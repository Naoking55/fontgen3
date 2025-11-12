# Rclone Google Drive セットアップガイド

このガイドでは、rcloneを使用してGoogle Driveとモデルのチェックポイントを同期する方法を説明します。

## 初回セットアップ（約5分）

### 1. Rcloneのインストール

```bash
curl https://rclone.org/install.sh | sudo bash
```

インストール確認:
```bash
rclone version
```

### 2. Google Driveの設定

```bash
rclone config
```

以下の手順で設定します:

1. `n` を入力して新しいリモートを作成
2. 名前を `gdrive` と入力
3. ストレージタイプで `drive` (Google Drive) を選択
4. Client IDとClient Secretは空白のままEnter（デフォルト使用）
5. スコープは `1` (Full access) を選択
6. Root folder IDは空白のままEnter
7. Service Account Fileは空白のままEnter
8. 詳細設定は `n` (No)
9. Auto configは:
   - ローカル環境: `y` (Yes) - ブラウザが開きます
   - リモートサーバー: `n` (No) - 表示されたURLをローカルブラウザで開いてコードを入力
10. Google アカウントでログインして認証
11. Team Driveは `n` (No)
12. 設定確認後 `y` (Yes)
13. `q` で終了

### 3. 設定の確認

```bash
rclone listremotes
```

`gdrive:` と表示されればOKです。

## 使用方法

### チェックポイントのアップロード

```bash
# チェックポイントのみアップロード
rclone copy outputs/hiragana_kanji/checkpoints/ gdrive:fontgen-ai/checkpoints/ -P

# 全データをアップロード
rclone copy outputs/hiragana_kanji/ gdrive:fontgen-ai/hiragana_kanji/ -P
```

### チェックポイントのダウンロード

```bash
# チェックポイントのみダウンロード
rclone copy gdrive:fontgen-ai/checkpoints/ outputs/hiragana_kanji/checkpoints/ -P

# 全データをダウンロード
rclone copy gdrive:fontgen-ai/hiragana_kanji/ outputs/hiragana_kanji/ -P
```

### 差分同期（最も効率的）

**注意**: `sync` コマンドは送信元と完全に同期します（送信先の余分なファイルは削除されます）

```bash
# ローカル → Google Drive
rclone sync outputs/hiragana_kanji/ gdrive:fontgen-ai/hiragana_kanji/ -P

# Google Drive → ローカル
rclone sync gdrive:fontgen-ai/hiragana_kanji/ outputs/hiragana_kanji/ -P
```

### ユーティリティスクリプトの使用

便利なスクリプトを用意しています:

```bash
# 実行権限を付与
chmod +x rclone_sync.sh

# チェックポイントをアップロード
./rclone_sync.sh upload-checkpoints

# チェックポイントをダウンロード
./rclone_sync.sh download-checkpoints

# 全データを同期（ローカル → クラウド）
./rclone_sync.sh sync-to-cloud

# 全データを同期（クラウド → ローカル）
./rclone_sync.sh sync-from-cloud
```

## オプション

- `-P`: 進捗状況を表示
- `--dry-run`: 実際には実行せず、何が起こるかを表示
- `--exclude "pattern"`: 特定のパターンを除外
- `--transfers 4`: 並列転送数（デフォルト: 4）
- `--bwlimit 10M`: 帯域幅制限（例: 10MB/s）

## トラブルシューティング

### 認証エラー

```bash
rclone config reconnect gdrive:
```

### 設定の再作成

```bash
rclone config delete gdrive
rclone config  # 再度設定
```

### 接続テスト

```bash
rclone lsd gdrive:
```

## ディレクトリ構造

Google Drive上の推奨構造:
```
fontgen-ai/
├── hiragana_kanji/
│   ├── checkpoints/
│   ├── logs/
│   └── samples/
├── checkpoints/  # チェックポイントのバックアップ
└── models/       # 完成したモデル
```

## 参考リンク

- [Rclone公式ドキュメント](https://rclone.org/docs/)
- [Google Driveリモート設定](https://rclone.org/drive/)
- [Rcloneコマンドリファレンス](https://rclone.org/commands/)

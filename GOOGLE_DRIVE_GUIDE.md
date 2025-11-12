# Google Drive活用ガイド - fontgen-aiプロジェクト

**最終更新:** 2025-11-12

Google Driveを使って機械学習プロジェクトのデータ・モデルを保存し、セッション間で引き継ぐ方法を説明します。

---

## 📊 プロジェクトのデータサイズ推定

```
前処理データ:        約10-50MB
学習済みモデル:      約80-200MB/モデル
評価結果:            約5-10MB
合計（1回の学習）:   約100-250MB
```

**Google Driveプラン:**
- 無料: 15GB（不十分）
- 100GB: ¥250/月 ✅ 十分
- 2TB: ¥1,300/月 ✅ 余裕

---

## 🚀 実装方法（3つのアプローチ）

### オプション1: gdown（最もシンプル）⭐⭐⭐⭐⭐

**メリット:**
- インストールが簡単
- スクリプトで自動化可能
- 認証不要（共有リンク使用）

**デメリット:**
- アップロードはWebから手動
- 大きなファイルは制限がある場合も

#### 実装手順

**1. モデルをGoogle Driveにアップロード（手動）**

```bash
# 学習完了後、以下のファイルをダウンロード
outputs/hiragana_kanji/checkpoints/best.pth
outputs/hiragana_kanji/checkpoints/last.pth
outputs/hiragana_kanji/evaluation/

# Webブラウザでdrive.google.comにアクセス
# "fontgen-ai" フォルダを作成
# ファイルをドラッグ&ドロップでアップロード
```

**2. 共有リンクを取得**

```
1. アップロードしたファイルを右クリック
2. 「共有」→「リンクをコピー」
3. URLをメモ: https://drive.google.com/file/d/FILE_ID/view?usp=sharing
```

**3. 次のセッションでダウンロード**

```bash
# gdownをインストール
pip install gdown

# ファイルIDを抽出（URLから）
# https://drive.google.com/file/d/1ABC123xyz/view?usp=sharing
# ↓
# FILE_ID = 1ABC123xyz

# ダウンロード
gdown 1ABC123xyz -O outputs/hiragana_kanji/checkpoints/best.pth
```

#### 自動化スクリプト例

```python
# scripts/download_from_gdrive.py
import gdown
import os
from pathlib import Path

# Google DriveのファイルID（実際の値に置き換え）
GDRIVE_FILES = {
    "best_model": "1ABC123xyz",  # best.pthのファイルID
    "last_model": "1DEF456abc",  # last.pthのファイルID
}

def download_model(file_id, output_path):
    """Google Driveからモデルをダウンロード"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"📥 Downloading to {output_path}...")
    gdown.download(
        f"https://drive.google.com/uc?id={file_id}",
        output_path,
        quiet=False
    )
    print(f"✅ Downloaded: {output_path}")

# 使用例
if __name__ == "__main__":
    download_model(
        GDRIVE_FILES["best_model"],
        "outputs/hiragana_kanji/checkpoints/best.pth"
    )
    download_model(
        GDRIVE_FILES["last_model"],
        "outputs/hiragana_kanji/checkpoints/last.pth"
    )
```

**使い方:**
```bash
python scripts/download_from_gdrive.py
```

---

### オプション2: PyDrive2（完全自動化）⭐⭐⭐⭐

**メリット:**
- アップロード・ダウンロード両方を自動化
- フォルダ構造を維持
- バッチ処理可能

**デメリット:**
- 初回認証が必要
- 設定がやや複雑

#### 実装手順

**1. PyDrive2をインストール**

```bash
pip install PyDrive2
```

**2. Google Cloud Console設定**

```
1. https://console.cloud.google.com/ にアクセス
2. 新規プロジェクト作成: "fontgen-ai"
3. Google Drive APIを有効化
4. 認証情報を作成:
   - OAuth 2.0 クライアントID
   - アプリケーションの種類: デスクトップアプリ
5. credentials.jsonをダウンロード
6. fontgen-ai/credentials.json に配置
```

**3. 認証スクリプト作成**

```python
# scripts/gdrive_manager.py
from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
from pathlib import Path
import os

class GDriveManager:
    def __init__(self):
        """Google Drive認証"""
        gauth = GoogleAuth()

        # 認証情報のキャッシュ
        if os.path.exists("credentials_cache.json"):
            gauth.LoadCredentialsFile("credentials_cache.json")

        if gauth.credentials is None:
            # 初回認証
            gauth.LocalWebserverAuth()
        elif gauth.access_token_expired:
            # トークン更新
            gauth.Refresh()
        else:
            # 既存の認証情報を使用
            gauth.Authorize()

        # 認証情報を保存
        gauth.SaveCredentialsFile("credentials_cache.json")

        self.drive = GoogleDrive(gauth)
        self.folder_id = None

    def create_folder(self, folder_name, parent_id=None):
        """フォルダを作成"""
        metadata = {
            'title': folder_name,
            'mimeType': 'application/vnd.google-apps.folder'
        }
        if parent_id:
            metadata['parents'] = [{'id': parent_id}]

        folder = self.drive.CreateFile(metadata)
        folder.Upload()
        print(f"✅ Created folder: {folder_name}")
        return folder['id']

    def upload_file(self, local_path, gdrive_folder_id=None):
        """ファイルをアップロード"""
        file_name = Path(local_path).name

        metadata = {'title': file_name}
        if gdrive_folder_id:
            metadata['parents'] = [{'id': gdrive_folder_id}]

        file = self.drive.CreateFile(metadata)
        file.SetContentFile(local_path)
        file.Upload()

        print(f"✅ Uploaded: {local_path} -> {file_name}")
        return file['id']

    def download_file(self, file_id, local_path):
        """ファイルをダウンロード"""
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        file = self.drive.CreateFile({'id': file_id})
        file.GetContentFile(local_path)

        print(f"✅ Downloaded: {file['title']} -> {local_path}")

    def list_files(self, folder_id=None):
        """ファイル一覧を取得"""
        query = f"'{folder_id}' in parents" if folder_id else None
        file_list = self.drive.ListFile({'q': query}).GetList()

        for file in file_list:
            print(f"- {file['title']} (ID: {file['id']})")

        return file_list

# 使用例
if __name__ == "__main__":
    manager = GDriveManager()

    # フォルダ作成
    folder_id = manager.create_folder("fontgen-ai-models")

    # モデルをアップロード
    manager.upload_file(
        "outputs/hiragana_kanji/checkpoints/best.pth",
        folder_id
    )
```

**4. 学習後に自動アップロード**

```python
# scripts/backup_to_gdrive.py
from gdrive_manager import GDriveManager
from pathlib import Path

def backup_training_results(output_dir):
    """学習結果をGoogle Driveにバックアップ"""
    manager = GDriveManager()

    # プロジェクトフォルダを取得または作成
    folders = manager.list_files()
    project_folder = None
    for f in folders:
        if f['title'] == 'fontgen-ai-models':
            project_folder = f['id']
            break

    if not project_folder:
        project_folder = manager.create_folder('fontgen-ai-models')

    # 重要なファイルをアップロード
    files_to_backup = [
        f"{output_dir}/checkpoints/best.pth",
        f"{output_dir}/checkpoints/last.pth",
        f"{output_dir}/samples/grid.png",
    ]

    for file_path in files_to_backup:
        if Path(file_path).exists():
            manager.upload_file(file_path, project_folder)
        else:
            print(f"⚠️ File not found: {file_path}")

# 使用例
if __name__ == "__main__":
    backup_training_results("outputs/hiragana_kanji")
```

---

### オプション3: rclone（プロフェッショナル）⭐⭐⭐⭐⭐

**メリット:**
- 最も高速
- rsyncのように差分同期
- 複数のクラウドサービスに対応
- コマンドラインで完結

**デメリット:**
- インストールと設定が必要
- Linuxコマンドの知識が必要

#### 実装手順

**1. rcloneをインストール**

```bash
curl https://rclone.org/install.sh | sudo bash
```

**2. Google Driveを設定**

```bash
rclone config

# 対話型設定
n) New remote
name> gdrive
Storage> drive  # Google Drive
client_id> （Enterでスキップ）
client_secret> （Enterでスキップ）
scope> 1  # Full access
root_folder_id> （Enterでスキップ）
service_account_file> （Enterでスキップ）
# ブラウザで認証
y) Yes this is OK
```

**3. アップロード・ダウンロード**

```bash
# アップロード
rclone copy outputs/hiragana_kanji/checkpoints/ gdrive:fontgen-ai/checkpoints/

# ダウンロード
rclone copy gdrive:fontgen-ai/checkpoints/ outputs/hiragana_kanji/checkpoints/

# 同期（差分のみ）
rclone sync outputs/hiragana_kanji/ gdrive:fontgen-ai/hiragana_kanji/

# 確認
rclone ls gdrive:fontgen-ai/
```

**4. 自動化スクリプト**

```bash
# scripts/sync_to_gdrive.sh
#!/bin/bash

OUTPUT_DIR="outputs/hiragana_kanji"
GDRIVE_PATH="gdrive:fontgen-ai/hiragana_kanji"

echo "📤 Syncing to Google Drive..."

# 重要なファイルのみ同期
rclone copy $OUTPUT_DIR/checkpoints/best.pth $GDRIVE_PATH/checkpoints/
rclone copy $OUTPUT_DIR/checkpoints/last.pth $GDRIVE_PATH/checkpoints/
rclone copy $OUTPUT_DIR/evaluation/ $GDRIVE_PATH/evaluation/ --exclude "*.log"

echo "✅ Sync complete!"

# 確認
rclone ls $GDRIVE_PATH
```

```bash
# 実行
chmod +x scripts/sync_to_gdrive.sh
./scripts/sync_to_gdrive.sh
```

---

## 🎯 推奨ワークフロー

### シンプル版（初心者向け）

**学習完了後:**
```bash
# 1. ファイルをローカルに確認
ls -lh outputs/hiragana_kanji/checkpoints/

# 2. Webブラウザでdrive.google.comを開く
# 3. "fontgen-ai"フォルダを作成
# 4. best.pth, last.pthをアップロード
# 5. 共有リンクを取得してメモ
```

**次のセッション:**
```bash
# 1. gdownをインストール
pip install gdown

# 2. ダウンロード
gdown FILE_ID -O outputs/hiragana_kanji/checkpoints/best.pth

# 3. 学習を再開
python cli/train.py --resume outputs/hiragana_kanji/checkpoints/best.pth
```

### 自動化版（上級者向け）

**学習スクリプトに統合:**
```python
# cli/train.py の最後に追加
if training_complete:
    print("💾 Backing up to Google Drive...")
    from scripts.backup_to_gdrive import backup_training_results
    backup_training_results(args.output_dir)
```

---

## 📝 設定ファイル管理

### ファイルIDを設定ファイルに保存

```yaml
# config/gdrive_config.yaml
gdrive:
  enabled: true
  folder_id: "1XYZ_FOLDER_ID"

  files:
    hiragana_kanji:
      best_model: "1ABC_BEST_MODEL_ID"
      last_model: "1DEF_LAST_MODEL_ID"
      evaluation: "1GHI_EVAL_FOLDER_ID"
```

```python
# scripts/gdrive_config.py
import yaml

def load_gdrive_config():
    with open("config/gdrive_config.yaml") as f:
        return yaml.safe_load(f)

def save_file_id(model_name, file_type, file_id):
    config = load_gdrive_config()
    if model_name not in config['gdrive']['files']:
        config['gdrive']['files'][model_name] = {}
    config['gdrive']['files'][model_name][file_type] = file_id

    with open("config/gdrive_config.yaml", 'w') as f:
        yaml.dump(config, f)
```

---

## 🔐 セキュリティ注意事項

### 認証情報の管理

```bash
# .gitignoreに追加
echo "credentials.json" >> .gitignore
echo "credentials_cache.json" >> .gitignore
echo "token.pickle" >> .gitignore
```

### 共有設定

- **プライベートプロジェクト**: 自分のみアクセス可
- **チーム開発**: 特定のGoogleアカウントと共有
- **公開プロジェクト**: リンクを知っている全員

---

## 💡 ベストプラクティス

### フォルダ構造

```
Google Drive/
└── fontgen-ai/
    ├── models/
    │   ├── hiragana_kanji/
    │   │   ├── best.pth
    │   │   ├── last.pth
    │   │   └── config.yaml
    │   └── hiragana_only/
    │       └── ...
    ├── data/
    │   └── processed_hiragana_kanji.zip
    └── evaluation/
        └── reports/
```

### バージョン管理

```
モデル名の命名規則:
- best_v1.0_ssim0.65.pth
- best_v2.0_ssim0.70.pth
- last_epoch50.pth
```

### 定期バックアップ

```bash
# cron jobで自動バックアップ（毎日深夜2時）
0 2 * * * /path/to/scripts/sync_to_gdrive.sh
```

---

## 🎓 実際の使用例

### ケース1: 学習の中断と再開

```bash
# セッション1: 学習開始
python cli/train.py --config config/training_config.hiragana_kanji.yaml

# 学習完了後、手動でGoogle Driveにアップロード

# セッション2: 学習再開
pip install gdown
gdown 1ABC123xyz -O outputs/hiragana_kanji/checkpoints/best.pth
python cli/generate.py --model outputs/hiragana_kanji/checkpoints/best.pth
```

### ケース2: 複数マシンで作業

```bash
# マシンA: 学習
python cli/train.py ...
rclone sync outputs/ gdrive:fontgen-ai/outputs/

# マシンB: 評価
rclone sync gdrive:fontgen-ai/outputs/ outputs/
python cli/evaluate_quality.py --model outputs/hiragana_kanji/checkpoints/best.pth
```

---

## 📊 コスト試算

**100GBプラン（¥250/月）の場合:**
```
- モデル1つ: 100MB
- 保存可能モデル数: 約1,000モデル
- 実用上: 50-100モデル（各バージョン + 評価結果含む）
```

**十分すぎる容量です！**

---

## 🚨 トラブルシューティング

### Q: ダウンロードが「ウイルススキャン警告」で止まる
```bash
# gdownの場合
gdown --fuzzy "GOOGLE_DRIVE_URL"

# 手動でスキップ
curl -L "DIRECT_DOWNLOAD_URL" -o model.pth
```

### Q: アップロードが遅い
```bash
# rcloneで並列アップロード
rclone copy . gdrive:fontgen-ai/ --transfers=8 --checkers=16
```

### Q: 認証エラー
```bash
# PyDrive2: 認証情報を削除して再認証
rm credentials_cache.json
python scripts/gdrive_manager.py
```

---

## 📦 まとめ

| 方法 | 難易度 | 自動化 | 推奨度 |
|------|--------|--------|--------|
| gdown | ⭐ | ✗ | ⭐⭐⭐⭐⭐ 初心者 |
| PyDrive2 | ⭐⭐⭐ | ✓ | ⭐⭐⭐⭐ 自動化 |
| rclone | ⭐⭐ | ✓ | ⭐⭐⭐⭐⭐ プロ |

**このプロジェクトの推奨:**
1. **今すぐ始める**: gdown（手動アップロード）
2. **慣れてきたら**: rclone（同期自動化）
3. **複雑な管理**: PyDrive2（完全プログラマブル）

---

**次のステップ:**
1. まず手動でアップロード・ダウンロードを試す
2. うまくいったらスクリプト化
3. 余裕があればrcloneで自動化

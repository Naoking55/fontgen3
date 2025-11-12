# 外部ストレージ選択肢ガイド

**最終更新:** 2025-11-12

機械学習プロジェクトの大きなファイル（モデル、データセット）を保存するための外部ストレージサービスの比較です。

---

## 📦 主要な選択肢

### 1. 🤗 Hugging Face Hub（推奨 - ML特化）

**概要:**
機械学習モデルとデータセットの共有に特化したプラットフォーム

**メリット:**
- ✅ **無料**: 無制限の公開リポジトリ
- ✅ **ML特化**: モデルとデータセット用に最適化
- ✅ **バージョン管理**: Git LFS統合
- ✅ **簡単な共有**: URLで簡単にアクセス
- ✅ **PyTorch統合**: `torch.hub.load()`で直接ロード可能
- ✅ **API**: Python SDKで簡単にアップロード/ダウンロード

**デメリット:**
- ⚠️ プライベートリポジトリは有料（$9/月〜）
- ⚠️ ファイルサイズ制限: 50GB/ファイル

**使用例:**
```python
# インストール
pip install huggingface_hub

# アップロード
from huggingface_hub import HfApi
api = HfApi()
api.upload_file(
    path_or_fileobj="outputs/best.pth",
    path_in_repo="models/best.pth",
    repo_id="username/fontgen-ai",
    token="your_token"
)

# ダウンロード
from huggingface_hub import hf_hub_download
model_path = hf_hub_download(
    repo_id="username/fontgen-ai",
    filename="models/best.pth"
)
```

**料金:**
- 無料: 公開リポジトリ無制限
- Pro ($9/月): プライベートリポジトリ無制限
- Enterprise ($20/月〜): チーム機能

**おすすめ度: ⭐⭐⭐⭐⭐**

---

### 2. ☁️ Google Drive

**概要:**
一般的なクラウドストレージサービス

**メリット:**
- ✅ **無料枠**: 15GB無料
- ✅ **使いやすい**: GUIで簡単操作
- ✅ **共有簡単**: リンク共有
- ✅ **Google Colab統合**: 簡単にマウント可能

**デメリット:**
- ⚠️ 無料は15GBまで
- ⚠️ ダウンロード制限（大きなファイルは1日の制限あり）
- ⚠️ API使用には認証が複雑

**使用例:**
```python
# Google Colab での使用
from google.colab import drive
drive.mount('/content/drive')

# gdownでダウンロード（スクリプト用）
pip install gdown
import gdown
url = 'https://drive.google.com/uc?id=FILE_ID'
output = 'best.pth'
gdown.download(url, output, quiet=False)
```

**料金:**
- 無料: 15GB
- Google One 100GB: ¥250/月
- Google One 2TB: ¥1,300/月

**おすすめ度: ⭐⭐⭐⭐**

---

### 3. 📦 AWS S3

**概要:**
Amazon Web Servicesのオブジェクトストレージ

**メリット:**
- ✅ **スケーラブル**: 容量無制限
- ✅ **高速**: 高速なアップロード/ダウンロード
- ✅ **信頼性**: 99.999999999% の耐久性
- ✅ **PyTorch統合**: `s3://` URLで直接アクセス可能
- ✅ **バージョニング**: ファイルのバージョン管理

**デメリット:**
- ⚠️ 無料枠は12ヶ月のみ（5GB）
- ⚠️ 従量課金制（使った分だけ課金）
- ⚠️ 設定が複雑

**使用例:**
```python
# インストール
pip install boto3

# アップロード
import boto3
s3 = boto3.client('s3')
s3.upload_file('outputs/best.pth', 'my-bucket', 'models/best.pth')

# ダウンロード
s3.download_file('my-bucket', 'models/best.pth', 'best.pth')
```

**料金:**
- 無料枠（12ヶ月）: 5GB + 20,000リクエスト
- その後: $0.023/GB/月 + データ転送料

**おすすめ度: ⭐⭐⭐⭐**（大規模プロジェクト向け）

---

### 4. 💧 Dropbox

**概要:**
シンプルなクラウドストレージ

**メリット:**
- ✅ **使いやすい**: シンプルなUI
- ✅ **ローカル同期**: デスクトップアプリで自動同期
- ✅ **共有簡単**: リンク共有

**デメリット:**
- ⚠️ 無料は2GBのみ
- ⚠️ 有料プランが高い
- ⚠️ APIが複雑

**料金:**
- 無料: 2GB
- Plus: $11.99/月（2TB）
- Professional: $19.99/月（3TB）

**おすすめ度: ⭐⭐⭐**

---

### 5. 🗄️ Azure Blob Storage

**概要:**
Microsoft Azureのオブジェクトストレージ

**メリット:**
- ✅ **信頼性**: エンタープライズグレード
- ✅ **PyTorch統合**: Azure ML統合
- ✅ **スケーラブル**: 容量無制限

**デメリット:**
- ⚠️ 無料枠は12ヶ月のみ（5GB）
- ⚠️ 従量課金
- ⚠️ S3より若干高い

**料金:**
- 無料枠（12ヶ月）: 5GB
- その後: $0.0184/GB/月〜

**おすすめ度: ⭐⭐⭐**（Azure利用者向け）

---

### 6. 🐙 Git LFS（推奨 - Git統合）

**概要:**
Git Large File Storage - Gitで大きなファイルを管理

**メリット:**
- ✅ **Git統合**: 通常のGitワークフローで使える
- ✅ **バージョン管理**: Gitでモデルをバージョン管理
- ✅ **GitHub統合**: GitHubで直接使える
- ✅ **無料枠**: GitHub 1GB/月、GitLab 10GB

**デメリット:**
- ⚠️ 容量制限: GitHub 1GB、GitLab 10GB
- ⚠️ 帯域幅制限: GitHub 1GB/月

**使用例:**
```bash
# インストール
git lfs install

# 追跡設定
git lfs track "*.pth"
git lfs track "*.h5"
git add .gitattributes

# 通常通りコミット
git add outputs/best.pth
git commit -m "Add trained model"
git push
```

**料金:**
- GitHub: 無料1GB、データパック50GB $5/月
- GitLab: 無料10GB

**おすすめ度: ⭐⭐⭐⭐⭐**（小〜中規模プロジェクト）

---

## 🎯 プロジェクトに最適な選択

### このプロジェクト（fontgen-ai）の場合

#### データサイズ推定
```
前処理データ: 約10MB
ベストモデル: 約80-100MB
学習ログ: 約10MB
合計: 約100-120MB
```

#### 推奨オプション（優先順）

**1位: Git LFS** ⭐⭐⭐⭐⭐
- **理由**:
  - 100MBならギリギリ収まる
  - Gitワークフローと統合
  - 無料で使える
  - セッション引き継ぎが最も簡単

**2位: Hugging Face Hub** ⭐⭐⭐⭐⭐
- **理由**:
  - ML特化で使いやすい
  - 公開リポジトリなら無料無制限
  - PyTorchと親和性が高い
  - コミュニティで共有しやすい

**3位: Google Drive** ⭐⭐⭐⭐
- **理由**:
  - 15GB無料枠で十分
  - 使いやすい
  - Google Colabとの連携が簡単

---

## 🛠️ 実装例

### Git LFS実装

```bash
# 1. Git LFSインストール
git lfs install

# 2. 大きなファイルを追跡
git lfs track "*.pth"
git lfs track "outputs/*/checkpoints/*.pth"
git lfs track "*.png"
git add .gitattributes

# 3. 通常通り使用
git add outputs/hiragana_kanji/checkpoints/best.pth
git commit -m "Add best model"
git push

# 4. 次のセッションで
git checkout <branch>
# ファイルが自動的にダウンロードされる
```

### Hugging Face Hub実装

```python
# setup_hf_storage.py
from huggingface_hub import HfApi, hf_hub_download
import os

class HFStorage:
    def __init__(self, repo_id, token):
        self.api = HfApi()
        self.repo_id = repo_id
        self.token = token

    def upload_model(self, local_path, remote_path):
        """モデルをアップロード"""
        self.api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=self.repo_id,
            token=self.token
        )
        print(f"✅ Uploaded: {local_path} -> {remote_path}")

    def download_model(self, remote_path, local_path):
        """モデルをダウンロード"""
        downloaded = hf_hub_download(
            repo_id=self.repo_id,
            filename=remote_path,
            cache_dir=".cache"
        )
        os.rename(downloaded, local_path)
        print(f"✅ Downloaded: {remote_path} -> {local_path}")

# 使用例
storage = HFStorage(
    repo_id="username/fontgen-ai",
    token=os.environ["HF_TOKEN"]
)

# 学習後にアップロード
storage.upload_model(
    "outputs/best.pth",
    "models/hiragana_kanji_best.pth"
)

# 次のセッションでダウンロード
storage.download_model(
    "models/hiragana_kanji_best.pth",
    "outputs/best.pth"
)
```

---

## 💡 使い分けガイド

| 用途 | 推奨サービス | 理由 |
|------|------------|------|
| 個人プロジェクト（<100MB） | Git LFS | 無料、統合が簡単 |
| 個人プロジェクト（>100MB） | Hugging Face Hub | ML特化、無料 |
| チーム開発 | AWS S3 / Hugging Face | スケーラブル、権限管理 |
| 実験的プロジェクト | Google Drive | 手軽、無料枠十分 |
| プロダクション | AWS S3 / Azure | 信頼性、SLA保証 |
| 公開モデル | Hugging Face Hub | コミュニティ共有 |

---

## 📝 このプロジェクトの推奨実装

### フェーズ1: Git LFS（現在〜Phase 3）
```bash
# .gitattributesを作成
git lfs track "outputs/*/checkpoints/best.pth"
git lfs track "outputs/*/checkpoints/last.pth"
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

### フェーズ2: Hugging Face Hub（Phase 4以降）
モデルが大きくなったり、公開したくなったら移行

```python
# upload_to_hf.py
from huggingface_hub import HfApi

api = HfApi()
api.upload_folder(
    folder_path="outputs/hiragana_kanji/checkpoints",
    repo_id="Naoking55/fontgen-ai",
    path_in_repo="models/v1",
)
```

---

## ⚠️ セキュリティ注意事項

### トークン管理
```bash
# 環境変数で管理（絶対にコミットしない！）
export HF_TOKEN="your_token_here"
export AWS_ACCESS_KEY_ID="your_key"
export AWS_SECRET_ACCESS_KEY="your_secret"

# .gitignoreに追加
echo ".env" >> .gitignore
echo "*.token" >> .gitignore
```

### プライベートデータ
- 個人的なプロジェクト → プライベートリポジトリ
- 公開しても良いモデル → 公開リポジトリ
- 商用プロジェクト → プライベート + アクセス制御

---

## 🎓 まとめ

**このプロジェクトの推奨:**
1. **現在**: Git LFS（モデルサイズが100MB以下なら）
2. **将来**: Hugging Face Hub（モデルが大きくなったら/公開したくなったら）

**最もシンプル**: Google Drive
**最も専門的**: Hugging Face Hub
**最もスケーラブル**: AWS S3

選択のポイント：
- 💰 予算
- 📏 データサイズ
- 👥 チームの有無
- 🔒 公開/非公開
- 🛠️ 既存インフラ

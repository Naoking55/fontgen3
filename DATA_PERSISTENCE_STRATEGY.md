# データ永続化戦略

**最終更新:** 2025-11-12

---

## 問題の概要

Claude Codeのセッション間でデータを引き継ぐために、Gitリポジトリを活用します。
しかし、機械学習プロジェクトには大量のファイルが含まれるため、戦略的な管理が必要です。

---

## 採用した戦略

### ✅ 選択的Git管理アプローチ

**基本方針:**
- 重要なファイル（モデル、評価結果、前処理データ）をGitで管理
- 中間生成物や大量のログは除外
- GitHubの制限内（100MB/ファイル、リポジトリ全体で1GB推奨）

---

## ファイル管理ポリシー

### 🟢 Gitに含めるファイル

#### モデルファイル
- ✅ `outputs/*/checkpoints/best.pth` - ベストモデル
- ✅ `outputs/*/checkpoints/last.pth` - 最終モデル
- ❌ `outputs/*/checkpoints/epoch_*.pth` - 中間チェックポイント（除外）

#### 前処理データ
- ✅ `data/processed_*/metadata.json` - メタデータ
- ✅ `data/processed_*/*.png` - 前処理済み画像（約10MB）
- ✅ `data/fonts/` - フォントファイル（既に含まれている）

#### 評価結果
- ✅ `outputs/*/evaluation/` - 評価レポート
- ✅ `outputs/*/samples/grid.png` - グリッドサンプル
- ✅ `outputs/*/samples/epoch_final.png` - 最終サンプル
- ❌ `outputs/*/samples/epoch_*.png` - 中間サンプル（除外）

#### ドキュメント
- ✅ すべてのMarkdownファイル
- ✅ 設定ファイル（`config/*.yaml`）

### 🔴 Gitから除外するファイル

#### ログとキャッシュ
- ❌ `outputs/*/logs/` - TensorBoardログ
- ❌ `outputs/*/tensorboard/` - TensorBoard
- ❌ `*.log` - テキストログ
- ❌ `wandb/` - Weights & Biases

#### 中間ファイル
- ❌ `outputs/*/checkpoints/epoch_*.pth` - 中間チェックポイント
- ❌ `outputs/*/samples/epoch_[0-9]*.png` - 中間サンプル画像

---

## 推定データサイズ

### 前処理データ（data/processed_hiragana_kanji/）
```
- 画像数: 258文字 × 5フォント = 1,290画像
- 画像サイズ: 128x128 グレースケール
- PNG圧縮後: 約7KB/画像
- 合計: 約10MB
```
**判定:** ✅ Git管理可能

### モデルファイル（outputs/*/checkpoints/）
```
- best.pth: 約80-100MB（モデルアーキテクチャによる）
- last.pth: 約80-100MB
```
**判定:** ⚠️ GitHub制限ギリギリ（100MB/ファイル）

### 評価結果（outputs/*/evaluation/）
```
- レポート: 数KB
- サンプル画像: 数MB
```
**判定:** ✅ Git管理可能

---

## 実装された.gitignore修正

```gitignore
# PyTorch - 中間チェックポイントのみ除外、ベストと最終は保持
outputs/*/checkpoints/epoch_*.pth
outputs/*/checkpoints/checkpoint_*.pth
*.pt
*.ckpt

# Data - 前処理済みデータは保持
# data/processed/ はコメントアウト（保持）
data/skeleton_db/

# Output - 重要なファイルは保持
outputs/*/logs/
outputs/*/tensorboard/
models/pretrained/

# 画像 - 重要なサンプルのみ保持
outputs/*/samples/*.png
!outputs/*/samples/epoch_final.png
!outputs/*/samples/grid.png
```

---

## セッション引き継ぎワークフロー

### 1. 学習完了時
```bash
# 重要なファイルを確認
ls -lh outputs/hiragana_kanji/checkpoints/best.pth
ls -lh outputs/hiragana_kanji/checkpoints/last.pth

# Gitステータス確認
git status

# コミット
git add -A
git commit -m "ひらがな+漢字学習完了: SSIM 0.XXX"
git push -u origin <branch-name>
```

### 2. 次のセッション開始時
```bash
# ブランチをチェックアウト
git checkout <branch-name>

# データが存在するか確認
ls -la data/processed_hiragana_kanji/
ls -la outputs/hiragana_kanji/checkpoints/

# 学習を再開（必要に応じて）
python cli/train.py --resume outputs/hiragana_kanji/checkpoints/last.pth
```

---

## モデルサイズの対策

### 問題: モデルファイルが100MBを超える場合

#### オプションA: Git LFS（推奨）
```bash
# Git LFS設定
git lfs install
git lfs track "*.pth"
git add .gitattributes
git commit -m "Git LFS設定"

# 通常通りコミット
git add outputs/hiragana_kanji/checkpoints/best.pth
git commit -m "ベストモデルを追加"
git push
```

#### オプションB: モデル圧縮
```python
# チェックポイント保存時に不要な情報を削除
torch.save({
    'model_state_dict': model.state_dict(),
    # optimizer、schedulerは除外
}, 'best.pth')
```

#### オプションC: 外部ストレージ
- Google Drive
- Dropbox
- Amazon S3
- Hugging Face Hub

---

## ベストプラクティス

### ✅ DO
1. **定期的にコミット**: 重要なマイルストーンごとにコミット
2. **明確なメッセージ**: コミットメッセージに学習結果を記載
3. **サイズ確認**: コミット前に`du -sh`でサイズ確認
4. **プッシュ確認**: セッション終了前に必ずプッシュ

### ❌ DON'T
1. **大量の中間ファイルをコミットしない**
2. **未整理のデータをコミットしない**
3. **100MB超えのファイルを直接コミットしない**（LFS使用）
4. **セッション終了時にプッシュし忘れない**

---

## トラブルシューティング

### Q: モデルファイルが100MBを超えてプッシュできない
**A:** Git LFSを使用するか、モデルを分割保存

### Q: リポジトリ全体が大きすぎる
**A:** 古い中間ファイルを削除し、`.gitignore`を再調整

### Q: 前処理データが大きすぎる
**A:** メタデータのみコミットし、画像は毎回生成

### Q: セッション間でデータが消えた
**A:** `.gitignore`で除外されていないか確認し、コミット漏れをチェック

---

## まとめ

この戦略により：
- ✅ セッション間でのデータ引き継ぎが可能
- ✅ 重要なファイルのみGitで管理
- ✅ リポジトリサイズが管理可能
- ✅ 学習の再開が容易

**次のセッションでも確実にデータを引き継げます！**

# TODO リスト - AI フォント生成システム

**最終更新:** 2025-11-12

---

## 🔴 次のセッションでの最優先タスク

### 1. ひらがな+漢字学習の状態確認
```bash
cd /home/user/fontgen3/fontgen-ai
ls -la outputs/
```

- [ ] 学習が完了しているかチェック
- [ ] チェックポイントファイルの存在確認
- [ ] ログファイルの確認

### 2. 学習結果の評価
**学習が完了している場合:**
```bash
python cli/evaluate_quality.py \
  --model outputs/hiragana_kanji/checkpoints/best.pth \
  --data-dir data/processed_hiragana_kanji \
  --output-dir outputs/hiragana_kanji/evaluation
```

**チェック項目:**
- [ ] SSIM値の確認（目標: > 0.65）
- [ ] サンプル画像の品質確認
- [ ] 評価レポートの作成

### 3. 未完了の場合は学習を実行/再開
```bash
python cli/train.py \
  --config config/training_config.hiragana_kanji.yaml \
  --data-dir data/processed_hiragana_kanji \
  --output-dir outputs/hiragana_kanji
```

---

## 🟡 Phase 2 完了後のタスク

### Phase 3: 生成機能実装

#### 3.1 生成エンジン実装
- [ ] `src/generator.py` を作成
  - [ ] `FontGenerator` クラス
  - [ ] `extract_style()` メソッド
  - [ ] `generate_character()` メソッド
  - [ ] `generate_batch()` メソッド

#### 3.2 骨格データベース構築
- [ ] `cli/build_skeleton_db.py` を作成
- [ ] 標準文字の骨格（z_content）を抽出
- [ ] データベースファイルに保存

#### 3.3 ベクトル化モジュール
- [ ] `src/vectorizer.py` を作成
  - [ ] ラスター → ベクトル変換
  - [ ] Potrace統合
  - [ ] ベジェ曲線最適化

#### 3.4 フォントビルダー
- [ ] `src/font_builder.py` を作成
  - [ ] fontTools統合
  - [ ] TTF/OTF出力機能

#### 3.5 生成CLI
- [ ] `cli/generate.py` を実装
  - [ ] スタイル指定
  - [ ] バッチ生成
  - [ ] フォントファイル出力

---

## 📋 現在の進捗（Phase別）

- [x] **Phase 0:** プロジェクトセットアップ
- [x] **Phase 1:** データパイプライン構築
- [ ] **Phase 2:** モデル実装と学習（95%完了）
  - [x] VAEモデル実装
  - [x] 学習ループ実装
  - [x] 複数フォント学習（ひらがな）完了
  - [ ] ひらがな+漢字学習（確認待ち）
- [ ] **Phase 3:** 生成機能実装（0%）
- [ ] **Phase 4:** GUI実装（0%）
- [ ] **Phase 5:** 高度な機能実装（0%）
- [ ] **Phase 6:** 最適化と公開準備（0%）

---

## 📝 メモ

### 完了した主な成果
- ✅ 4フォントでの学習パイプライン構築
- ✅ 実用レベルのモデル（SSIM 0.649）
- ✅ 評価ツール実装（`evaluate_quality.py`, `generate_grid.py`）
- ✅ 複数の設定ファイル整備

### 既知の課題
- ひらがな+漢字学習の完了確認が必要
- Phase 3（生成機能）がまだ未着手

### 次のマイルストーン
**Phase 2完了:** ひらがな+漢字学習で実用レベル（SSIM > 0.65）達成
**Phase 3開始:** 文字生成機能の実装着手

---

**このTODOリストは WORK_CONTEXT.md と併せて確認してください**

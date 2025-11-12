# MPS (Apple Silicon) セットアップガイド

M1/M2 Mac で MPS を使って学習を高速化する方法

## 🚀 クイックスタート

### 方法1: 自動セットアップ（推奨）

```bash
cd fontgen-ai

# セットアップスクリプト実行
bash setup_venv.sh

# 仮想環境を有効化
source venv/bin/activate

# テスト実行
python test_vae_training.py
```

---

### 方法2: 手動セットアップ

```bash
cd fontgen-ai

# 1. 仮想環境作成
python3 -m venv venv

# 2. 仮想環境を有効化
source venv/bin/activate

# 3. PyTorchインストール
pip install --upgrade pip
pip install torch torchvision torchaudio

# 4. 依存関係インストール
pip install -r requirements.txt

# 5. MPS確認
python -c "import torch; print(f'MPS Available: {torch.backends.mps.is_available()}')"
```

**期待される出力:**
```
MPS Available: True
```

---

## ✅ 動作確認

### テスト実行

```bash
# 仮想環境がアクティブな状態で
python test_vae_training.py
```

**正常な出力:**
```
✓ Using MPS (Apple Silicon)  ← これが表示されればOK
📦 データセット作成中...
🤖 モデル作成中...
  Total parameters: 21,406,529
🎓 学習開始...
Epoch 1/5: 100%|████| 16/16 [00:02<00:00, ...]  ← 高速化！
```

**速度比較:**
- CPU: 約27秒/エポック
- MPS: 約2-3秒/エポック（**10倍高速！**）

---

## 🔧 トラブルシューティング

### Q1: `MPS Available: False` と表示される

**原因:** PyTorchのバージョンが古い

**解決策:**
```bash
pip install --upgrade torch torchvision torchaudio
python -c "import torch; print(torch.__version__)"  # 2.0以上を確認
```

---

### Q2: "RuntimeError: MPS backend out of memory"

**原因:** バッチサイズが大きすぎる

**解決策:**
```yaml
# config/training_config.yaml
training:
  batch_size: 32  # 64 → 32 に減らす
```

---

### Q3: 学習が異常に遅い

**確認:**
```bash
# デバイス確認
python -c "from src.device_utils import get_device; print(get_device())"
```

**出力が `cpu` の場合:**
- 仮想環境が有効化されていない可能性
- `source venv/bin/activate` を実行

---

## 📊 性能比較

### M1 Ultra での測定結果

| デバイス | 1エポックの時間 | 学習速度 |
|---------|---------------|---------|
| CPU | 27秒 | 1x |
| MPS | 2-3秒 | **10-12x** |
| CUDA (参考) | 1-2秒 | 15-20x |

**M1 Ultra は CUDA に迫る性能！**

---

## 💡 ベストプラクティス

### 1. 常に仮想環境を使う

```bash
# プロジェクト開始時
cd fontgen-ai
source venv/bin/activate

# 作業終了時
deactivate
```

### 2. 設定でデバイスを指定

```bash
# 自動検出（推奨）
python cli/train.py --device auto ...

# 明示的にMPS指定
python cli/train.py --device mps ...
```

### 3. バッチサイズを調整

M1 Ultra (64GB メモリ) の推奨値:
- **画像サイズ 128x128**: batch_size = 64-128
- **画像サイズ 256x256**: batch_size = 32-64

---

## 🎯 次のステップ

1. **テスト実行** → 動作確認
2. **実データで学習** → 品質確認
3. **Phase 3に進む** → 生成機能実装

---

## 📝 メモ

- PyTorch 2.0以上が必須
- macOS 12.3以上が必須
- Xcode Command Line Tools が必要

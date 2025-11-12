#!/bin/bash
# セットアップスクリプト - MPS対応仮想環境

set -e

echo "================================================"
echo " AI フォント生成システム - セットアップ"
echo "================================================"

# 仮想環境作成
echo ""
echo "📦 仮想環境を作成中..."
python3 -m venv venv

# 仮想環境を有効化
echo ""
echo "🔧 仮想環境を有効化..."
source venv/bin/activate

# pipをアップグレード
echo ""
echo "⬆️  pipをアップグレード中..."
pip install --upgrade pip

# PyTorchインストール
echo ""
echo "🔥 PyTorch (MPS対応) をインストール中..."
pip install torch torchvision torchaudio

# 依存関係インストール
echo ""
echo "📚 依存関係をインストール中..."
pip install -r requirements.txt

# 確認
echo ""
echo "✅ セットアップ完了！"
echo ""
echo "================================================"
echo " デバイス情報"
echo "================================================"
python -c "import torch; print(f'PyTorch Version: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'MPS Available: {torch.backends.mps.is_available()}')"
echo "================================================"
echo ""
echo "次のステップ:"
echo "  1. 仮想環境を有効化:"
echo "     source venv/bin/activate"
echo ""
echo "  2. テスト実行:"
echo "     python test_vae_training.py"
echo ""
echo "  3. 本格学習:"
echo "     python cli/train.py --config config/training_config.minimal.yaml ..."
echo ""

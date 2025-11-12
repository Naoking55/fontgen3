#!/bin/bash
# プロジェクト状態確認スクリプト

set -e

echo "=================================="
echo "fontgen3 プロジェクト状態チェック"
echo "=================================="
echo ""

# カラー定義
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 1. Python環境
echo "1. Python環境"
echo "  Python バージョン: $(python3 --version)"
if command -v pip &> /dev/null; then
    echo -e "  ${GREEN}✓${NC} pip 利用可能"
else
    echo -e "  ${RED}✗${NC} pip が見つかりません"
fi
echo ""

# 2. PyTorch
echo "2. PyTorch"
if python3 -c "import torch" 2>/dev/null; then
    TORCH_VERSION=$(python3 -c "import torch; print(torch.__version__)")
    echo -e "  ${GREEN}✓${NC} PyTorch インストール済み (v${TORCH_VERSION})"
    CUDA_AVAILABLE=$(python3 -c "import torch; print(torch.cuda.is_available())")
    if [ "$CUDA_AVAILABLE" = "True" ]; then
        GPU_NAME=$(python3 -c "import torch; print(torch.cuda.get_device_name(0))")
        echo -e "  ${GREEN}✓${NC} CUDA 利用可能: ${GPU_NAME}"
    else
        echo -e "  ${YELLOW}⚠${NC}  CUDA 利用不可（CPUモード）"
    fi
else
    echo -e "  ${RED}✗${NC} PyTorch がインストールされていません"
    echo "     インストール: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
fi
echo ""

# 3. Rclone
echo "3. Rclone（Google Drive同期用）"
if command -v rclone &> /dev/null; then
    RCLONE_VERSION=$(rclone version | head -1 | awk '{print $2}')
    echo -e "  ${GREEN}✓${NC} Rclone インストール済み (v${RCLONE_VERSION})"

    if rclone listremotes | grep -q "^gdrive:$"; then
        echo -e "  ${GREEN}✓${NC} 'gdrive' リモート設定済み"
    else
        echo -e "  ${YELLOW}⚠${NC}  'gdrive' リモートが未設定"
        echo "     設定: rclone config"
    fi
else
    echo -e "  ${RED}✗${NC} Rclone がインストールされていません"
    echo "     インストール: curl https://rclone.org/install.sh | sudo bash"
fi
echo ""

# 4. データセット
echo "4. データセット"
DATA_DIR="fontgen-ai/data/processed_hiragana_kanji"
if [ -d "$DATA_DIR" ]; then
    echo -e "  ${GREEN}✓${NC} データディレクトリ存在: $DATA_DIR"

    if [ -f "$DATA_DIR/metadata.json" ]; then
        NUM_CHARS=$(python3 -c "import json; data=json.load(open('$DATA_DIR/metadata.json')); print(len(data['characters']))" 2>/dev/null || echo "?")
        NUM_FONTS=$(python3 -c "import json; data=json.load(open('$DATA_DIR/metadata.json')); print(len(data['fonts']))" 2>/dev/null || echo "?")
        echo "     文字数: ${NUM_CHARS}, フォント数: ${NUM_FONTS}"
    fi

    # データファイル数
    TRAIN_FILES=$(find "$DATA_DIR/train" -name "*.pt" 2>/dev/null | wc -l)
    VAL_FILES=$(find "$DATA_DIR/val" -name "*.pt" 2>/dev/null | wc -l)

    if [ "$TRAIN_FILES" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} 学習データ: ${TRAIN_FILES} ファイル"
    else
        echo -e "  ${YELLOW}⚠${NC}  学習データが見つかりません"
    fi

    if [ "$VAL_FILES" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} 検証データ: ${VAL_FILES} ファイル"
    else
        echo -e "  ${YELLOW}⚠${NC}  検証データが見つかりません"
    fi
else
    echo -e "  ${RED}✗${NC} データディレクトリが見つかりません: $DATA_DIR"
    echo "     ダウンロード: ./rclone_sync.sh download-all"
fi
echo ""

# 5. チェックポイント
echo "5. チェックポイント"
CHECKPOINT_DIR="outputs/hiragana_kanji/checkpoints"
if [ -d "$CHECKPOINT_DIR" ]; then
    echo -e "  ${GREEN}✓${NC} チェックポイントディレクトリ存在"

    if [ -f "$CHECKPOINT_DIR/model_final.pt" ]; then
        SIZE=$(du -h "$CHECKPOINT_DIR/model_final.pt" | cut -f1)
        MTIME=$(stat -c %y "$CHECKPOINT_DIR/model_final.pt" 2>/dev/null | cut -d' ' -f1 || echo "?")
        echo -e "  ${GREEN}✓${NC} model_final.pt: ${SIZE} (${MTIME})"
    else
        echo -e "  ${YELLOW}⚠${NC}  model_final.pt が見つかりません"
    fi

    if [ -f "$CHECKPOINT_DIR/model_best.pt" ]; then
        SIZE=$(du -h "$CHECKPOINT_DIR/model_best.pt" | cut -f1)
        echo -e "  ${GREEN}✓${NC} model_best.pt: ${SIZE}"
    fi

    NUM_CHECKPOINTS=$(find "$CHECKPOINT_DIR" -name "checkpoint_epoch_*.pt" 2>/dev/null | wc -l)
    if [ "$NUM_CHECKPOINTS" -gt 0 ]; then
        echo "     エポックチェックポイント: ${NUM_CHECKPOINTS} 個"
    fi
else
    echo -e "  ${RED}✗${NC} チェックポイントディレクトリが見つかりません"
    echo "     ダウンロード: ./rclone_sync.sh download-checkpoints"
fi
echo ""

# 6. フォントファイル
echo "6. フォントファイル"
FONT_DIR="fontgen-ai/data/fonts"
if [ -d "$FONT_DIR" ]; then
    NUM_FONTS=$(find "$FONT_DIR" -type f \( -iname "*.ttf" -o -iname "*.otf" -o -iname "*.ttc" \) | wc -l)
    if [ "$NUM_FONTS" -gt 0 ]; then
        echo -e "  ${GREEN}✓${NC} ${NUM_FONTS} 個のフォントファイル"
        find "$FONT_DIR" -type f \( -iname "*.ttf" -o -iname "*.otf" -o -iname "*.ttc" \) -exec basename {} \; | sed 's/^/     - /'
    else
        echo -e "  ${YELLOW}⚠${NC}  フォントファイルが見つかりません"
    fi
else
    echo -e "  ${RED}✗${NC} フォントディレクトリが見つかりません"
fi
echo ""

# まとめ
echo "=================================="
echo "次のステップ"
echo "=================================="

HAS_TORCH=$(python3 -c "import torch" 2>/dev/null && echo "yes" || echo "no")
HAS_RCLONE=$(command -v rclone &> /dev/null && echo "yes" || echo "no")
HAS_DATA=$([ "$TRAIN_FILES" -gt 0 ] && echo "yes" || echo "no")
HAS_CHECKPOINT=$([ -f "$CHECKPOINT_DIR/model_final.pt" ] && echo "yes" || echo "no")

if [ "$HAS_TORCH" = "no" ]; then
    echo "1. PyTorchをインストール:"
    echo "   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118"
    echo ""
fi

if [ "$HAS_RCLONE" = "no" ]; then
    echo "2. Rcloneをインストール:"
    echo "   curl https://rclone.org/install.sh | sudo bash"
    echo "   rclone config  # Google Drive設定"
    echo ""
fi

if [ "$HAS_DATA" = "no" ]; then
    echo "3. データをダウンロード:"
    echo "   ./rclone_sync.sh download-all"
    echo ""
fi

if [ "$HAS_CHECKPOINT" = "no" ]; then
    echo "4. チェックポイントをダウンロード:"
    echo "   ./rclone_sync.sh download-checkpoints"
    echo ""
fi

if [ "$HAS_TORCH" = "yes" ] && [ "$HAS_DATA" = "yes" ] && [ "$HAS_CHECKPOINT" = "yes" ]; then
    echo -e "${GREEN}✓ 学習を再開できます！${NC}"
    echo ""
    echo "学習再開コマンド:"
    echo "  cd fontgen-ai"
    echo "  python cli/train.py \\"
    echo "    --config config/training_config.hiragana_kanji.yaml \\"
    echo "    --data-dir ./data/processed_hiragana_kanji \\"
    echo "    --output-dir ../outputs/hiragana_kanji \\"
    echo "    --resume ../outputs/hiragana_kanji/checkpoints/model_final.pt"
else
    echo -e "${YELLOW}上記の手順を完了してから学習を再開してください${NC}"
    echo ""
    echo "詳細は TRAINING_RESUME_GUIDE.md を参照してください"
fi
echo ""

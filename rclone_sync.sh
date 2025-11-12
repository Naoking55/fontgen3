#!/bin/bash

# Rclone Sync Utility for FontGen AI Project
# Usage: ./rclone_sync.sh [command]

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
REMOTE="gdrive:fontgen-ai"
LOCAL_BASE="outputs/hiragana_kanji"
RCLONE_OPTIONS="-P --transfers 4"

# Functions
print_usage() {
    echo "Usage: $0 [command]"
    echo ""
    echo "Commands:"
    echo "  upload-checkpoints     - チェックポイントをGoogle Driveにアップロード"
    echo "  download-checkpoints   - チェックポイントをGoogle Driveからダウンロード"
    echo "  sync-to-cloud          - 全データをクラウドに同期（ローカル → クラウド）"
    echo "  sync-from-cloud        - 全データをクラウドから同期（クラウド → ローカル）"
    echo "  upload-all             - 全データをアップロード（既存ファイルは保持）"
    echo "  download-all           - 全データをダウンロード（既存ファイルは保持）"
    echo "  list-remote            - Google Drive上のファイル一覧を表示"
    echo "  test-connection        - Google Driveへの接続をテスト"
    echo ""
    echo "Examples:"
    echo "  $0 upload-checkpoints"
    echo "  $0 sync-to-cloud"
}

check_rclone() {
    if ! command -v rclone &> /dev/null; then
        echo -e "${RED}Error: rclone is not installed${NC}"
        echo "Install with: curl https://rclone.org/install.sh | sudo bash"
        exit 1
    fi
}

check_remote() {
    if ! rclone listremotes | grep -q "^gdrive:$"; then
        echo -e "${RED}Error: 'gdrive' remote not configured${NC}"
        echo "Run: rclone config"
        echo "See RCLONE_SETUP.md for detailed instructions"
        exit 1
    fi
}

test_connection() {
    echo -e "${YELLOW}Testing connection to Google Drive...${NC}"
    if rclone lsd gdrive: &> /dev/null; then
        echo -e "${GREEN}✓ Connection successful${NC}"
        return 0
    else
        echo -e "${RED}✗ Connection failed${NC}"
        echo "Try: rclone config reconnect gdrive:"
        return 1
    fi
}

upload_checkpoints() {
    echo -e "${YELLOW}Uploading checkpoints to Google Drive...${NC}"
    if [ ! -d "$LOCAL_BASE/checkpoints" ]; then
        echo -e "${RED}Error: Local checkpoints directory not found: $LOCAL_BASE/checkpoints${NC}"
        exit 1
    fi
    rclone copy "$LOCAL_BASE/checkpoints/" "$REMOTE/checkpoints/" $RCLONE_OPTIONS
    echo -e "${GREEN}✓ Checkpoints uploaded successfully${NC}"
}

download_checkpoints() {
    echo -e "${YELLOW}Downloading checkpoints from Google Drive...${NC}"
    mkdir -p "$LOCAL_BASE/checkpoints"
    rclone copy "$REMOTE/checkpoints/" "$LOCAL_BASE/checkpoints/" $RCLONE_OPTIONS
    echo -e "${GREEN}✓ Checkpoints downloaded successfully${NC}"
}

sync_to_cloud() {
    echo -e "${YELLOW}WARNING: This will make Google Drive match local files exactly${NC}"
    echo -e "${YELLOW}Files on Google Drive that don't exist locally will be DELETED${NC}"
    read -p "Continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled"
        exit 0
    fi

    echo -e "${YELLOW}Syncing to Google Drive...${NC}"
    if [ ! -d "$LOCAL_BASE" ]; then
        echo -e "${RED}Error: Local directory not found: $LOCAL_BASE${NC}"
        exit 1
    fi
    rclone sync "$LOCAL_BASE/" "$REMOTE/hiragana_kanji/" $RCLONE_OPTIONS
    echo -e "${GREEN}✓ Synced to cloud successfully${NC}"
}

sync_from_cloud() {
    echo -e "${YELLOW}WARNING: This will make local files match Google Drive exactly${NC}"
    echo -e "${YELLOW}Local files that don't exist on Google Drive will be DELETED${NC}"
    read -p "Continue? (yes/no): " confirm
    if [ "$confirm" != "yes" ]; then
        echo "Cancelled"
        exit 0
    fi

    echo -e "${YELLOW}Syncing from Google Drive...${NC}"
    mkdir -p "$LOCAL_BASE"
    rclone sync "$REMOTE/hiragana_kanji/" "$LOCAL_BASE/" $RCLONE_OPTIONS
    echo -e "${GREEN}✓ Synced from cloud successfully${NC}"
}

upload_all() {
    echo -e "${YELLOW}Uploading all data to Google Drive...${NC}"
    if [ ! -d "$LOCAL_BASE" ]; then
        echo -e "${RED}Error: Local directory not found: $LOCAL_BASE${NC}"
        exit 1
    fi
    rclone copy "$LOCAL_BASE/" "$REMOTE/hiragana_kanji/" $RCLONE_OPTIONS
    echo -e "${GREEN}✓ All data uploaded successfully${NC}"
}

download_all() {
    echo -e "${YELLOW}Downloading all data from Google Drive...${NC}"
    mkdir -p "$LOCAL_BASE"
    rclone copy "$REMOTE/hiragana_kanji/" "$LOCAL_BASE/" $RCLONE_OPTIONS
    echo -e "${GREEN}✓ All data downloaded successfully${NC}"
}

list_remote() {
    echo -e "${YELLOW}Files on Google Drive:${NC}"
    rclone tree "$REMOTE" --dirs-only -L 3
}

# Main
check_rclone
check_remote

case "${1:-}" in
    upload-checkpoints)
        test_connection && upload_checkpoints
        ;;
    download-checkpoints)
        test_connection && download_checkpoints
        ;;
    sync-to-cloud)
        test_connection && sync_to_cloud
        ;;
    sync-from-cloud)
        test_connection && sync_from_cloud
        ;;
    upload-all)
        test_connection && upload_all
        ;;
    download-all)
        test_connection && download_all
        ;;
    list-remote)
        test_connection && list_remote
        ;;
    test-connection)
        test_connection
        ;;
    *)
        print_usage
        exit 1
        ;;
esac

#!/bin/bash
# ============================================================
# sync_from_autodl.sh
# Downloads trained model logs from AutoDL to local machine
# Usage: ./sync_from_autodl.sh
# ============================================================

# --- FILL THESE IN (same as sync_to_autodl.sh) ---
AUTODL_HOST="connect.cqa1.seetacloud.com"
AUTODL_PORT="37134"
# --------------------------------------------------

# Change this line in sync_from_autodl.sh
REMOTE_LOGS="/root/autodl-tmp/logs/"
LOCAL_LOGS="$(cd "$(dirname "$0")" && pwd)/logs/"

echo "=============================="
echo " Downloading logs from AutoDL"
echo " From : root@$AUTODL_HOST:$REMOTE_LOGS"
echo " To   : $LOCAL_LOGS"
echo "=============================="

rsync -avz --progress \
  --exclude="*.tmp" \
  --exclude="__pycache__/" \
  --exclude="*.pyc" \
  -e "ssh -p $AUTODL_PORT" \
  root@$AUTODL_HOST:$REMOTE_LOGS \
  "$LOCAL_LOGS"

echo ""
echo "✅ Download complete!"
echo "Models saved to: $LOCAL_LOGS"
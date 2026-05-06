#!/bin/bash
# ============================================================
# sync_to_autodl.sh
# Syncs biped project code to AutoDL instance
# Usage: ./sync_to_autodl.sh
# ============================================================

# --- FILL THESE IN FROM AUTODL DASHBOARD ---
AUTODL_HOST="connect.cqa1.seetacloud.com"
AUTODL_PORT="37134"
# -------------------------------------------

LOCAL_DIR="$(cd "$(dirname "$0")" && pwd)/"   # always runs from project root
REMOTE_DIR="/root/autodl-tmp/biped/"

echo "=============================="
echo " Syncing biped → AutoDL"
echo " From : $LOCAL_DIR"
echo " To   : root@$AUTODL_HOST:$REMOTE_DIR"
echo "=============================="

rsync -avz --progress \
  --exclude '__pycache__/' \
  --exclude '*.pyc' \
  --exclude '*.egg-info/' \
  --exclude 'logs/' \
  --exclude 'outputs/' \
  --exclude 'sim2sim_mujoco/' \
  --exclude 'behaviour_cloning/' \
  --exclude 'MUJOCO_LOG.TXT' \
  --exclude '.git/' \
  --exclude '*.mp4' \
  --exclude '*.png' \
  --exclude '*.vscode/' \
  --exclude '*.dockerignore/' \
  --exclude '*.md/' \
  -e "ssh -p $AUTODL_PORT" \
  "$LOCAL_DIR" \
  root@$AUTODL_HOST:$REMOTE_DIR

echo ""
echo "✅ Sy
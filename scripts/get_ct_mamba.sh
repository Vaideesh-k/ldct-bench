#!/usr/bin/env bash
# Download the official CT-Mamba code into third_party/CT-Mamba (git-ignored).
#
#     bash scripts/get_ct_mamba.sh
#
# Pinned to one commit, so everyone runs exactly the code that
# ldct/models/ct_mamba.py was written against (its line references match it).
set -euo pipefail

URL=https://github.com/linxuan-li/CT-Mamba
COMMIT=89227e123d0dba6c9551417a947e4dfece0de1b4
cd "$(dirname "$0")/.."   # repo root
DEST=third_party/CT-Mamba

if [ ! -d "$DEST/.git" ]; then
  mkdir -p third_party
  git clone "$URL" "$DEST"
fi
git -C "$DEST" fetch --quiet origin || echo "(offline: using the copy already downloaded)"
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$COMMIT"
echo "CT-Mamba ready in $DEST at commit $(git -C "$DEST" rev-parse --short HEAD)"

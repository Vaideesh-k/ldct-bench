#!/usr/bin/env bash
# Download the official CTformer code into third_party/CTformer (git-ignored).
#
#     bash scripts/get_ctformer.sh
#
# Pinned to one commit, so everyone runs exactly the code that
# ldct/models/ctformer.py was written against (its line references match it).
set -euo pipefail

URL=https://github.com/wdayang/CTformer
COMMIT=3c62f58f227f7074b86ba4205f8127fa0b2ed4e9   # last commit on main, 4 Apr 2022
cd "$(dirname "$0")/.."   # repo root
DEST=third_party/CTformer

if [ ! -d "$DEST/.git" ]; then
  mkdir -p third_party
  git clone "$URL" "$DEST"
fi
git -C "$DEST" fetch --quiet origin || echo "(offline: using the copy already downloaded)"
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$COMMIT"
echo "CTformer ready in $DEST at commit $(git -C "$DEST" rev-parse --short HEAD)"

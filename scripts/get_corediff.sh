#!/usr/bin/env bash
# Download the official CoreDiff code into third_party/CoreDiff (git-ignored).
#
#     bash scripts/get_corediff.sh
#
# Pinned to one commit, so everyone runs exactly the code that
# ldct/models/corediff.py was written against (its line references match it).
set -euo pipefail

URL=https://github.com/qgao21/CoreDiff
COMMIT=9bc31dc1d6184a880bd6334db34c466569a3642a   # last commit on main, 22 Jan 2025
cd "$(dirname "$0")/.."   # repo root
DEST=third_party/CoreDiff

if [ ! -d "$DEST/.git" ]; then
  mkdir -p third_party
  git clone "$URL" "$DEST"
fi
git -C "$DEST" fetch --quiet origin || echo "(offline: using the copy already downloaded)"
git -C "$DEST" -c advice.detachedHead=false checkout --quiet "$COMMIT"
echo "CoreDiff ready in $DEST at commit $(git -C "$DEST" rev-parse --short HEAD)"

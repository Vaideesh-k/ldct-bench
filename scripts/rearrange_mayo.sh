#!/usr/bin/env bash
# Rearrange the unzipped Mayo 2016 files into one folder per patient.
#
#     bash scripts/rearrange_mayo.sh ~/datasets/mayo_raw
#
# The zips unpack as                         prep_mayo.py wants
#   <raw>/quarter_1mm/L067/quarter_1mm/*.IMA   ->  <raw>/L067/quarter_1mm/*.IMA
#   <raw>/full_1mm/L067/full_1mm/*.IMA         ->  <raw>/L067/full_1mm/*.IMA
# Works for any number of patients. Folders are moved, not copied (no extra
# disk space), and running it twice is harmless.
set -euo pipefail

RAW=${1:?usage: bash scripts/rearrange_mayo.sh <mayo_raw_dir>}
RAW=${RAW%/}   # drop a trailing slash
moved=0

for kind in quarter_1mm full_1mm; do
  if [ ! -d "$RAW/$kind" ]; then
    echo "skip: no $RAW/$kind folder"
    continue
  fi
  for pdir in "$RAW/$kind"/L*; do
    [ -d "$pdir" ] || continue            # no patients at all (the pattern did not match)
    p=$(basename "$pdir")
    src="$pdir/$kind"
    dst="$RAW/$p/$kind"
    if [ -e "$dst" ]; then
      echo "skip: $dst already exists"
      continue
    fi
    if [ ! -d "$src" ]; then
      echo "WARNING: expected $src, found: $(ls "$pdir" | head -3 | tr '\n' ' ')"
      continue
    fi
    mkdir -p "$RAW/$p"
    mv "$src" "$dst"
    rmdir "$pdir" 2>/dev/null || true     # remove the now-empty folder
    moved=$((moved + 1))
  done
  rmdir "$RAW/$kind" 2>/dev/null || true
done

echo "Moved $moved folders."
echo "Patients in $RAW:"
for p in "$RAW"/L*; do
  [ -d "$p" ] && echo "  $(basename "$p"): $(ls "$p" | tr '\n' ' ')"
done
exit 0

"""Make the LOCKED patient split of LiTS (Kaggle andrewmvd/liver-tumor-segmentation,
part 1 = volumes 0-50) and write it to configs/lits_split.json, which is committed.

    python scripts/make_lits_split.py                 # writes the file (refuses to overwrite)
    python scripts/make_lits_split.py --check         # re-derives it and compares

RULE: the test patients are locked. They are never used for training, validation,
tuning or choosing checkpoints, of the denoisers or of the tumour finder.
The split is by patient (a patient's slices are all train or all test), drawn with
a fixed seed so it can be re-derived, and the file is the single source of truth.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from ldct.config import ROOT

OUT = ROOT / "configs" / "lits_split.json"
PART1 = list(range(0, 51))      # volume-0.nii ... volume-50.nii live in part 1
SEED = 0
TEST_SHARE = 0.2                # 20 % of patients -> 10 of 51


def make_split(patients=PART1, seed=SEED, test_share=TEST_SHARE):
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(patients))
    n_test = int(round(test_share * len(patients)))
    test = sorted(int(patients[i]) for i in order[:n_test])
    train = sorted(int(patients[i]) for i in order[n_test:])
    return {
        "source": "Kaggle andrewmvd/liver-tumor-segmentation (LiTS part 1: volume-0 .. volume-50)",
        "rule": "split by patient with numpy RandomState(seed).permutation; test patients are LOCKED",
        "seed": seed, "test_share": test_share,
        "train": train, "test": test,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare the saved file with a fresh derivation")
    ap.add_argument("--force", action="store_true", help="overwrite an existing file (changes the locked split!)")
    args = ap.parse_args()
    split = make_split()
    if args.check:
        with open(OUT) as f:
            saved = json.load(f)
        same = saved["train"] == split["train"] and saved["test"] == split["test"]
        print(f"{OUT.name}: {len(saved['train'])} train, {len(saved['test'])} test, matches seed {SEED}: {same}")
        sys.exit(0 if same else 1)
    if OUT.exists() and not args.force:
        raise SystemExit(f"{OUT} exists; it is the locked split. Use --force only if you really mean to change it.")
    with open(OUT, "w") as f:
        json.dump(split, f, indent=1)
    print(f"wrote {OUT}: {len(split['train'])} train, {len(split['test'])} test: {split['test']}")


if __name__ == "__main__":
    main()

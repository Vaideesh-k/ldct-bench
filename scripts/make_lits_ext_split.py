"""Step 6: choose the EXTENDED LiTS test patients from the mask scan, by a rule fixed before any
image of these patients is converted or scored (results/lits_ext_scan.json from
scripts/scan_lits_ext.py; masks only).

    python scripts/make_lits_ext_split.py --scan results/lits_ext_scan.json --out configs/lits_ext_split.json

RULE: candidates = LiTS patients 51-130 (never used to train anything: CT-Mamba/CTformer/
CoreDiff saw only Mayo, nnU-Net and the refiner/control only the part-1 train split) with at
least one tumour and at most --max-slices liver slices (disk and time). Order: most tumours
under 10 mm first, then most tumours, then patient number. Add patients in that order until
the 34 tumours of the existing 10 test patients plus the new ones reach --target (100).
The extended test set = existing 10 locked patients + these. The file is committed before
any scoring.
"""
import argparse
import json


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--scan", default="results/lits_ext_scan.json")
    ap.add_argument("--out", default="configs/lits_ext_split.json")
    ap.add_argument("--existing-tumours", type=int, default=34)
    ap.add_argument("--target", type=int, default=100)
    ap.add_argument("--max-slices", type=int, default=400)
    args = ap.parse_args()
    scan = json.load(open(args.scan))
    cand = [(int(k), v) for k, v in scan.items() if "tumours" in v and v["tumours"] > 0 and 51 <= int(k) <= 130]
    too_big = sorted(k for k, v in cand if v["slices"] > args.max_slices)
    cand = [(k, v) for k, v in cand if v["slices"] <= args.max_slices]
    cand.sort(key=lambda kv: (-kv[1]["small"], -kv[1]["tumours"], kv[0]))
    chosen, total, small, slices = [], args.existing_tumours, 0, 0
    for k, v in cand:
        if total >= args.target:
            break
        chosen.append(k); total += v["tumours"]; small += v["small"]; slices += v["slices"]
    out = {"source": "LiTS part 2 volumes (andrewmvd/liver-tumor-segmentation-part-2) + masks from part 1",
           "rule": __doc__.split("RULE: ")[1].split("The file is")[0].strip().replace("\n", " "),
           "existing_test": "configs/lits_split.json part test (10 patients, 34 tumours)",
           "train": [], "test": sorted(chosen),
           "new_tumours": total - args.existing_tumours, "new_small_tumours": small, "new_slices": slices,
           "excluded_too_many_slices": too_big,
           "scan_patients": len(scan), "candidates_with_tumour": len(cand) + len(too_big)}
    json.dump(out, open(args.out, "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "rule"}, indent=1))


if __name__ == "__main__":
    main()

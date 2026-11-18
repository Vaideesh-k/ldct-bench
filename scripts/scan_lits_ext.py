"""Step 6, part 1: count the tumours of every LiTS part-2 patient from its MASK ONLY, so the
extended test set can be chosen (and committed) before anything is scored.

    python scripts/scan_lits_ext.py --patients 51-130 --out results/lits_ext_scan.json

Masks (segmentation-<k>.nii) of all 131 LiTS patients are in the part-1 archive; they are
read one at a time straight from the remote archive (scripts/remote_zip.py), scanned and
deleted. Tumours are counted exactly as scripts/detect_eval.py counts ground-truth lesions:
3D components of label 2 with 26-connectivity, size = equivalent-sphere diameter. The slice
count is the liver range +- 5 slices, i.e. what scripts/make_lits_lowdose.py would keep.
No image, model or prediction is involved. Resumable: patients already in --out are skipped.
"""
import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from detect_eval import STRUCT, equivalent_diameter_mm   # noqa: E402
from make_lits_lowdose import LABEL_LIVER, liver_slice_range, load_canonical   # noqa: E402
from remote_zip import extract, open_archive   # noqa: E402

MASKS = "andrewmvd/liver-tumor-segmentation"


def scan(mask_xyz, zooms, small_mm=10.0, margin=5):
    tumour = mask_xyz == 2
    lab, n = ndimage.label(tumour, STRUCT)
    sizes = np.bincount(lab.ravel())[1:] if n else np.array([], int)
    diam = [equivalent_diameter_mm(s, zooms) for s in sizes]
    z0, z1 = liver_slice_range(mask_xyz, margin) if (mask_xyz >= LABEL_LIVER).any() else (0, -1)
    return {"tumours": int(n), "small": int(sum(d < small_mm for d in diam)),
            "diameters_mm": [round(d, 2) for d in sorted(diam)],
            "slices": int(z1 - z0 + 1), "spacing_mm": [round(z, 4) for z in zooms]}


def parse_range(items):
    out = []
    for it in items:
        a, _, b = it.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--patients", nargs="+", default=["51-130"])
    ap.add_argument("--out", default="results/lits_ext_scan.json")
    ap.add_argument("--tmp", default="data/lits_raw_tmp")
    args = ap.parse_args()
    res = json.load(open(args.out)) if os.path.exists(args.out) else {}
    zf, members = None, None
    for k in parse_range(args.patients):
        if str(k) in res:
            continue
        for attempt in range(3):
            try:
                if zf is None:
                    zf, _ = open_archive(MASKS)
                    members = {os.path.basename(n): n for n in zf.namelist()}
                t0 = time.time()
                path = extract(zf, members[f"segmentation-{k}.nii"], args.tmp)
                msk, zooms = load_canonical(path, integer=True)
                os.remove(path)
                res[str(k)] = scan(msk, zooms)
                break
            except Exception as e:   # noqa: BLE001  (signed URL expiry, network): reopen and retry
                print(f"{k}: attempt {attempt + 1} failed: {e!r}", flush=True)
                zf = None
                time.sleep(20)
        else:
            res[str(k)] = {"error": "failed 3 times"}
        r = res[str(k)]
        print(f"lits_{k:03d}: {r.get('tumours')} tumours ({r.get('small')} < 10 mm), {r.get('slices')} slices, "
              f"{time.time() - t0:.0f} s", flush=True)
        json.dump(res, open(args.out + ".tmp", "w"), indent=1)
        os.replace(args.out + ".tmp", args.out)
    ok = [v for v in res.values() if "tumours" in v]
    print(f"done: {len(ok)} patients, {sum(v['tumours'] for v in ok)} tumours, "
          f"{sum(v['small'] for v in ok)} under 10 mm, {sum(v['slices'] for v in ok)} liver slices")


if __name__ == "__main__":
    main()

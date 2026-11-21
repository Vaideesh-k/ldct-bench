"""Step 6: combine the detection results of the existing 10 LiTS test patients
(detect_<input>.json) and the 10 extended patients (detect_ext_<input>.json) into one 20-patient
file per input, for scripts/final_stats.py --detect-results <out>.

    python scripts/merge_detect.py --results dgx_results --out dgx_results/combined20
"""
import argparse
import json
import os

import numpy as np

INPUTS = ["full", "quarter", "ct_mamba", "ctformer", "corediff", "ct_mamba_control",
          "ct_mamba_refiner", "ct_mamba_refiner25d"]


def summarise(pp, small_mm=10.0):
    les = [l for p in pp.values() for l in p["per_lesion"]]
    small = [l for l in les if l["diameter_mm"] < small_mm]
    return {"patients": len(pp), "lesions": len(les), "detected": sum(l["detected"] for l in les),
            "detection_rate": sum(l["detected"] for l in les) / len(les),
            "small_lesions_under_10mm": len(small), "small_detected": sum(l["detected"] for l in small),
            "small_detection_rate": sum(l["detected"] for l in small) / len(small) if small else None,
            "false_positives_per_patient": float(np.mean([p["false_positives"] for p in pp.values()])),
            "mean_per_lesion_dice": float(np.mean([l["dice"] for l in les])),
            "mean_per_patient_dice": float(np.mean([p["dice"] for p in pp.values()]))}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="dgx_results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    for name in INPUTS:
        a = json.load(open(os.path.join(args.results, f"detect_{name}.json")))["per_patient"]
        b = json.load(open(os.path.join(args.results, f"detect_ext_{name}.json")))["per_patient"]
        assert not set(a) & set(b), f"{name}: patients in both sets"
        pp = {**a, **b}
        s = summarise(pp)
        # sanity: the 10-patient summary must be reproduced by the same code
        old = json.load(open(os.path.join(args.results, f"detect_{name}.json")))["summary"]
        assert abs(summarise(a)["mean_per_lesion_dice"] - old["mean_per_lesion_dice"]) < 1e-9, name
        json.dump({"summary": s, "per_patient": pp, "sources": [f"detect_{name}.json", f"detect_ext_{name}.json"]},
                  open(os.path.join(args.out, f"detect_{name}.json"), "w"), indent=1)
        print(f"{name:20s} {s['detected']}/{s['lesions']} small {s['small_detected']}/{s['small_lesions_under_10mm']} "
              f"FP/pt {s['false_positives_per_patient']:.2f} Dice {s['mean_per_lesion_dice']:.4f}")


if __name__ == "__main__":
    main()

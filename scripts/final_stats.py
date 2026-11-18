"""Step 5/7: final tables, bootstrap CIs, paired tests and the success-criteria verdict.

    python scripts/final_stats.py --results dgx_results --tag existing10 --out dgx_results/final_existing10

Inputs (all written by the normal pipeline):
  <results>/<input>_scores_mayo_L506.csv      per-slice PSNR/SSIM/RMSE (quarter_dose, models)
  <results>/detect_<input>.json               lesion-wise detection (full, quarter, models)
Inputs compared: normal dose (full), quarter dose, CT-Mamba, CTformer, CoreDiff, control, refiner.
Bootstrap: 10,000 resamples, seed 0, percentile 95% CI; slices are resampled for L506, tumours
for per-tumour Dice, patients for false positives per patient. Paired = the same slices /
tumours / patients for both inputs. Wilcoxon signed-rank, two-sided.
Criteria: docs/success_criteria.md (fixed before any result).
Writes <out>.json, <out>_image.csv, <out>_detection.csv, <out>_paired.csv, <out>.md.
"""
import argparse
import csv
import json
import os

import numpy as np
from scipy.stats import wilcoxon

NAMES = {"full": "normal dose", "quarter": "quarter dose", "quarter_dose": "quarter dose",
         "ct_mamba": "CT-Mamba", "ctformer": "CTformer", "corediff": "CoreDiff",
         "ct_mamba_control": "control", "ct_mamba_refiner": "refiner"}
BASELINES = ["ct_mamba", "ctformer", "corediff"]
B, SEED = 10_000, 0


def boot_mean(x, rng):
    x = np.asarray(x, float)
    idx = rng.integers(0, len(x), (B, len(x)))
    m = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def paired(a, b, rng):
    d = np.asarray(a, float) - np.asarray(b, float)
    mean, lo, hi = boot_mean(d, rng)
    p = float(wilcoxon(a, b).pvalue) if np.any(d != 0) else 1.0
    return {"mean_difference": mean, "ci_low": lo, "ci_high": hi, "wilcoxon_p": p,
            "significant": bool(p < 0.05 and (lo > 0 or hi < 0))}


def load_scores(results, name):
    f = os.path.join(results, f"{name}_scores_mayo_L506.csv")
    if not os.path.exists(f):
        return None
    rows = list(csv.DictReader(open(f)))
    rows.sort(key=lambda r: r["name"])
    return {k: np.array([float(r[k]) for r in rows]) for k in ("psnr", "ssim", "rmse")} | {"names": [r["name"] for r in rows]}


def load_detect(results, name):
    f = os.path.join(results, f"detect_{name}.json")
    if not os.path.exists(f):
        return None
    d = json.load(open(f))
    pp = d["per_patient"]
    les = [(p, i, l) for p in sorted(pp) for i, l in enumerate(pp[p]["per_lesion"])]
    return {"summary": d["summary"], "key": [(p, i, l["voxels"]) for p, i, l in les],
            "dice": np.array([l["dice"] for _, _, l in les]),
            "detected": np.array([l["detected"] for _, _, l in les]),
            "diam": np.array([l["diameter_mm"] for _, _, l in les]),
            "fp": np.array([pp[p]["false_positives"] for p in sorted(pp)], float)}


def fmt_small(v):
    return "-" if v["per_tumour_dice_small"] is None else f"{v['per_tumour_dice_small']:.3f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="dgx_results")
    ap.add_argument("--detect-results", help="folder with detect_*.json (default: --results)")
    ap.add_argument("--tag", default="existing10")
    ap.add_argument("--out", required=True)
    ap.add_argument("--small-mm", type=float, default=10.0)
    ap.add_argument("--psnr-target", type=float, default=29.42)
    args = ap.parse_args()
    dres = args.detect_results or args.results
    img_inputs = ["quarter_dose"] + BASELINES + ["ct_mamba_control", "ct_mamba_refiner"]
    det_inputs = ["full", "quarter"] + BASELINES + ["ct_mamba_control", "ct_mamba_refiner"]
    S = {n: load_scores(args.results, n) for n in img_inputs}
    D = {n: load_detect(dres, n) for n in det_inputs}
    S = {k: v for k, v in S.items() if v is not None}
    D = {k: v for k, v in D.items() if v is not None}
    for n in S:
        assert S[n]["names"] == S["ct_mamba"]["names"], f"{n}: different L506 slices"
    for n in D:
        assert D[n]["key"] == D["full"]["key"], f"{n}: different tumours"
    rng = np.random.default_rng(SEED)
    out = {"tag": args.tag, "bootstrap": {"resamples": B, "seed": SEED}, "image": {}, "detection": {}, "paired": {}}

    for n, s in S.items():
        out["image"][n] = {k: dict(zip(("mean", "ci_low", "ci_high"), boot_mean(s[k], rng))) for k in ("psnr", "ssim", "rmse")}
        out["image"][n]["slices"] = len(s["psnr"])
    small = next(iter(D.values()))["diam"] < args.small_mm if D else None
    for n, d in D.items():
        sm = d["summary"]
        out["detection"][n] = {
            "tumours": int(len(d["dice"])), "detected": int(d["detected"].sum()),
            "small": int(small.sum()), "small_detected": int(d["detected"][small].sum()),
            "fp_per_patient": dict(zip(("mean", "ci_low", "ci_high"), boot_mean(d["fp"], rng))),
            "per_tumour_dice": dict(zip(("mean", "ci_low", "ci_high"), boot_mean(d["dice"], rng))),
            "per_tumour_dice_small": float(d["dice"][small].mean()) if small.any() else None,
            "per_patient_dice": sm.get("mean_per_patient_dice"), "patients": sm["patients"]}

    ref = "ct_mamba_refiner"
    others = [n for n in ("ct_mamba_control",) + tuple(BASELINES) + ("quarter", "full") if n in D]
    if ref in D:
        for n in others:
            out["paired"][f"dice_{ref}_vs_{n}"] = paired(D[ref]["dice"], D[n]["dice"], rng)
            out["paired"][f"fp_{ref}_vs_{n}"] = paired(D[ref]["fp"], D[n]["fp"], rng)
    if ref in S:
        for n in [k for k in S if k != ref]:
            for k in ("psnr", "ssim", "rmse"):
                out["paired"][f"{k}_{ref}_vs_{n}"] = paired(S[ref][k], S[n][k], rng)
    if "ct_mamba_control" in D:   # does the extra training alone help?
        out["paired"]["dice_ct_mamba_control_vs_ct_mamba"] = paired(D["ct_mamba_control"]["dice"], D["ct_mamba"]["dice"], rng)

    # ---- success criteria (docs/success_criteria.md) ----
    crit = {}
    if ref in S:
        v = out["image"][ref]["psnr"]["mean"]
        crit["a"] = {"rule": f"L506 PSNR >= {args.psnr_target}", "value": v, "pass": v >= args.psnr_target}
    if ref in D:
        best = max(BASELINES, key=lambda n: out["detection"][n]["per_tumour_dice"]["mean"])
        r, b = out["detection"][ref]["per_tumour_dice"]["mean"], out["detection"][best]["per_tumour_dice"]["mean"]
        crit["b"] = {"rule": f"per-tumour Dice >= best baseline ({NAMES[best]} {b:.4f})", "value": r, "pass": r >= b}
        if "ct_mamba_control" in D:
            c = out["detection"]["ct_mamba_control"]["per_tumour_dice"]["mean"]
            crit["c"] = {"rule": f"per-tumour Dice > control ({c:.4f})", "value": r, "pass": r > c}
            pc, pb = out["paired"][f"dice_{ref}_vs_ct_mamba_control"], out["paired"][f"dice_{ref}_vs_{best}"]
            crit["d"] = {"rule": "refiner vs control AND vs best baseline: Wilcoxon p<0.05 and CI excludes 0, in the refiner's favour",
                         "vs_control": pc, "vs_best_baseline": pb,
                         "pass": pc["significant"] and pc["mean_difference"] > 0 and pb["significant"] and pb["mean_difference"] > 0}
        f_r, f_m = out["detection"][ref]["fp_per_patient"]["mean"], out["detection"]["ct_mamba"]["fp_per_patient"]["mean"]
        crit["e"] = {"rule": f"FP/patient <= CT-Mamba ({f_m:.2f})", "value": f_r, "pass": f_r <= f_m}
    out["criteria"] = crit
    out["verdict"] = ("WIN" if crit and len(crit) == 5 and all(c["pass"] for c in crit.values())
                      else "NOT A WIN" if len(crit) == 5 else "INCOMPLETE")

    json.dump(out, open(args.out + ".json", "w"), indent=1)
    with open(args.out + "_image.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["input", "slices", "psnr", "psnr_ci_low", "psnr_ci_high", "ssim", "ssim_ci_low", "ssim_ci_high", "rmse", "rmse_ci_low", "rmse_ci_high"])
        for n, v in out["image"].items():
            w.writerow([NAMES[n], v["slices"]] + [round(v[k][q], 4) for k in ("psnr", "ssim", "rmse") for q in ("mean", "ci_low", "ci_high")])
    with open(args.out + "_detection.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["input", "tumours", "detected", "small", "small_detected", "fp_per_patient", "fp_ci_low", "fp_ci_high", "per_tumour_dice", "dice_ci_low", "dice_ci_high", "dice_small", "per_patient_dice"])
        for n, v in out["detection"].items():
            w.writerow([NAMES[n], v["tumours"], v["detected"], v["small"], v["small_detected"]] +
                       [round(v["fp_per_patient"][q], 3) for q in ("mean", "ci_low", "ci_high")] +
                       [round(v["per_tumour_dice"][q], 4) for q in ("mean", "ci_low", "ci_high")] +
                       [None if v["per_tumour_dice_small"] is None else round(v["per_tumour_dice_small"], 4), round(v["per_patient_dice"], 4)])
    with open(args.out + "_paired.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["comparison", "mean_difference", "ci_low", "ci_high", "wilcoxon_p", "significant"])
        for k, v in out["paired"].items():
            w.writerow([k, round(v["mean_difference"], 5), round(v["ci_low"], 5), round(v["ci_high"], 5), round(v["wilcoxon_p"], 4), v["significant"]])

    L = [f"# Results: {args.tag}", "", "## Mayo L506 (window -160..240 HU), mean [95% bootstrap CI]", "",
         "| input | PSNR dB | SSIM | RMSE HU |", "|---|---|---|---|"]
    for n, v in out["image"].items():
        L.append(f"| {NAMES[n]} | " + " | ".join(f"{v[k]['mean']:.{p}f} [{v[k]['ci_low']:.{p}f}, {v[k]['ci_high']:.{p}f}]"
                                                 for k, p in (("psnr", 2), ("ssim", 4), ("rmse", 2))) + " |")
    L += ["", f"## LiTS detection ({next(iter(out['detection'].values()))['patients'] if out['detection'] else '?'} patients), same nnU-Net", "",
          "| input | detected | < 10 mm detected | FP / patient | per-tumour Dice | Dice < 10 mm |", "|---|---|---|---|---|---|"]
    for n, v in out["detection"].items():
        L.append(f"| {NAMES[n]} | {v['detected']}/{v['tumours']} | {v['small_detected']}/{v['small']} | "
                 f"{v['fp_per_patient']['mean']:.2f} [{v['fp_per_patient']['ci_low']:.2f}, {v['fp_per_patient']['ci_high']:.2f}] | "
                 f"{v['per_tumour_dice']['mean']:.3f} [{v['per_tumour_dice']['ci_low']:.3f}, {v['per_tumour_dice']['ci_high']:.3f}] | "
                 f"{fmt_small(v)} |")
    if out["paired"]:
        L += ["", "## Paired comparisons (difference = first - second)", "", "| comparison | mean diff [95% CI] | Wilcoxon p | significant |", "|---|---|---|---|"]
        for k, v in out["paired"].items():
            L.append(f"| {k} | {v['mean_difference']:+.4f} [{v['ci_low']:+.4f}, {v['ci_high']:+.4f}] | {v['wilcoxon_p']:.4f} | {'yes' if v['significant'] else 'no'} |")
    L += ["", "## Success criteria (docs/success_criteria.md)", ""]
    for k in "abcde":
        if k in crit:
            c = crit[k]
            val = f" (refiner {c['value']:.4f})" if "value" in c else ""
            L.append(f"- ({k}) {c['rule']}{val}: **{'PASS' if c['pass'] else 'FAIL'}**")
        else:
            L.append(f"- ({k}) not evaluated yet")
    L += ["", f"**Verdict: {out['verdict']}**"]
    open(args.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

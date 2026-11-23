"""Verdict for the edge-preserving fine-tune (docs/success_criteria_edge.md, fixed before training).

    python scripts/edge_stats.py --results dgx_results --out dgx_results/final_fresh_edge

Inputs (fetched from the DGX into --results):
  <input>_scores_mayo_L506.csv, <input>_scores_lits_fresh.csv   per-slice PSNR/SSIM/RMSE
  detect_fresh_<input>.json                                    lesion-wise detection on lits_fresh
Same bootstrap / Wilcoxon helpers as final_stats.py. Writes <out>.json and <out>.md.
"""
import argparse
import csv
import json
import os

import numpy as np

from final_stats import BASELINES, NAMES, boot_mean, load_detect, paired

NAMES = dict(NAMES, ct_mamba_edge="edge model")
CAND, CONTROL = "ct_mamba_edge", "ct_mamba_control"
INPUTS = ["full", "quarter"] + BASELINES + [CONTROL, "ct_mamba_refiner", "ct_mamba_refiner25d", CAND]


def scores(results, name, setname):
    f = os.path.join(results, f"{name}_scores_{setname}.csv")
    if not os.path.exists(f):
        return None
    rows = sorted(csv.DictReader(open(f)), key=lambda r: r["name"])
    return np.array([float(r["psnr"]) for r in rows]), [r["name"] for r in rows]


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", default="dgx_results")
    ap.add_argument("--out", required=True)
    ap.add_argument("--small-mm", type=float, default=10.0)
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    out = {"image": {}, "detection": {}, "paired": {}, "criteria": {}}
    psnr = {}
    for n in ["quarter_dose"] + INPUTS[2:]:
        for s in ("mayo_L506", "lits_fresh"):
            v = scores(args.results, n, s)
            if v:
                psnr[(n, s)] = v
                m, lo, hi = boot_mean(v[0], rng)
                out["image"].setdefault(n, {})[s] = {"slices": len(v[0]), "psnr": m, "ci_low": lo, "ci_high": hi}
    D = {n: load_detect(args.results, f"fresh_{n}") for n in INPUTS}
    D = {n: d for n, d in D.items() if d is not None}
    for n, d in D.items():
        m, lo, hi = boot_mean(d["dice"], rng)
        small = d["diam"] < args.small_mm
        fm, flo, fhi = boot_mean(d["fp"], rng)
        out["detection"][n] = {"tumours": len(d["dice"]), "detected": int(d["detected"].sum()),
                               "small": int(small.sum()), "small_detected": int(d["detected"][small].sum()),
                               "dice": m, "dice_ci": [lo, hi], "dice_small": float(d["dice"][small].mean()) if small.any() else None,
                               "fp_per_patient": fm, "fp_ci": [flo, fhi], "per_patient_dice": d["summary"]["mean_per_patient_dice"]}
    ok = CAND in D and (CAND, "lits_fresh") in psnr and (CAND, "mayo_L506") in psnr
    if ok:
        for o in [CONTROL] + BASELINES + ["ct_mamba_refiner", "ct_mamba_refiner25d", "full"]:
            if o in D:
                assert D[o]["key"] == D[CAND]["key"], f"tumour lists differ: {o}"
                out["paired"][o] = paired(D[CAND]["dice"], D[o]["dice"], rng)
        mayo = out["image"][CAND]["mayo_L506"]["psnr"]
        crit = out["criteria"]
        crit["a"] = {"rule": "L506 PSNR >= 29.42 dB (CT-Mamba)", "value": mayo, "pass": mayo >= 29.42}
        bb = max(BASELINES, key=lambda n: out["image"][n]["lits_fresh"]["psnr"])
        fr, bv = out["image"][CAND]["lits_fresh"]["psnr"], out["image"][bb]["lits_fresh"]["psnr"]
        crit["b"] = {"rule": f"lits_fresh PSNR > best baseline ({NAMES[bb]} {bv:.2f})", "value": fr, "pass": fr > bv}
        bd = max(BASELINES, key=lambda n: out["detection"][n]["dice"])
        pc = out["paired"][bd]
        crit["c"] = {"rule": f"Dice > best baseline ({NAMES[bd]} {out['detection'][bd]['dice']:.4f}), significant",
                     "value": out["detection"][CAND]["dice"], "paired": pc, "pass": pc["mean_difference"] > 0 and pc["significant"]}
        pk = out["paired"][CONTROL]
        crit["d"] = {"rule": f"Dice > control ({out['detection'][CONTROL]['dice']:.4f}), significant",
                     "value": out["detection"][CAND]["dice"], "paired": pk, "pass": pk["mean_difference"] > 0 and pk["significant"]}
        out["verdict"] = "WIN" if all(c["pass"] for c in crit.values()) else "NOT A WIN"
    json.dump(out, open(args.out + ".json", "w"), indent=1)

    L = ["# Edge model on the fresh test set (criteria: docs/success_criteria_edge.md)", "",
         "| input | L506 PSNR | fresh PSNR | detected | < 10 mm | FP/pt | per-tumour Dice [95% CI] | Dice < 10 mm |",
         "|---|---|---|---|---|---|---|---|"]
    for n in INPUTS:
        img = out["image"].get("quarter_dose" if n == "quarter" else n, {})
        d = out["detection"].get(n)
        f = lambda s: f"{img[s]['psnr']:.2f}" if s in img else "-"
        det = (f"{d['detected']}/{d['tumours']} | {d['small_detected']}/{d['small']} | {d['fp_per_patient']:.2f} | "
               f"{d['dice']:.3f} [{d['dice_ci'][0]:.3f}, {d['dice_ci'][1]:.3f}] | "
               f"{'-' if d['dice_small'] is None else format(d['dice_small'], '.3f')}") if d else "- | - | - | - | -"
        L.append(f"| {NAMES[n]} | {f('mayo_L506')} | {f('lits_fresh')} | {det} |")
    if out["criteria"]:
        L += ["", "## Criteria", ""]
        for k, c in out["criteria"].items():
            L.append(f"- ({k}) {c['rule']}: {c['value']:.4f} -> {'PASS' if c['pass'] else 'FAIL'}")
        L += ["", f"**Verdict: {out['verdict']}**", "", "## Paired per-tumour Dice, edge minus other", ""]
        for o, p in out["paired"].items():
            L.append(f"- vs {NAMES[o]}: {p['mean_difference']:+.4f} [{p['ci_low']:+.4f}, {p['ci_high']:+.4f}], "
                     f"Wilcoxon p {p['wilcoxon_p']:.2g}{' (significant)' if p['significant'] else ''}")
    open(args.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()

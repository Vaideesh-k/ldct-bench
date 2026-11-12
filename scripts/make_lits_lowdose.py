"""Convert one LiTS patient (NIfTI scan + mask) into our .npy layout and make a
simulated low-dose copy of every kept slice.

    python scripts/make_lits_lowdose.py --volume volume-0.nii --mask segmentation-0.nii \
        --out data/lits --id lits_000 [--I0 2e5 --dose 0.25 --margin 5 --workers 16]

Output (HU, int16, standard radiological orientation, like data/mayo):
    <out>/<id>/full/<id>_0000.npy      the original normal-dose slice
    <out>/<id>/quarter/<id>_0000.npy   the simulated low-dose slice (dose fraction --dose)
    <out>/<id>/mask/<id>_0000.npy      uint8: 0 background, 1 liver, 2 tumour
    <out>/<id>/meta.json               spacing, kept slice range, I0, noise statistics
Only slices from `margin` below the liver to `margin` above it are kept.

LOW-DOSE SIMULATION (noise insertion in the projection domain, after Yu et al.,
"Development and validation of a practical lower-dose-simulation tool for optimizing
computed tomography scan protocols", JCAT 2012):
  1. HU -> linear attenuation per pixel, mu = mu_water * spacing * (1 + HU/1000).
  2. Parallel-beam projections p = radon(mu) over 180 degrees (n_angles views).
  3. Photons at the detector for the normal-dose scan: I0 * exp(-p). A scan at dose
     fraction d has I0*d photons, so its quantum noise variance in p is larger by
     (1/d - 1) * exp(p) / I0. We ADD exactly that extra noise, so the original image
     noise is kept and the result has quarter-dose noise, by drawing Poisson counts at
     I_eff = I0 / (1/d - 1) photons plus Gaussian electronic noise (sigma_e counts,
     scaled so that its extra variance also matches), and taking
     dp = -ln(N_noisy / N_eff)  (noise only, zero mean).
  4. The noise image is FBP(dp) with a ramp filter (skimage iradon) and is added to
     the original HU image.
  I0 is calibrated once (scripts/calibrate_lits_noise.py) so that the added noise in
  uniform liver tissue matches the Mayo quarter-dose minus full-dose noise.
  LIMITATION (documented): parallel-beam geometry, no bow-tie filter, no beam
  hardening, no detector cross-talk; the noise level is matched to Mayo, the fine
  noise texture is only an approximation of a fan-beam scanner's.
"""
import argparse
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

MU_WATER = 0.0192      # linear attenuation of water, per mm (~70 keV effective energy)
HU_MIN, HU_MAX = -1024, 3072
LABEL_LIVER, LABEL_TUMOUR = 1, 2


# ----------------------------------------------------------------------------- simulation
def simulate_low_dose(hu, spacing_mm, I0=2e5, dose=0.25, sigma_e=10.0, n_angles=720, seed=0):
    """Low-dose version (float32 HU) of one HU slice, see the module docstring."""
    from skimage.transform import iradon, radon
    rng = np.random.default_rng(seed)
    hu = np.asarray(hu, dtype=np.float64)
    mu = np.clip(MU_WATER * spacing_mm * (1.0 + hu / 1000.0), 0.0, None)   # per pixel
    theta = np.linspace(0.0, 180.0, n_angles, endpoint=False)
    p = radon(mu, theta=theta, circle=False)                               # line integrals

    extra = 1.0 / dose - 1.0                       # 3 for quarter dose
    n_eff = (I0 / extra) * np.exp(-p)              # photons whose Poisson noise = the extra quantum noise
    sigma_g = sigma_e * np.sqrt((1.0 / dose ** 2 - 1.0)) / extra   # electronic noise, extra part only
    noisy = rng.poisson(n_eff).astype(np.float64) + rng.normal(0.0, sigma_g, size=p.shape)
    noisy = np.maximum(noisy, 0.5)                 # photon starvation guard (keeps the log finite)
    dp = -np.log(noisy / n_eff)                    # projection-domain noise, zero mean

    noise_mu = iradon(dp, theta=theta, circle=False, output_size=hu.shape[0], filter_name="ramp")
    noise_hu = noise_mu / (MU_WATER * spacing_mm) * 1000.0
    return (hu + noise_hu).astype(np.float32)


def added_noise_std(low, full, roi):
    """Std of (low - full) inside a boolean ROI, or nan if the ROI is too small."""
    if roi.sum() < 200:
        return float("nan")
    d = (np.asarray(low, np.float64) - np.asarray(full, np.float64))[roi]
    return float(d.std())


def liver_roi(mask, erode=3):
    """Liver pixels (liver or tumour label) away from the liver edge."""
    from scipy.ndimage import binary_erosion
    liver = mask >= LABEL_LIVER
    return binary_erosion(liver, iterations=erode) if liver.any() else liver


# ----------------------------------------------------------------------------- NIfTI handling
def load_canonical(path, integer=False):
    """Volume as (x, y, z) in RAS orientation (so slices can be shown the standard way)
    plus voxel spacing (dx, dy, dz) in mm."""
    import nibabel as nib
    img = nib.as_closest_canonical(nib.load(path))
    if integer:
        data = np.rint(np.asanyarray(img.dataobj)).astype(np.int16)
    else:
        data = np.asanyarray(img.dataobj).astype(np.float32)
    zooms = tuple(float(z) for z in img.header.get_zooms()[:3])
    return data, zooms


def axial(data, z):
    """One axial slice in radiological display orientation: anterior at the top,
    the patient's right on the left of the image (rows = -y, columns = -x of RAS)."""
    return np.ascontiguousarray(data[:, :, z].T[::-1, ::-1])


def liver_slice_range(mask, margin):
    """First and last z index to keep: the liver's extent plus `margin` slices each way."""
    zs = np.where((mask >= LABEL_LIVER).any(axis=(0, 1)))[0]
    if len(zs) == 0:
        raise ValueError("the mask contains no liver")
    return max(int(zs[0]) - margin, 0), min(int(zs[-1]) + margin, mask.shape[2] - 1)


# ----------------------------------------------------------------------------- per patient
def _one_slice(job):
    full, spacing, I0, dose, seed = job
    return simulate_low_dose(full, spacing, I0=I0, dose=dose, seed=seed)


def convert_patient(volume_path, mask_path, out_root, pid, I0, dose=0.25, margin=5, workers=8, seed=0):
    vol, zooms = load_canonical(volume_path)
    msk, mz = load_canonical(mask_path, integer=True)
    if vol.shape != msk.shape:
        raise ValueError(f"{pid}: scan {vol.shape} and mask {msk.shape} differ in shape")
    if not np.allclose(zooms, mz, atol=1e-3):
        print(f"WARNING {pid}: scan spacing {zooms} != mask spacing {mz}")
    if abs(zooms[0] - zooms[1]) > 1e-3:
        print(f"WARNING {pid}: non-square pixels {zooms[:2]} mm; using their mean for the simulation")
    spacing = float(np.mean(zooms[:2]))
    z0, z1 = liver_slice_range(msk, margin)

    fulls = [np.clip(axial(vol, z), HU_MIN, HU_MAX) for z in range(z0, z1 + 1)]
    masks = [axial(msk, z).astype(np.uint8) for z in range(z0, z1 + 1)]
    jobs = [(f, spacing, I0, dose, seed * 100_000 + i) for i, f in enumerate(fulls)]
    if workers > 1:
        with Pool(workers) as pool:
            lows = pool.map(_one_slice, jobs, chunksize=1)
    else:
        lows = [_one_slice(j) for j in jobs]

    for kind in ("full", "quarter", "mask"):
        os.makedirs(os.path.join(out_root, pid, kind), exist_ok=True)
    noise = []
    for i, (f, l, m) in enumerate(zip(fulls, lows, masks)):
        name = f"{pid}_{i:04d}.npy"
        np.save(os.path.join(out_root, pid, "full", name), np.rint(f).astype(np.int16))
        np.save(os.path.join(out_root, pid, "quarter", name), np.rint(np.clip(l, HU_MIN, HU_MAX)).astype(np.int16))
        np.save(os.path.join(out_root, pid, "mask", name), m)
        noise.append(added_noise_std(l, f, liver_roi(m)))

    noise = np.array(noise)
    meta = {
        "id": pid, "source_volume": os.path.basename(volume_path), "source_mask": os.path.basename(mask_path),
        "spacing_mm": {"x": zooms[0], "y": zooms[1], "z": zooms[2]}, "shape_xyz": list(vol.shape),
        "kept_slices": {"first": z0, "last": z1, "count": len(fulls)}, "margin": margin,
        "simulation": {"I0": I0, "dose": dose, "sigma_e": 10.0, "n_angles": 720, "seed": seed,
                       "geometry": "parallel-beam, skimage radon/iradon, noise insertion"},
        "liver_added_noise_std_hu": {"median": float(np.nanmedian(noise)) if np.isfinite(noise).any() else None,
                                     "slices_with_liver_roi": int(np.isfinite(noise).sum())},
        "tumour_voxels": int(sum(int((m == LABEL_TUMOUR).sum()) for m in masks)),
        "liver_voxels": int(sum(int((m >= LABEL_LIVER).sum()) for m in masks)),
    }
    with open(os.path.join(out_root, pid, "meta.json"), "w") as f:
        json.dump(meta, f, indent=1)
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--volume", required=True)
    ap.add_argument("--mask", required=True)
    ap.add_argument("--out", default="data/lits")
    ap.add_argument("--id", required=True, help="patient folder name, e.g. lits_000")
    ap.add_argument("--I0", type=float, required=True, help="incident photons per detector element (calibrated)")
    ap.add_argument("--dose", type=float, default=0.25)
    ap.add_argument("--margin", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    meta = convert_patient(args.volume, args.mask, args.out, args.id, args.I0, args.dose, args.margin, args.workers, args.seed)
    k, n = meta["kept_slices"], meta["liver_added_noise_std_hu"]
    print(f"{args.id}: slices {k['first']}-{k['last']} ({k['count']} kept), spacing {meta['spacing_mm']}, "
          f"liver added-noise std {n['median']} HU over {n['slices_with_liver_roi']} slices, "
          f"tumour voxels {meta['tumour_voxels']}")


if __name__ == "__main__":
    main()

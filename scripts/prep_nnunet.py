"""Write LiTS TRAIN patients from data/lits (normal-dose slices + masks) as an nnU-Net v2
raw dataset, and (later) any denoised/low-dose LiTS patients as inference inputs.

    python scripts/prep_nnunet.py --lits data/lits --split configs/lits_split.json \
        --out $nnUNet_raw --dataset-id 501 [--name LiTSLiver]
    python scripts/prep_nnunet.py --lits data/lits --split configs/lits_split.json --part test \
        --images-only --out <folder>  [--from-predictions runs/<model>/predictions/lits]

Output (training): <out>/Dataset501_LiTSLiver/{imagesTr/<id>_0000.nii.gz, labelsTr/<id>.nii.gz,
dataset.json}. Volumes are rebuilt from the per-slice .npy files (z, y, x) with the voxel
spacing from meta.json, so the 3D trainer sees the real geometry. Labels: 0 background,
1 liver, 2 tumour (as LiTS). Only the slices around the liver exist in data/lits, which is
exactly the region that matters here.
RULE: training data = the 'train' part of the split only (the script refuses 'test' for labels).
"""
import argparse
import glob
import json
import os

import numpy as np


def load_patient(folder, kind):
    files = sorted(glob.glob(os.path.join(folder, kind, "*.npy")))
    if not files:
        raise FileNotFoundError(f"no {kind} slices in {folder}")
    return np.stack([np.load(f) for f in files])          # (z, y, x)


def save_nifti(vol_zyx, spacing_zyx, path):
    """nnU-Net reads NIfTI with nibabel/SimpleITK; keep the array as (x, y, z) with a
    diagonal affine from the spacing (orientation is irrelevant for training as long as
    image and label agree, and we write both the same way)."""
    import nibabel as nib
    arr = np.ascontiguousarray(np.transpose(vol_zyx, (2, 1, 0)))
    affine = np.diag([spacing_zyx[2], spacing_zyx[1], spacing_zyx[0], 1.0])
    nib.save(nib.Nifti1Image(arr, affine), path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lits", default="data/lits")
    ap.add_argument("--split", default="configs/lits_split.json")
    ap.add_argument("--part", default="train", choices=["train", "test"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--dataset-id", type=int, default=501)
    ap.add_argument("--name", default="LiTSLiver")
    ap.add_argument("--images-only", action="store_true", help="write only images (inference inputs), flat in --out")
    ap.add_argument("--from-predictions", help="take the images from a predictions folder (<pid>/<name>.npy) instead of data/lits/<pid>/full")
    args = ap.parse_args()

    with open(args.split) as f:
        split = json.load(f)
    ids = [f"lits_{k:03d}" for k in split[args.part]]
    if args.part == "test" and not args.images_only:
        raise SystemExit("the locked test patients are never written with labels for training")

    if args.images_only:
        img_dir, lab_dir = args.out, None
    else:
        ds_dir = os.path.join(args.out, f"Dataset{args.dataset_id:03d}_{args.name}")
        img_dir, lab_dir = os.path.join(ds_dir, "imagesTr"), os.path.join(ds_dir, "labelsTr")
        os.makedirs(lab_dir, exist_ok=True)
    os.makedirs(img_dir, exist_ok=True)

    written = []
    for pid in ids:
        folder = os.path.join(args.lits, pid)
        if not os.path.exists(os.path.join(folder, "meta.json")):
            print(f"{pid}: not converted yet, skipped")
            continue
        with open(os.path.join(folder, "meta.json")) as f:
            sp = json.load(f)["spacing_mm"]
        spacing = (sp["z"], sp["y"], sp["x"])
        if args.from_predictions:
            files = sorted(glob.glob(os.path.join(args.from_predictions, pid, "*.npy")))
            if not files:
                print(f"{pid}: no predictions, skipped")
                continue
            img = np.stack([np.load(f) for f in files]).astype(np.float32)
        else:
            img = load_patient(folder, "full").astype(np.float32)
        save_nifti(img, spacing, os.path.join(img_dir, f"{pid}_0000.nii.gz"))
        if lab_dir:
            lab = load_patient(folder, "mask").astype(np.uint8)
            if lab.shape != img.shape:
                raise SystemExit(f"{pid}: mask {lab.shape} vs image {img.shape}")
            save_nifti(lab, spacing, os.path.join(lab_dir, f"{pid}.nii.gz"))
        written.append(pid)
        print(f"{pid}: {img.shape} spacing {spacing}")

    if lab_dir:
        dataset = {
            "channel_names": {"0": "CT"},
            "labels": {"background": 0, "liver": 1, "tumour": 2},
            "numTraining": len(written),
            "file_ending": ".nii.gz",
            "name": args.name,
            "description": "LiTS part 1 train patients (locked split configs/lits_split.json), liver region, normal dose",
        }
        with open(os.path.join(ds_dir, "dataset.json"), "w") as f:
            json.dump(dataset, f, indent=1)
        print(f"wrote {ds_dir} with {len(written)} training cases")
    else:
        print(f"wrote {len(written)} images to {img_dir}")


if __name__ == "__main__":
    main()


"""Convert Mayo 2016 DICOM files into the pipeline's .npy layout (HU, int16).

    python scripts/prep_mayo.py --src /path/to/Mayo2016 --dst data/mayo
"""
import argparse
import os
import re
from glob import glob

import numpy as np
import pydicom

PATIENT = re.compile(r"^L\d{3}$")   # patient folders look like L067, L506


def find_series(patient_dir, pattern):
    """Find the one folder matching pattern (e.g. '*quarter*1mm*') and list its DICOM files."""
    dirs = [d for d in glob(os.path.join(patient_dir, "**", pattern), recursive=True)
            if os.path.isdir(d)]
    if len(dirs) != 1:
        raise RuntimeError(f"Expected one folder matching '{pattern}' in {patient_dir}, found {dirs}")
    files = [f for f in glob(os.path.join(dirs[0], "**", "*"), recursive=True)
             if os.path.isfile(f) and f.lower().endswith((".ima", ".dcm"))]
    if not files:
        raise RuntimeError(f"No .IMA or .dcm files in {dirs[0]}")
    return files
def load_hu(files):
    """Read DICOM slices, put them in body order, and convert them to HU."""
    # 1. open every file (picture + note)
    slices = [pydicom.dcmread(f) for f in files]

    # 2. sort by position in the body (the z coordinate), so noisy and clean line up
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    # 3. convert raw pixel numbers to HU using the numbers stored in the note
    images = []
    for s in slices:
        raw = s.pixel_array.astype(np.float32)
        slope = float(getattr(s, "RescaleSlope", 1))
        intercept = float(getattr(s, "RescaleIntercept", 0))
        images.append(raw * slope + intercept)

    # 4. read the slice thickness, so we can check it really is 1 mm
    thickness = float(getattr(slices[0], "SliceThickness", 0))

    # 5. keep each slice's body position, so noisy and clean can be checked against each other
    z = [float(s.ImagePositionPatient[2]) for s in slices]
    return images, thickness, z
def main():
    # read the options typed on the command line
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="folder that contains L067, L096, ... L506")
    ap.add_argument("--dst", default="data/mayo")
    ap.add_argument("--low-pattern", default="*quarter*1mm*")
    ap.add_argument("--high-pattern", default="*full*1mm*")
    args = ap.parse_args()

    # find patient folders (L + 3 digits) and skip anything else
    patients = sorted(d for d in os.listdir(args.src) if PATIENT.match(d))
    if not patients:
        raise SystemExit(f"No patient folders like L067 found in {args.src}")

    for p in patients:
        pdir = os.path.join(args.src, p)
        low, t_low, z_low = load_hu(find_series(pdir, args.low_pattern))      # noisy
        high, t_high, z_high = load_hu(find_series(pdir, args.high_pattern))  # clean

        # every noisy slice must have a clean partner
        if len(low) != len(high):
            raise RuntimeError(f"{p}: {len(low)} quarter-dose vs {len(high)} full-dose slices")

        # ...at the same place in the body, otherwise the pairs show different anatomy
        if not np.allclose(z_low, z_high, atol=0.01):
            bad = next(i for i, (a, b) in enumerate(zip(z_low, z_high)) if abs(a - b) > 0.01)
            raise RuntimeError(f"{p}: slice {bad} is at z={z_low[bad]} (quarter) "
                               f"but z={z_high[bad]} (full); noisy and clean do not line up")

        # save as data/mayo/<patient>/quarter/<patient>_0000.npy, and the same in full/
        for kind, images in (("quarter", low), ("full", high)):
            os.makedirs(os.path.join(args.dst, p, kind), exist_ok=True)
            for i, img in enumerate(images):
                np.save(os.path.join(args.dst, p, kind, f"{p}_{i:04d}.npy"), np.round(img).astype(np.int16))

        print(f"{p}: {len(low)} slice pairs, thickness {t_low} / {t_high} mm")


if __name__ == "__main__":
    main()
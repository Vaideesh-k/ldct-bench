"""Build data/lits one patient at a time from the Kaggle LiTS part 1
(andrewmvd/liver-tumor-segmentation): download scan + mask, convert with
scripts/make_lits_lowdose.py, verify, delete the raw files, move on.

    python scripts/prep_lits.py --order 0 2 4 ... --I0 <calibrated> [--max-used-gb 38]
    python scripts/prep_lits.py --zip data/lits_raw/liver-tumor-segmentation.zip --order ... --I0 ...

Kaggle's per-file download (-f) answers 404 for this dataset's files in subfolders
(tested 12 Nov 2026 with kaggle 1.7.4.5, CLI and REST), so the working path is --zip:
download the whole part-1 archive once (5.2 GB, on the login node) and extract ONE
patient's two files at a time from it; the archive is never unpacked as a whole.
Runs inside a cpu_student job (slurm/lits_prep.sbatch). Patients already converted
(meta.json present) are skipped, so the job can be resubmitted. Stops when the home
folder exceeds --max-used-gb (quota 50 GB, keep 10 GB free). Logs disk use every
10 patients. Needs the kaggle CLI with ~/.kaggle/kaggle.json.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from make_lits_lowdose import convert_patient   # noqa: E402

DATASET = "andrewmvd/liver-tumor-segmentation"


def sh(cmd, **kw):
    print("$", " ".join(cmd), flush=True)
    return subprocess.run(cmd, check=True, text=True, capture_output=True, **kw).stdout


def kaggle_file_map(cache):
    """{basename: path-in-dataset} for every file of part 1 (cached as json)."""
    if os.path.exists(cache):
        return json.load(open(cache))
    out = sh(["kaggle", "datasets", "files", "-d", DATASET, "--page-size", "500"])
    files = {}
    for line in out.splitlines():
        parts = line.split()
        if parts and parts[0].endswith(".nii"):
            files[os.path.basename(parts[0])] = parts[0]
    json.dump(files, open(cache, "w"))
    return files


def download(path_in_dataset, dest_dir):
    """Download one file; Kaggle may deliver it zipped. Returns the .nii path."""
    base = os.path.basename(path_in_dataset)
    for attempt in range(1, 4):
        try:
            sh(["kaggle", "datasets", "download", "-d", DATASET, "-f", path_in_dataset, "-p", dest_dir, "--force"])
            break
        except subprocess.CalledProcessError as e:
            print(f"download attempt {attempt} failed: {e.stderr[-500:]}", flush=True)
            time.sleep(30 * attempt)
    else:
        raise RuntimeError(f"could not download {path_in_dataset} after 3 attempts")
    z = os.path.join(dest_dir, base + ".zip")
    if os.path.exists(z):
        with zipfile.ZipFile(z) as zf:
            zf.extractall(dest_dir)
        os.remove(z)
    nii = os.path.join(dest_dir, base)
    if not os.path.exists(nii):
        found = glob.glob(os.path.join(dest_dir, "**", base), recursive=True)
        if not found:
            raise RuntimeError(f"{base} missing after download; dir has {os.listdir(dest_dir)}")
        shutil.move(found[0], nii)
    return nii


def extract_from_zip(zip_path, member, dest_dir):
    """Extract one member of the archive to dest_dir/<basename>; returns that path."""
    out = os.path.join(dest_dir, os.path.basename(member))
    with zipfile.ZipFile(zip_path) as zf, zf.open(member) as src, open(out, "wb") as dst:
        shutil.copyfileobj(src, dst, 16 * 1024 * 1024)
    return out


def zip_file_map(zip_path):
    """{basename: member path} of every .nii in the archive."""
    with zipfile.ZipFile(zip_path) as zf:
        return {os.path.basename(n): n for n in zf.namelist() if n.endswith(".nii")}


def used_gb(path):
    out = subprocess.run(["du", "-s", "--block-size=1M", path], text=True, capture_output=True).stdout
    return int(out.split()[0]) / 1024


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--order", nargs="+", type=int, required=True, help="patient numbers in processing order")
    ap.add_argument("--I0", type=float, required=True)
    ap.add_argument("--out", default="data/lits")
    ap.add_argument("--raw", default="data/lits_raw", help="scratch folder for the downloaded files")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--margin", type=int, default=5)
    ap.add_argument("--max-used-gb", type=float, default=38.0)
    ap.add_argument("--home", default=os.path.expanduser("~"))
    ap.add_argument("--zip", help="part-1 archive: extract from it instead of downloading per file")
    args = ap.parse_args()

    os.makedirs(args.raw, exist_ok=True)
    os.makedirs(args.out, exist_ok=True)
    files = zip_file_map(args.zip) if args.zip else kaggle_file_map(os.path.join(args.out, "kaggle_part1_files.json"))
    fetch = (lambda m: extract_from_zip(args.zip, m, args.raw)) if args.zip else (lambda m: download(m, args.raw))
    done, failed = [], []
    for n, k in enumerate(args.order):
        pid = f"lits_{k:03d}"
        if os.path.exists(os.path.join(args.out, pid, "meta.json")):
            print(f"{pid}: already converted, skipping", flush=True)
            continue
        used = used_gb(args.home)
        if used > args.max_used_gb:
            print(f"STOP: home uses {used:.1f} GB > {args.max_used_gb} GB; not processing {pid} or later", flush=True)
            break
        vol_name, msk_name = f"volume-{k}.nii", f"segmentation-{k}.nii"
        if vol_name not in files or msk_name not in files:
            print(f"{pid}: {vol_name} or {msk_name} not in part 1; skipping", flush=True)
            failed.append(pid)
            continue
        t0 = time.time()
        try:
            vol = fetch(files[vol_name])
            msk = fetch(files[msk_name])
            shutil.rmtree(os.path.join(args.out, pid), ignore_errors=True)   # a half-done patient starts clean
            meta = convert_patient(vol, msk, args.out, pid, args.I0, margin=args.margin, workers=args.workers, seed=k)
            # verify: counts match and every file loads
            n_slices = meta["kept_slices"]["count"]
            for kind in ("full", "quarter", "mask"):
                got = sorted(os.listdir(os.path.join(args.out, pid, kind)))
                assert len(got) == n_slices, f"{kind}: {len(got)} files, expected {n_slices}"
            ks = meta["kept_slices"]
            print(f"{pid}: OK {n_slices} slices (z {ks['first']}-{ks['last']}), liver added-noise std "
                  f"{meta['liver_added_noise_std_hu']['median']} HU, tumour voxels {meta['tumour_voxels']}, "
                  f"{time.time() - t0:.0f}s", flush=True)
            done.append(pid)
        except Exception as e:   # noqa: BLE001  (log and continue with the next patient)
            print(f"{pid}: FAILED: {e!r}", flush=True)
            failed.append(pid)
            shutil.rmtree(os.path.join(args.out, pid), ignore_errors=True)
        finally:
            for f in glob.glob(os.path.join(args.raw, "*")):
                os.remove(f) if os.path.isfile(f) else shutil.rmtree(f)
        if (n + 1) % 10 == 0:
            print(f"[disk] home uses {used_gb(args.home):.1f} GB after {n + 1} patients", flush=True)
    print(f"finished: {len(done)} converted, {len(failed)} failed {failed}; home uses {used_gb(args.home):.1f} GB", flush=True)


if __name__ == "__main__":
    main()

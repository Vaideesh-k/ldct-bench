"""Read single files out of a Kaggle dataset archive WITHOUT downloading the archive.

Kaggle answers 404 to per-file downloads (-f) for files in subfolders of the LiTS datasets
(tested on part 1 on 12 Nov and part 2 on 18 Nov 2026), and the whole archives (5.2 GB +
11.6 GB) do not fit in the home quota. The archive download URL redirects to a cloud-storage
URL that supports HTTP range requests, and a zip file has its table of contents at the end,
so zipfile can open it remotely and fetch only the bytes of the members it needs.

    python scripts/remote_zip.py --dataset andrewmvd/liver-tumor-segmentation-part-2 --list
    python scripts/remote_zip.py --dataset ... --get volume_pt6/volume-51.nii --out data/lits_raw_tmp

Credentials: ~/.kaggle/kaggle.json (read here, never printed or logged).
"""
import argparse
import io
import json
import os
import shutil
import time
import zipfile

import requests



def signed_url(dataset):
    """The short-lived storage URL the archive download redirects to (via the kaggle
    client itself, so authentication is exactly what `kaggle datasets download` does)."""
    from kaggle.api.kaggle_api_extended import KaggleApi
    from kagglesdk.datasets.types.dataset_api_service import ApiDownloadDatasetRequest
    api = KaggleApi()
    api.authenticate()
    owner, slug, version = api.split_dataset_string(dataset)
    with api.build_kaggle_client() as kaggle:
        req = ApiDownloadDatasetRequest()
        req.owner_slug, req.dataset_slug, req.dataset_version_number = owner, slug, version
        # the client's own call() follows the redirect and downloads the whole archive, so
        # build the same authenticated request and stop at the redirect
        http = kaggle.datasets.dataset_api_client._client
        http._init_session()
        prepared = http._prepare_request("datasets.DatasetApiService", "ApiDownloadDataset", req)
        resp = http._session.send(prepared, allow_redirects=False, stream=True, timeout=60)
        resp.close()
    if resp.status_code not in (301, 302, 303, 307, 308):
        raise RuntimeError(f"expected a redirect from Kaggle, got HTTP {resp.status_code}")
    url = resp.headers["Location"]
    return url


class RemoteFile(io.RawIOBase):
    """A read-only, seekable file over HTTP range requests (with retries and URL refresh)."""

    def __init__(self, dataset, block=8 << 20):
        self.dataset, self.block = dataset, block
        self.url = signed_url(dataset)
        self.session = requests.Session()
        r = self.session.get(self.url, headers={"Range": "bytes=0-0"}, timeout=60)
        if r.status_code != 206:
            raise RuntimeError(f"server does not support range requests (HTTP {r.status_code})")
        self.size = int(r.headers["Content-Range"].split("/")[1])
        self.pos = 0

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        self.pos = {0: offset, 1: self.pos + offset, 2: self.size + offset}[whence]
        return self.pos

    def _get(self, start, end):
        for attempt in range(5):
            try:
                r = self.session.get(self.url, headers={"Range": f"bytes={start}-{end}"}, timeout=120)
                if r.status_code == 206:
                    return r.content
                if r.status_code in (400, 401, 403):     # signed URL expired: ask Kaggle again
                    self.url = signed_url(self.dataset)
                    continue
                raise RuntimeError(f"HTTP {r.status_code}")
            except (requests.RequestException, RuntimeError) as e:
                if attempt == 4:
                    raise
                print(f"range request failed ({e}); retry {attempt + 1}", flush=True)
                time.sleep(10 * (attempt + 1))

    def read(self, n=-1):
        if self.pos >= self.size:
            return b""
        end = self.size - 1 if n is None or n < 0 else min(self.size, self.pos + n) - 1
        data = self._get(self.pos, end)
        self.pos += len(data)
        return data

    def readinto(self, b):
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


def open_archive(dataset):
    """zipfile.ZipFile over the remote archive (buffered so zipfile's small reads are cheap)."""
    raw = RemoteFile(dataset)
    return zipfile.ZipFile(io.BufferedReader(raw, buffer_size=raw.block)), raw.size


def extract(zf, member, out_dir):
    """Copy one member to out_dir/<basename>; returns the path."""
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, os.path.basename(member))
    with zf.open(member) as src, open(out + ".part", "wb") as dst:
        shutil.copyfileobj(src, dst, 8 << 20)
    os.replace(out + ".part", out)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--get", nargs="+", default=[])
    ap.add_argument("--out", default="data/lits_raw_tmp")
    args = ap.parse_args()
    zf, size = open_archive(args.dataset)
    print(f"archive {args.dataset}: {size / 1e9:.2f} GB, {len(zf.namelist())} members")
    if args.list:
        for i in zf.infolist():
            print(f"{i.file_size:>12d} {i.compress_size:>12d} {i.filename}")
    for m in args.get:
        t0 = time.time()
        p = extract(zf, m, args.out)
        print(f"{m} -> {p} ({os.path.getsize(p) / 1e6:.0f} MB, {time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()

"""Check prep_mayo.py using small fake DICOM files."""
import os
import subprocess
import sys
import tempfile

import numpy as np
from pydicom.dataset import Dataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, generate_uid

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def write_slice(path, hu, z):
    """Write one fake CT slice as a DICOM file, stored the way scanners do (raw = HU + 1024)."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.2"   # "CT image"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    raw = (hu + 1024).astype(np.uint16)
    ds.Rows, ds.Columns = raw.shape
    ds.SamplesPerPixel, ds.PhotometricInterpretation = 1, "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit, ds.PixelRepresentation = 16, 16, 15, 0
    ds.RescaleSlope, ds.RescaleIntercept = 1, -1024    # the "note" that converts raw -> HU
    ds.ImagePositionPatient = [0, 0, z]                # where the slice is in the body
    ds.SliceThickness = 1.0
    ds.PixelData = raw.tobytes()
    ds.save_as(path, enforce_file_format=True)


src, dst = tempfile.mkdtemp(), tempfile.mkdtemp()
for p in ["L067", "L506"]:
    for kind in ["quarter_1mm", "full_1mm"]:
        folder = os.path.join(src, p, f"{p}_{kind}")
        os.makedirs(folder)
        # files written OUT of body order on purpose: z = -30, -10, -20
        for i, z in enumerate([-30.0, -10.0, -20.0]):
            hu = np.full((64, 64), 40 + z, np.float32)   # pixel value reveals the slice position
            write_slice(os.path.join(folder, f"{p}_{i}.IMA"), hu, z)

subprocess.run([sys.executable, os.path.join(REPO, "scripts", "prep_mayo.py"),
                "--src", src, "--dst", dst], check=True)

values = [float(np.load(os.path.join(dst, "L067", "quarter", f"L067_{i:04d}.npy"))[0, 0]) for i in range(3)]
print("slice values in saved order:", values)
print("sorted by body position:", values == [10.0, 20.0, 30.0])
print("noisy and clean files match up:",
      sorted(os.listdir(os.path.join(dst, "L067", "quarter"))) == sorted(os.listdir(os.path.join(dst, "L067", "full"))))

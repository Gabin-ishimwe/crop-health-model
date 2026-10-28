"""Load a dataset zip made by pack_dataset (e.g. from Google Drive) into a local folder.

The zip is first copied to local disk in one go, which is fast even from a mounted
Google Drive, and then extracted. If the same zip was already extracted, nothing
is done, so this can safely run at the start of every training session.

Usage (from the repo root):
    python -m crop_health_model.data.unpack_dataset --src "/content/drive/MyDrive/crop-health/crop-data.zip"
"""

import argparse
import json
import os
import zipfile

from tqdm import tqdm

from crop_health_model.data.pack_dataset import MANIFEST_NAME

COPY_CHUNK = 64 * 1024 * 1024


def read_manifest(zip_path: str) -> dict:
    with zipfile.ZipFile(zip_path) as zf:
        return json.loads(zf.read(MANIFEST_NAME))


def copy_with_progress(src: str, dst: str) -> None:
    total = os.path.getsize(src)
    with open(src, "rb") as fin, open(dst, "wb") as fout, tqdm(
        total=total, unit="B", unit_scale=True, unit_divisor=1024, desc="Copying"
    ) as pbar:
        while chunk := fin.read(COPY_CHUNK):
            fout.write(chunk)
            pbar.update(len(chunk))


def unpack(src: str, data_dir: str) -> bool:
    """Extract the zip at src into data_dir. Returns False if it was already there."""
    manifest = read_manifest(src)
    local_manifest = os.path.join(data_dir, MANIFEST_NAME)
    if os.path.exists(local_manifest):
        with open(local_manifest) as f:
            if json.load(f) == manifest:
                print(f"{data_dir} already has this dataset ({manifest['created']})")
                return False

    os.makedirs(data_dir, exist_ok=True)
    local_zip = os.path.join(data_dir, "_dataset.zip.tmp")
    copy_with_progress(src, local_zip)

    with zipfile.ZipFile(local_zip) as zf:
        members = [m for m in zf.namelist() if m != MANIFEST_NAME]
        for member in tqdm(members, desc="Extracting", unit="file"):
            zf.extract(member, data_dir)

    os.remove(local_zip)

    # Written last, so an interrupted extraction is redone on the next run
    with open(local_manifest, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Extracted {manifest['images']} images into {data_dir}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--src", required=True, help="Path of the dataset zip")
    parser.add_argument("--data-dir", default=".data", help="Folder to extract into")
    args = parser.parse_args()
    unpack(args.src, args.data_dir)


if __name__ == "__main__":
    main()

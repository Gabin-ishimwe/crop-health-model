"""Pack the prepared dataset into a single zip file, e.g. in a Google Drive folder.

Reading thousands of small images straight from a mounted Google Drive is very slow,
so the prepared `.data` folder is shared as one uncompressed zip instead. Images are
already JPEG-compressed, so compressing them again would only cost time.

Usage (from the repo root, after make_dataset and make_annotations_file):
    python -m crop_health_model.data.pack_dataset --out "/content/drive/MyDrive/crop-health/crop-data.zip"
"""

import argparse
import json
import os
import zipfile
from datetime import datetime, timezone

from tqdm import tqdm

MANIFEST_NAME = "manifest.json"
# Leftovers from make_dataset that should not be shared
SKIP_SUFFIXES = (".zip", ".rar", ".tmp")
SKIP_DIRS = {".staging", "__MACOSX"}


def collect_files(data_dir: str) -> list[str]:
    """Return the paths (relative to data_dir) of all files to pack."""
    files = []
    for root, dirs, names in os.walk(data_dir):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
        for name in names:
            if name.endswith(SKIP_SUFFIXES) or name == MANIFEST_NAME:
                continue
            files.append(os.path.relpath(os.path.join(root, name), data_dir))
    return sorted(files)


def pack(data_dir: str, out_path: str, require_annotations: bool = True) -> dict:
    """Write every prepared file in data_dir to a zip at out_path.

    The zip is written under a temporary name and renamed when complete, so a
    half-written file is never picked up by Drive sync or unpack_dataset.
    """
    if require_annotations and not os.path.exists(
        os.path.join(data_dir, "annotations.csv")
    ):
        raise FileNotFoundError(
            f"No annotations.csv in {data_dir}. Run make_annotations_file first."
        )

    files = collect_files(data_dir)
    manifest = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "files": len(files),
        "images": sum(f.lower().endswith((".jpg", ".jpeg")) for f in files),
    }

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    tmp_path = out_path + ".tmp"
    with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr(MANIFEST_NAME, json.dumps(manifest, indent=2))
        for rel in tqdm(files, desc="Packing", unit="file"):
            zf.write(os.path.join(data_dir, rel), arcname=rel)
    os.replace(tmp_path, out_path)

    size_gb = os.path.getsize(out_path) / 1024**3
    print(f"Packed {manifest['images']} images into {out_path} ({size_gb:.2f} GB)")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, help="Path of the zip file to write")
    parser.add_argument("--data-dir", default=".data", help="Prepared data folder")
    args = parser.parse_args()
    pack(args.data_dir, args.out)


if __name__ == "__main__":
    main()

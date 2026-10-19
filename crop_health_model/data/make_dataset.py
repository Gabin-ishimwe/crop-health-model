import asyncio
import os
import shutil
import zipfile
from concurrent.futures import ProcessPoolExecutor

import httpx
import rarfile
from PIL import Image, ImageOps
from tqdm import tqdm as sync_tqdm
from tqdm.asyncio import tqdm as async_tqdm

from crop_health_model.data.metadata import all_datasets

harvard_dataverse_base_url = "https://dataverse.harvard.edu/api/access/datafile/"

DATA_DIR = ".data"
# Number of files downloaded at the same time. Dataverse times out if this is too high.
MAX_CONCURRENT_DOWNLOADS = 4
DOWNLOAD_RETRIES = 3
# The source photos are several megapixels. Training resizes to 256, so images are
# stored with their shorter side at this size to keep the dataset small on disk.
MAX_SHORT_SIDE = 512
# Delete each archive once its images are extracted. The full set of archives is ~95 GB.
DELETE_ARCHIVES = True


def is_extracted(file_path: str) -> bool:
    return os.path.exists(file_path + ".extracted")


async def download(
    client: httpx.AsyncClient, semaphore: asyncio.Semaphore, url: str, file_path: str
) -> None:
    """Download a file from a URL and save it to the specified path.

    The file is written to a temporary path and renamed when complete, so an
    interrupted download is never mistaken for a finished one.

    Args:
        client (httpx.AsyncClient): Shared HTTP client.
        semaphore (asyncio.Semaphore): Limits the number of concurrent downloads.
        url (str): URL to download the file from.
        file_path (str): Path to save the downloaded file.
    """
    folder, filename = os.path.split(file_path)
    os.makedirs(folder, exist_ok=True)

    if os.path.exists(file_path) or is_extracted(file_path):
        print(f"File already exists: {file_path}")
        return

    tmp_path = file_path + ".tmp"
    async with semaphore:
        for attempt in range(1, DOWNLOAD_RETRIES + 1):
            try:
                async with client.stream("GET", url, follow_redirects=True) as response:
                    if response.status_code != 200:
                        print(f"Failed to download {url}: {response.status_code}")
                        return
                    total_size = int(response.headers.get("Content-Length", 0))
                    with open(tmp_path, "wb") as f:
                        with async_tqdm(
                            total=total_size,
                            unit="B",
                            unit_scale=True,
                            unit_divisor=1024,
                            desc=filename,
                        ) as pbar:
                            async for chunk in response.aiter_bytes(
                                chunk_size=1024 * 1024
                            ):
                                f.write(chunk)
                                pbar.update(len(chunk))
                os.rename(tmp_path, file_path)
                return
            except httpx.HTTPError as e:
                print(f"Attempt {attempt} failed for {filename}: {e!r}")
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
        print(f"Giving up on {url}")


async def download_dataset(dataset: dict) -> None:
    """Download all files of one dataset."""
    semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)
    timeout = httpx.Timeout(120.0, connect=30.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        tasks = [
            download(
                client,
                semaphore,
                harvard_dataverse_base_url + id,
                os.path.join(DATA_DIR, dataset["folder"], fname),
            )
            for id, fname in dataset["ids"]
        ]
        await asyncio.gather(*tasks)


def resize_image(src: str, dst: str) -> None:
    """Save a copy of an image with its shorter side at most MAX_SHORT_SIDE."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    try:
        with Image.open(src) as img:
            img.draft("RGB", (MAX_SHORT_SIDE, MAX_SHORT_SIDE))
            img = ImageOps.exif_transpose(img).convert("RGB")
            scale = MAX_SHORT_SIDE / min(img.size)
            if scale < 1:
                new_size = (round(img.width * scale), round(img.height * scale))
                img = img.resize(new_size, Image.BILINEAR)
            img.save(dst, "JPEG", quality=90)
    except Exception as e:
        print(f"Skipping unreadable image {src}: {e}")


def extract(file_path: str) -> None:
    """Extract the images from a zip or rar archive and downscale them.

    The archive is unpacked into a staging folder, then each image is resized
    into the dataset folder, keeping its relative path (labels come from it).

    Args:
        file_path (str): Path to the archive file.
    """
    folder, filename = os.path.split(file_path)
    staging = os.path.join(folder, ".staging")
    shutil.rmtree(staging, ignore_errors=True)

    if file_path.endswith(".zip"):
        archive = zipfile.ZipFile(file_path, "r")
    else:
        archive = rarfile.RarFile(file_path, "r")

    with archive:
        images = [
            f for f in archive.namelist() if f.lower().endswith((".jpg", ".jpeg"))
        ]
        print(f"Unpacking {filename} ({len(images)} images)")
        archive.extractall(path=staging, members=images)

    with ProcessPoolExecutor() as pool:
        jobs = [
            pool.submit(resize_image, os.path.join(staging, f), os.path.join(folder, f))
            for f in images
        ]
        for job in sync_tqdm(jobs, desc=f"Resizing {filename}", unit="file"):
            job.result()

    shutil.rmtree(staging, ignore_errors=True)


def archive_parts(folder: str, file_name: str) -> list[str]:
    """Return every file belonging to an archive, including multi-part RAR volumes."""
    if ".part1." not in file_name:
        return [file_name]
    prefix = file_name.split(".part1.")[0] + ".part"
    return [
        f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(".rar")
    ]


def extract_dataset(dataset: dict) -> None:
    """Extract all archives of one dataset, marking each one done afterwards."""
    folder = os.path.join(DATA_DIR, dataset["folder"])
    for file_name in sorted(os.listdir(folder)):
        file_path = os.path.join(folder, file_name)
        if not os.path.isfile(file_path) or not file_name.endswith((".zip", ".rar")):
            continue
        # Multi-part RAR archives are unpacked through their first part,
        # which reads the other parts as well.
        if ".part" in file_name and ".part1." not in file_name:
            continue
        if is_extracted(file_path):
            continue
        try:
            extract(file_path)
        except Exception as e:
            print(f"Failed to extract {file_path}: {e!r}")
            continue
        for part in archive_parts(folder, file_name):
            part_path = os.path.join(folder, part)
            open(part_path + ".extracted", "w").close()
            if DELETE_ARCHIVES:
                os.remove(part_path)


def main() -> None:
    """Download and extract one dataset at a time to limit disk usage."""
    for dataset in all_datasets:
        print(f"\n=== {dataset['folder']} ===")
        asyncio.run(download_dataset(dataset))
        extract_dataset(dataset)


if __name__ == "__main__":
    main()

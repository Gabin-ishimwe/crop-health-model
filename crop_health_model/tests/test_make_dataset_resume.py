import os
import shutil

from crop_health_model.data import make_dataset

DATASET_A = {"folder": "maize-a", "ids": [("1", "MSV.zip")]}
DATASET_B = {"folder": "beans-b", "ids": [("2", "ALS.zip"), ("3", "HLT.zip")]}


def fake_pipeline(monkeypatch, failing: set[str]) -> list[str]:
    """Replace downloading and extraction with fakes. Archives in `failing` fail to extract."""
    downloaded = []

    async def fake_download(dataset):
        downloaded.append(dataset["folder"])

    def fake_extract(dataset):
        folder = os.path.join(make_dataset.DATA_DIR, dataset["folder"])
        for _, fname in dataset["ids"]:
            if fname in failing:
                continue
            img_dir = os.path.join(folder, fname.split(".")[0])
            os.makedirs(img_dir, exist_ok=True)
            with open(os.path.join(img_dir, "1.jpg"), "wb") as f:
                f.write(b"jpg")
            open(os.path.join(folder, fname + ".extracted"), "w").close()

    monkeypatch.setattr(make_dataset, "download_dataset", fake_download)
    monkeypatch.setattr(make_dataset, "extract_dataset", fake_extract)
    monkeypatch.setattr(make_dataset, "generate_all_annotations", lambda: None)
    return downloaded


def test_resume_from_parts(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    parts = str(tmp_path / "drive" / "crop-data-parts")

    # First run: B has an archive that fails to extract
    downloaded = fake_pipeline(monkeypatch, failing={"HLT.zip"})
    complete = make_dataset.download_and_prepare([DATASET_A, DATASET_B], parts)
    assert complete is False
    assert downloaded == ["maize-a", "beans-b"]
    assert os.path.exists(os.path.join(parts, "maize-a.zip"))
    # an incomplete dataset is never saved
    assert not os.path.exists(os.path.join(parts, "beans-b.zip"))

    # Colab disconnects: the local disk is wiped
    shutil.rmtree(".data")

    # Second run: A is restored from Drive, only B is downloaded again
    downloaded = fake_pipeline(monkeypatch, failing=set())
    complete = make_dataset.download_and_prepare([DATASET_A, DATASET_B], parts)
    assert complete is True
    assert downloaded == ["beans-b"]
    assert open(".data/maize-a/MSV/1.jpg", "rb").read() == b"jpg"
    assert os.path.exists(os.path.join(parts, "beans-b.zip"))


def test_parts_dir_next_to_zip():
    assert (
        make_dataset.parts_dir_for("/content/drive/MyDrive/crop-health/crop-data.zip")
        == "/content/drive/MyDrive/crop-health/crop-data-parts"
    )

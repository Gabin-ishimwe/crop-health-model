import os

import pytest

from crop_health_model.data.pack_dataset import pack
from crop_health_model.data.unpack_dataset import unpack


@pytest.fixture
def prepared(tmp_path):
    data = tmp_path / "data"
    (data / "maize-a" / "MSV").mkdir(parents=True)
    (data / "maize-a" / "MSV" / "1.jpg").write_bytes(b"jpg")
    (data / "maize-a" / "MSV" / "2.JPG").write_bytes(b"jpg")
    (data / "maize-a" / "MSV.zip.extracted").write_text("")
    (data / "annotations.csv").write_text("image,label\n")
    # leftovers that should not be packed
    (data / "maize-a" / "faw_01.rar").write_bytes(b"archive")
    (data / "maize-a" / "faw_02.zip.tmp").write_bytes(b"partial")
    (data / "maize-a" / ".staging").mkdir()
    (data / "maize-a" / ".staging" / "x.jpg").write_bytes(b"jpg")
    return data


def test_round_trip(prepared, tmp_path):
    out = tmp_path / "drive" / "crop-data.zip"
    manifest = pack(str(prepared), str(out))
    assert manifest["images"] == 2
    assert out.exists()

    dest = tmp_path / "local"
    assert unpack(str(out), str(dest)) is True
    assert (dest / "annotations.csv").exists()
    assert (dest / "maize-a" / "MSV" / "1.jpg").read_bytes() == b"jpg"
    assert (dest / "maize-a" / "MSV.zip.extracted").exists()
    assert not (dest / "maize-a" / "faw_01.rar").exists()
    assert not (dest / "maize-a" / "faw_02.zip.tmp").exists()
    assert not (dest / "maize-a" / ".staging").exists()
    assert not any(name.endswith(".tmp") for name in os.listdir(dest))

    # a second run with the same zip does nothing
    assert unpack(str(out), str(dest)) is False


def test_pack_requires_annotations(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(FileNotFoundError):
        pack(str(tmp_path / "data"), str(tmp_path / "out.zip"))

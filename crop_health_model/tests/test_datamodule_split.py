import pandas as pd
import pytest

from crop_health_model.datasets.datamodule import CropHealthDataModule
from crop_health_model.datasets.dataset import CropHealthDataset


@pytest.fixture
def annotations(tmp_path):
    """100 maize images from two sources and 20 beans images from one source."""
    rows = []
    for i in range(60):
        rows.append((f"maize-a/HLT/{i}.jpg", "HLT", "maize"))
    for i in range(20):
        rows.append((f"maize-a/MSV/{i}.jpg", "MSV", "maize"))
    for i in range(10):
        rows.append((f"maize-b/MSV/{i}.jpg", "MSV", "maize"))
    for i in range(10):
        rows.append((f"maize-b/FAW/{i}.jpg", "FAW", "maize"))
    for i in range(20):
        rows.append((f"beans-a/ALS/{i}.jpg", "ALS", "beans"))
    df = pd.DataFrame(rows, columns=["image", "label", "crop_type"])
    df["width"] = 512
    df["height"] = 512
    df.to_csv(tmp_path / "annotations.csv", index=False)
    return tmp_path


def make_datamodule(data_dir, **kwargs):
    dm = CropHealthDataModule(
        batch_size=4, task="multi-HLT", data_dir=str(data_dir), **kwargs
    )
    dm.setup("fit")
    return dm


def test_crop_filter(annotations):
    dataset = CropHealthDataset(
        annotations_file=str(annotations / "annotations.csv"),
        img_dir=str(annotations),
        task="multi-HLT",
        crops=["maize"],
    )
    assert len(dataset) == 100
    assert set(dataset.class_map) == {"HLT_maize", "MSV_maize", "FAW_maize"}
    assert set(dataset.data_df["source"]) == {"maize-a", "maize-b"}


def test_limit_samples_all_classes(annotations):
    dataset = CropHealthDataset(
        annotations_file=str(annotations / "annotations.csv"),
        img_dir=str(annotations),
        task="multi-HLT",
        limit=60,
    )
    assert len(dataset) == 60
    # the first 60 rows are all HLT_maize, a random sample is not
    assert len(dataset.class_map) > 1


def test_stratified_split(annotations):
    dm = make_datamodule(annotations, crops=["maize"])
    train, val, test = dm.split_indices(dm.data.data_df)

    assert len(train) + len(val) + len(test) == 100
    assert not set(train) & set(val)
    assert not set(train) & set(test)
    assert not set(val) & set(test)

    # every class is present in every split
    labels = dm.data.data_df["label"]
    for split in (train, val, test):
        assert set(labels[split]) == {"HLT_maize", "MSV_maize", "FAW_maize"}


def test_split_is_reproducible(annotations):
    first = make_datamodule(annotations, crops=["maize"])
    second = make_datamodule(annotations, crops=["maize"])
    assert first.split_indices(first.data.data_df) == second.split_indices(
        second.data.data_df
    )


def test_test_sources(annotations):
    dm = make_datamodule(annotations, crops=["maize"], test_sources=["maize-b"])
    train, val, test = dm.split_indices(dm.data.data_df)
    df = dm.data.data_df

    # only maize-b images of classes also seen in maize-a are tested (FAW is dropped)
    assert set(df.loc[test, "source"]) == {"maize-b"}
    assert set(df.loc[test, "label"]) == {"MSV_maize"}
    assert len(test) == 10
    assert set(df.loc[train + val, "source"]) == {"maize-a"}


def test_class_weights(annotations):
    dm = make_datamodule(annotations, crops=["maize"])
    weights = dm.compute_class_weights()
    hlt = dm.data.class_map["HLT_maize"]
    faw = dm.data.class_map["FAW_maize"]
    # the rare class gets a larger weight
    assert weights[faw] > weights[hlt]

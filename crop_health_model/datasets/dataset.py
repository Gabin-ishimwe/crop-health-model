import os
from typing import Callable

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, Subset


class CropHealthDataset(Dataset):
    """A dataset for the Crop Health dataset."""

    def __init__(
        self,
        annotations_file: str,
        img_dir: str,
        task: str,
        transform: Callable | None = None,
        target_transform: Callable | None = None,
        limit: int | None = None,
        crops: list[str] | None = None,
    ) -> None:
        self.data_df = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

        # The first folder of each image path is the dataset it came from,
        # e.g. "maize-dataset-uganda". Used for source-based test splits.
        self.data_df["source"] = self.data_df["image"].str.split("/").str[0]

        # if crops is set, keep only images of those crop types (one model per crop)
        if crops:
            self.data_df = self.data_df[self.data_df["crop_type"].isin(crops)]
            print(f"Using only crops: {crops}")

        # if limit is set, use a random sample of `limit` rows. The annotations file
        # is ordered by dataset, so taking the first rows would give only one class.
        if limit and limit < len(self.data_df):
            self.data_df = self.data_df.sample(n=limit, random_state=0)
            print(f"Using only {limit} rows")

        self.data_df = self.data_df.reset_index(drop=True)

        match task:
            case "binary":
                # binary classification with one healthy (HLT) class and one sick (NOT_HLT) class
                # map all non "HLT" classes to "NOT_HLT"
                mask = self.data_df["label"] != "HLT"
                self.data_df.loc[mask, "label"] = "NOT_HLT"
            case "single-HLT":
                # keep everything as is, i.e., several sick classes and one healthy class
                pass
            case "multi-HLT":
                # multiple healthy classes (one for each crop type) and multiple sick classes
                # combine "label" with "crop_type" to create a new label
                self.data_df["label"] = (
                    self.data_df["label"] + "_" + self.data_df["crop_type"]
                )
            case _:
                raise ValueError(f"Invalid task: {task}")

        # print number of distinct classes
        print(f"Number of distinct classes: {len(self.data_df['label'].unique())}")

        # Define a mapping from the class labels to integers using the unique method from pandas
        self.class_map = {
            label: idx for idx, label in enumerate(self.data_df["label"].unique())
        }

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.data_df)

    def __getitem__(self, idx) -> tuple:
        """Return the image and its label at the given index."""
        row = self.data_df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["image"])
        # open image with PIL
        image = Image.open(img_path).convert("RGB")
        label_name = row["label"]
        label = self.class_map[label_name]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_class_counts(self) -> dict:
        """Return the counts of each class in the dataset as a mapping from class index to count."""
        return self.data_df["label"].map(self.class_map).value_counts().to_dict()


class TransformWrapperDataset(Dataset):
    """A dataset that wraps another dataset and applies a transform to the data.

    Useful when performing data splitting on original dataset and then applying transforms
    on the split datasets.
    """

    def __init__(self, dataset: Dataset, transform: Callable | None = None) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        """Return the number of images in the dataset."""
        return len(self.dataset)

    def __getitem__(self, idx) -> tuple:
        """Return the transformed data and its label at the given index."""
        data, target = self.dataset[idx]
        if self.transform:
            data = self.transform(data)
        return data, target

    def get_class_counts(self) -> dict:
        """Return the counts of each class in the dataset as a mapping from class index to count."""
        # We need to handle the case where the underlying dataset is a subset of another dataset
        # or a subset of a subset (of a dataset)
        if isinstance(self.dataset, Subset):
            original_dataset = self.dataset.dataset
            if isinstance(original_dataset, Subset):
                original_dataset = original_dataset.dataset
            indices = self.dataset.indices

            # Filter the labels using the subset indices
            filtered_labels = original_dataset.data_df.iloc[indices]["label"]

            # Compute counts for the filtered labels
            class_counts = (
                filtered_labels.map(original_dataset.class_map).value_counts().to_dict()
            )

        else:
            class_counts = self.dataset.get_class_counts()

        return class_counts

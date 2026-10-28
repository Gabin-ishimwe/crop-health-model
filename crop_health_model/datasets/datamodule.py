import os

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

from crop_health_model.datasets.dataset import (
    CropHealthDataset,
    TransformWrapperDataset,
)


class CropHealthDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for the Crop Health Dataset."""

    def __init__(
        self,
        batch_size: int,
        task: str,
        data_dir: str = "crop_health_model/.data",
        annotations_file: str = "annotations.csv",
        data_split: tuple = (0.7, 0.15, 0.15),
        limit: int | None = None,
        crops: list[str] | None = None,
        test_sources: list[str] | None = None,
        split_seed: int = 42,
        num_workers: int = 8,
        train_transforms: list[torch.nn.Module] | None = None,
        test_transforms: list[torch.nn.Module] | None = None,
        normalization: transforms.Normalize | None = None,
    ) -> None:
        super(CropHealthDataModule, self).__init__()
        self.task = task
        self.limit = limit
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.annotations_file = annotations_file
        self.data_split = data_split
        self.crops = crops
        self.test_sources = test_sources
        self.split_seed = split_seed
        self.num_workers = num_workers

        if train_transforms is not None:
            transformations = train_transforms + [transforms.ToTensor()]
            if normalization is not None:
                transformations += [normalization]
            self.train_transform = transforms.Compose(transformations)
        else:
            self.train_transform = None

        if test_transforms is not None:
            transformations = test_transforms + [transforms.ToTensor()]
            if normalization is not None:
                transformations += [normalization]
            self.test_transform = transforms.Compose(transformations)
            self.val_transform = transforms.Compose(transformations)
        else:
            self.test_transform = None
            self.val_transform = None

    def prepare_data(self) -> None:
        # assumes data has been downloaded using make_dataset.py
        # and prepared using make_annotations_file.py
        pass

    def setup(self, stage: str | None = None) -> None:
        self.data = CropHealthDataset(
            annotations_file=os.path.join(self.data_dir, self.annotations_file),
            img_dir=self.data_dir,
            task=self.task,
            transform=None,  # apply transforms later
            limit=self.limit,
            crops=self.crops,
        )

        train_idx, val_idx, test_idx = self.split_indices(self.data.data_df)
        train_data = Subset(self.data, train_idx)
        val_data = Subset(self.data, val_idx)
        test_data = Subset(self.data, test_idx)
        print(
            f"Split sizes: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
        )

        # Wrap datasets with transforms
        train_data = TransformWrapperDataset(train_data, transform=self.train_transform)
        val_data = TransformWrapperDataset(val_data, transform=self.val_transform)
        test_data = TransformWrapperDataset(test_data, transform=self.test_transform)

        # Assign train/val datasets
        if stage == "fit" or stage is None:
            self.train_data = train_data
            self.val_data = val_data

        # Assign validation dataset
        if stage == "validate":
            self.val_data = val_data

        # Assign test dataset
        if stage == "test":
            self.test_data = test_data

        if stage == "predict":
            self.predict_data = test_data

    def split_indices(self, df: pd.DataFrame) -> tuple[list, list, list]:
        """Split row indices into train, validation and test sets.

        Each class is split separately with the same proportions (stratified), so
        rare classes appear in every set. A fixed seed keeps the split identical
        between the fit and test runs.

        If `test_sources` is set, all images from those datasets become the test set
        (for classes that also exist in the other datasets), and the rest is split
        into train and validation. This measures how the model does on photos from
        a source it has never seen, which is closer to real use.
        """
        rng = np.random.default_rng(self.split_seed)
        train_frac, val_frac, test_frac = self.data_split

        if self.test_sources:
            held_out = df["source"].isin(self.test_sources)
            seen_labels = set(df.loc[~held_out, "label"])
            test_mask = held_out & df["label"].isin(seen_labels)
            dropped = int((held_out & ~test_mask).sum())
            if dropped:
                print(f"Dropping {dropped} held-out images of classes not seen in training")
            test_idx = df.index[test_mask].tolist()
            remaining = df[~held_out]
            val_share = val_frac / (train_frac + val_frac)
            train_idx, val_idx = [], []
            for _, group in remaining.groupby("label"):
                idx = rng.permutation(group.index.to_numpy())
                n_val = round(len(idx) * val_share)
                val_idx += idx[:n_val].tolist()
                train_idx += idx[n_val:].tolist()
            return train_idx, val_idx, test_idx

        train_idx, val_idx, test_idx = [], [], []
        for _, group in df.groupby("label"):
            idx = rng.permutation(group.index.to_numpy())
            n_test = round(len(idx) * test_frac)
            n_val = round(len(idx) * val_frac)
            test_idx += idx[:n_test].tolist()
            val_idx += idx[n_test : n_test + n_val].tolist()
            train_idx += idx[n_test + n_val :].tolist()
        return train_idx, val_idx, test_idx

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def predict_dataloader(self) -> DataLoader:
        return DataLoader(
            self.predict_data, batch_size=self.batch_size, num_workers=self.num_workers
        )

    def compute_class_weights(self) -> torch.Tensor:
        """Compute class weights for imbalanced datasets."""
        class_counts = self.train_data.get_class_counts()
        n_samples = sum(class_counts.values())
        n_classes = len(self.data.class_map)
        n_present = len(class_counts)

        # Compute class weights inversely proportional to class counts.
        # A class with no training images gets weight 0.
        weights = [
            n_samples / (n_present * class_counts[i]) if class_counts.get(i) else 0.0
            for i in range(n_classes)
        ]
        class_weights = torch.tensor(weights, dtype=torch.float)

        return class_weights

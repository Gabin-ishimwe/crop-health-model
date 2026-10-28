import os

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchmetrics.functional import accuracy, confusion_matrix, fbeta_score


class LitModel(pl.LightningModule):
    """PyTorch Lightning module for training the Crop Health model."""

    def __init__(
        self,
        model: nn.Module,
    ) -> None:
        super(LitModel, self).__init__()
        self.model = model
        self.class_weights = None
        self.test_preds = []
        self.test_targets = []

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def on_fit_start(self) -> None:
        """Check that the model has one output per class in the data.

        The model hyperparameters are logged here instead of being saved in the
        checkpoint, because LightningCLI would otherwise try to read them back as
        command-line options when running `test` with --ckpt_path and fail.
        """
        if self.logger:
            self.logger.log_hyperparams(self.model.get_hyperparameters())
        n_data_classes = len(self.trainer.datamodule.data.class_map)
        if n_data_classes != self.model.num_classes:
            raise ValueError(
                f"Config has num_classes={self.model.num_classes} but the data has "
                f"{n_data_classes} classes: {list(self.trainer.datamodule.data.class_map)}"
            )

    def set_class_weights(self, class_weights: torch.Tensor) -> None:
        """Set the class weights for the loss function."""
        self.class_weights = class_weights

    def _compute_loss(
        self, logits: torch.Tensor, y: torch.Tensor, weights=None
    ) -> torch.Tensor:
        """Compute the loss function."""
        return F.cross_entropy(logits, y, weight=weights)

    def step_wrapper(
        self, batch: tuple, batch_idx: int, prefix: str, use_weights: bool = False
    ) -> torch.Tensor:
        """Wrapper for the training/validation/test step."""
        x, y = batch
        logits = self.model(x)
        if use_weights:
            loss = self._compute_loss(logits, y, self.class_weights)
        else:
            loss = self._compute_loss(logits, y)

        # logging of metrics
        preds = torch.argmax(logits, dim=-1)
        task = "binary" if self.model.num_classes == 2 else "multiclass"
        acc = accuracy(preds, y, task=task, num_classes=self.model.num_classes)
        f1 = fbeta_score(
            preds=preds,
            target=y,
            task=task,
            beta=1.0,
            num_classes=self.model.num_classes,
            average="macro",
        )
        self.log(f"{prefix}_loss", loss, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}_acc", acc, prog_bar=True, on_epoch=True, sync_dist=True)
        self.log(f"{prefix}_f1", f1, prog_bar=True, on_epoch=True, sync_dist=True)
        if prefix == "test":
            self.test_preds.append(preds.cpu())
            self.test_targets.append(y.cpu())
        return loss

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        loss = self.step_wrapper(batch, batch_idx, prefix="train", use_weights=True)

        # log the learning rate
        self.log(
            "learning_rate",
            self.trainer.optimizers[0].param_groups[0]["lr"],
            on_step=False,
            on_epoch=True,
            sync_dist=True,
        )

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step_wrapper(batch, batch_idx, prefix="val")

    def test_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        return self.step_wrapper(batch, batch_idx, prefix="test")

    def on_test_epoch_start(self) -> None:
        self.test_preds = []
        self.test_targets = []

    def on_test_epoch_end(self) -> None:
        """Report recall per class and a confusion matrix for the test set.

        Overall accuracy can hide a class the model is bad at. Recall per class shows
        the share of each disease's photos that were correctly recognised, and the
        confusion matrix shows which classes get mistaken for which.
        """
        preds = torch.cat(self.test_preds)
        targets = torch.cat(self.test_targets)
        num_classes = self.model.num_classes
        class_map = self.trainer.datamodule.data.class_map
        names = [name for name, _ in sorted(class_map.items(), key=lambda kv: kv[1])]

        cm = confusion_matrix(
            preds, targets, task="multiclass", num_classes=num_classes
        ).numpy()
        support = cm.sum(axis=1)
        correct = cm.diagonal()
        predicted = cm.sum(axis=0)

        rows = []
        for i, name in enumerate(names):
            recall = correct[i] / support[i] if support[i] else float("nan")
            precision = correct[i] / predicted[i] if predicted[i] else float("nan")
            rows.append(
                {
                    "class": name,
                    "images": int(support[i]),
                    "recall": recall,
                    "precision": precision,
                }
            )
            if support[i]:
                self.log(f"test_recall_{name}", float(recall))
        report = pd.DataFrame(rows)
        print("\nPer-class results on the test set:")
        print(report.to_string(index=False, float_format="%.3f"))

        cm_df = pd.DataFrame(cm, index=names, columns=names)
        cm_df.index.name = "true \\ predicted"

        fig, ax = plt.subplots(figsize=(1.2 * num_classes + 2, 1.2 * num_classes + 1))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(num_classes), names, rotation=45, ha="right")
        ax.set_yticks(range(num_classes), names)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        for i in range(num_classes):
            for j in range(num_classes):
                color = "white" if cm[i, j] > cm.max() / 2 else "black"
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", color=color)
        fig.tight_layout()

        log_dir = self.trainer.log_dir
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            report.to_csv(os.path.join(log_dir, "test_per_class.csv"), index=False)
            cm_df.to_csv(os.path.join(log_dir, "test_confusion_matrix.csv"))
            fig.savefig(os.path.join(log_dir, "test_confusion_matrix.png"), dpi=150)
            print(f"Saved test reports to {log_dir}")
        if self.logger and hasattr(self.logger.experiment, "add_figure"):
            self.logger.experiment.add_figure("Test confusion matrix", fig)
        plt.close(fig)

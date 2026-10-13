import torch
import torch.nn as nn
from torchvision import models


class ResNet(nn.Module):
    """ResNet model for image classification."""

    def __init__(
        self,
        num_classes: int = 17,
        num_layers: int = 18,
        weights: str | None = None,
    ) -> None:
        """
        Args:
            num_classes (int): The number of classes in the dataset,
                               which is also the number of output units.
            num_layers (int): The number of layers in the ResNet model.
            weights (str, optional): The weights to initialize the model with. Defaults to None.
        """
        super(ResNet, self).__init__()

        if num_layers == 18:
            self.resnet = models.resnet18(weights=weights)
        elif num_layers == 34:
            self.resnet = models.resnet34(weights=weights)
        elif num_layers == 50:
            self.resnet = models.resnet50(weights=weights)
        elif num_layers == 101:
            self.resnet = models.resnet101(weights=weights)
        elif num_layers == 152:
            self.resnet = models.resnet152(weights=weights)
        else:
            raise ValueError("Invalid number of layers: {}".format(num_layers))

        # Replace the last fully connected layer of the model
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, num_classes)
        self.num_classes = num_classes
        self.num_layers = num_layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model."""
        return self.resnet(x)

    def get_hyperparameters(self) -> dict:
        """Return the hyperparameters of the model."""
        return {
            "num_classes": self.num_classes,
            "num_layers": self.num_layers,
        }

"""Predict the health class of leaf images with a trained model, without TorchServe.

Usage (from the repo root):
    python crop_health_model/scripts/predict.py --run runs/maize/version_0 leaf1.jpg leaf2.jpg

--run is a training run folder, holding index_to_name.json and checkpoints/best_model.pt.
"""

import argparse
import json
import os

import torch
from PIL import Image
from torchvision import transforms

from crop_health_model.models.model import ResNet

# Same preprocessing as the test set in the training configs
PREPARE = transforms.Compose(
    [
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def load_model(run_dir: str) -> tuple[torch.nn.Module, dict]:
    """Load the best weights of a training run and its class names."""
    with open(os.path.join(run_dir, "index_to_name.json")) as f:
        names = json.load(f)
    model = ResNet(num_classes=len(names), num_layers=18)
    weights = os.path.join(run_dir, "checkpoints", "best_model.pt")
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model, names


def predict(model: torch.nn.Module, names: dict, image_path: str) -> dict:
    """Return the probability of each class, highest first."""
    x = PREPARE(Image.open(image_path).convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    scores = {names[str(i)]: round(p.item(), 4) for i, p in enumerate(probs)}
    return dict(sorted(scores.items(), key=lambda kv: kv[1], reverse=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run", required=True, help="Training run folder")
    parser.add_argument("images", nargs="+", help="Image files to classify")
    args = parser.parse_args()

    model, names = load_model(args.run)
    for image_path in args.images:
        scores = predict(model, names, image_path)
        top, prob = next(iter(scores.items()))
        print(f"{image_path}: {top} ({prob:.0%})")
        print(f"  {json.dumps(scores)}")


if __name__ == "__main__":
    main()

from torchvision import transforms
from ts.torch_handler.image_classifier import ImageClassifier


class CustomHandler(ImageClassifier):
    topk = 17
    image_processing = transforms.Compose(
        [
            transforms.Resize(interpolation=2, size=256),
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

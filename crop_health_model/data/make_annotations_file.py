import os

import pandas as pd
import PIL
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from crop_health_model.data.metadata import all_datasets

# Define the transformation to apply to the images
# This transformation is similar to the one used in the training script
# Images that throw an error during transformation won't be included in the annotations file
transform = transforms.Compose(
    [
        transforms.Resize(256, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def generate_annotations(
    img_dir: str,
    keywords_to_avoid: list[str],
    annotations_file_path: str,
    classes: list[dict],
    total_image_count: int,
    crop_type: str,
) -> pd.DataFrame:
    """Generate annotations for a dataset.

    Args:
        img_dir (str): Path to the directory containing the images.
        keywords_to_avoid (list[str]): List of keywords to avoid in the image paths.
        annotations_file_path (str): Path to save the annotations file.
        classes (list[dict]): List of classes with raw and clean names.
        total_image_count (int): Total number of images expected in the dataset.
        crop_type (str): Crop type for the dataset.

    Returns:
        pd.DataFrame: DataFrame containing the annotations.
    """
    img_paths = []
    labels = []
    heights = []
    widths = []

    print(f"Generating annotations for {img_dir}")

    # Loop through all the subdirectories in the img_dir
    # But make sure to only add images to the img_paths list
    for root, dirs, files in os.walk(img_dir):
        for file in tqdm(files):
            if file.endswith((".jpg", ".jpeg")) and not any(
                keyword in root for keyword in keywords_to_avoid
            ):
                path = os.path.join(os.path.relpath(root, ".data"), file)

                # Try to open the image, if it fails, skip it
                try:
                    img = Image.open(os.path.join(".data", path))
                    # dimension
                    width, height = img.size
                except PIL.UnidentifiedImageError as e:
                    print(f"Error opening image: {path}, {e}")
                    continue

                # Try to apply the transformation, if it fails, skip it
                try:
                    img = transform(img)
                except OSError as e:
                    print(f"Error transforming image: {path}, {e}")
                    continue

                img_paths.append(path)
                heights.append(height)
                widths.append(width)

                # Get the label from the directory name
                identified_class_count = 0
                for class_ in classes:
                    if class_["raw"] in root:
                        identified_class_count += 1
                        labels.append(class_["clean"])
                if identified_class_count == 0:
                    print(f"Could not identify class for {path}")
                if identified_class_count > 1:
                    print(f"Multiple classes identified for {path}")

    # Verify that the number of labels matches the number of images
    print(
        f"Expected {total_image_count} images, found {len(img_paths)} images and {len(labels)} labels"
    )

    # Verify that the number of class labels is correct
    for class_ in classes:
        # Skip classes with count -1 as defined in the metadata
        if class_["count"] == -1:
            continue
        print(
            f"Expected {class_['count']} images for class {class_['clean']}, found {labels.count(class_['clean'])}"
        )

    # Create a list of crop types
    crop_types = [crop_type] * len(img_paths)

    # Create a DataFrame from the img_paths and labels lists
    df = pd.DataFrame(
        list(zip(img_paths, widths, heights, labels, crop_types)),
        columns=["image", "width", "height", "label", "crop_type"],
    )

    # Save the DataFrame to a CSV file
    df.to_csv(annotations_file_path, index=False)

    return df


def generate_all_annotations() -> None:
    """Generate annotations for all datasets."""
    all_dfs = []
    for dataset in all_datasets:
        img_dir = os.path.join(".data", dataset["folder"])
        # Datasets not downloaded yet (e.g. after make_dataset --datasets) are skipped
        if not os.path.isdir(img_dir):
            print(f"Skipping {dataset['folder']}: not downloaded")
            continue
        keywords_to_avoid = ["MACOSX"]
        annotations_file_path = os.path.join(
            ".data", dataset["folder"], "annotations.csv"
        )
        df = generate_annotations(
            img_dir=img_dir,
            keywords_to_avoid=keywords_to_avoid,
            annotations_file_path=annotations_file_path,
            classes=dataset["classes"],
            total_image_count=dataset["total_image_count"],
            crop_type=dataset["crop_type"],
        )
        print()
        all_dfs.append(df)
    if not all_dfs:
        raise FileNotFoundError("No datasets found in .data. Run make_dataset first.")
    all_df = pd.concat(all_dfs)
    all_df.to_csv(os.path.join(".data", "annotations.csv"), index=False)
    print("All annotations saved to .data/annotations.csv")


if __name__ == "__main__":
    generate_all_annotations()

"""Data classes and functions."""
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import app.utils as ut


class ImageDataset(Dataset):
    """Custom Image Dataset Class."""

    def __init__(self, images, labels):
        """Initialize ImageDataset Class."""
        self.images = images
        self.labels = labels
        self.labels = self.get_label_encoding()

    def __getitem__(self, index):
        """Gets an item from dataset."""
        return self.images[index], self.labels[index]

    def __len__(self):
        """Returns length of dataset."""
        return len(self.images)

    def get_n_labels(self) -> int:
        """Calculates how many unique labels are in this set."""
        lab_set = set()
        for label in self.labels:
            if label in lab_set:
                continue
            else:
                lab_set.add(label)
        return len(lab_set)

    def get_label_encoding(self) -> list[int]:
        """Gets label encodings for a given label."""
        n_labels = self.get_n_labels()

        lab_set = set()
        base_encoding = [0] * n_labels
        encoded_labels = []
        encodings = {}
        count = 0
        for label in self.labels:
            if label in lab_set:
                encoded_labels.append(encodings[label])
            else:
                lab_set.add(label)
                encoding = base_encoding
                encoding[count] = 1
                encodings[label] = encoding
                encoded_labels.append(encoding)
                count += 1

        return encoded_labels


def load_data() -> dict:
    """Loads data, kindof a bad function in this state."""
    lbl_file = "./data/labels.txt"
    files, labels = ut.get_files_and_labels(lbl_file)
    n_classes = ut.get_n_classes(labels)

    file_dir = "./data/images"
    images = []
    for _, file in enumerate(files):
        file_path = f"{file_dir}/{file}"
        image = Image.open(file_path).convert("RGB")
        image.load()
        data = np.asarray(image, dtype="float32")
        data = np.moveaxis(data, -1, 0)
        images.append(data)
    print(f"Loaded {len(files)} samples")
    print(
        "Image Dimensions",
        f"""Height: {data.shape[1]} px,
        Width: {data.shape[2]} px, Channels: {data.shape[0]}""",
    )

    return {"images": images, "labels": labels, "n_classes": n_classes}

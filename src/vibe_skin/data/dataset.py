import io
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset


def load_image(data: bytes | str | Path) -> np.ndarray:
    """Decode an image into an RGB array, applying the EXIF rotation that phone cameras use.
    Training, evaluation and the API all read images through this function."""
    image = Image.open(io.BytesIO(data) if isinstance(data, bytes) else data)
    image = ImageOps.exif_transpose(image)
    return np.asarray(image.convert("RGB"))


class LesionDataset(Dataset):
    """Images and integer labels from a table with `path` and `label` columns."""

    def __init__(self, df: pd.DataFrame, transform):
        self.paths = df["path"].astype(str).tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, int]:
        return self.transform(image=load_image(self.paths[i]))["image"], self.labels[i]
